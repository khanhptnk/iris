import random
from pathlib import Path
from typing import Union

import torch
from torch.distributions.categorical import Categorical
import torch.nn as nn

from models.actor_critic import ActorCritic
from models.tokenizer import Tokenizer
from models.world_model import WorldModel
from utils import extract_state_dict
from envs import SingleProcessEnv, MultiProcessEnv


class Agent(nn.Module):
    def __init__(
        self, tokenizer: Tokenizer, world_model: WorldModel, actor_critic: ActorCritic, seed: int = None
    ):
        super().__init__()
        self.tokenizer = tokenizer
        self.world_model = world_model
        self.actor_critic = actor_critic
        self.random = random.Random(seed + 576)

    @property
    def device(self):
        return self.actor_critic.net.device

    def load(
        self,
        path_to_checkpoint: Path,
        device: torch.device,
        load_tokenizer: bool = True,
        load_world_model: bool = True,
        load_actor_critic: bool = True,
    ) -> None:
        agent_state_dict = torch.load(path_to_checkpoint, map_location=device)
        if load_tokenizer:
            self.tokenizer.load_state_dict(
                extract_state_dict(agent_state_dict, "tokenizer")
            )
        if load_world_model:
            self.world_model.load_state_dict(
                extract_state_dict(agent_state_dict, "world_model")
            )
        if load_actor_critic:
            self.actor_critic.load_state_dict(
                extract_state_dict(agent_state_dict, "actor_critic")
            )

    def act(
        self,
        obs: torch.FloatTensor,
        should_sample: bool = True,
        temperature: float = 1.0,
        env: Union[SingleProcessEnv, MultiProcessEnv] = None
    ) -> torch.LongTensor:
        input_ac = (
            obs
            if self.actor_critic.use_original_obs
            else torch.clamp(
                self.tokenizer.encode_decode(
                    obs, should_preprocess=True, should_postprocess=True
                ),
                0,
                1,
            )
        )
        logits_actions = self.actor_critic(input_ac).logits_actions[:, -1] / temperature
        act_token = (
            Categorical(logits=logits_actions).sample()
            if should_sample
            else logits_actions.argmax(dim=-1)
        )
        return act_token


class MessengerRuleBasedAgent(Agent):

    WITH_MESSAGE_ID = 16
    CUSTOM_TO_MESSENGER_ENTITY = {
        "robot": "robot",
        "airplane": "airplane",
        "thief": "thief",
        "scientist": "scientist",
        "queen": "queen",
        "ship": "ship",
        "dog": "dog",
        "bird": "bird",
        "fish": "fish",
        "mage": "mage",
        "orb": "ball",
        "sword": "sword",
    }
    all_actions = [(0, -1, 0), (1, 1, 0), (2, 0, -1), (3, 0, 1), (4, 0, 0)]

    def act(
        self,
        obs: torch.FloatTensor,
        should_sample: bool = True,
        temperature: float = 1.0,
        env: Union[SingleProcessEnv, MultiProcessEnv] = None
    ) -> torch.LongTensor:

        device = obs.device
        obs = obs.permute(0, 2, 3, 1).cpu().numpy()
        self.ENTITY_IDS = env.get_attr('entity_ids')[0]
        parsed_manuals = env.get_attr('ground_truth')
        intentions = env.get_attr('intention')

        actions = []
        for i in range(obs.shape[0]):
            actions.append(self.compute_action(obs[i], parsed_manuals[i], intentions[i]))
        actions = torch.tensor(actions).to(device)

        #for e in parsed_manuals[0]:
        #    print(self.ENTITY_IDS[e[0]], e[1], e[2])
        #print(intentions, actions)

        return actions

    def compute_action(self, obs, parsed_manuals, intention):

        if intention == 'random':
            return self.random.choice(range(5))

        if intention == 'survive':
            avatar_id = self.get_avatar_id(obs)
            a_pos = self.get_position_by_id(obs, avatar_id)
            enemy_id = self.get_entity_id_by_role(parsed_manuals, 'enemy')
            e_pos = self.get_position_by_id(obs, enemy_id)
            goal_id = self.get_entity_id_by_role(parsed_manuals, 'goal')
            g_pos = self.get_position_by_id(obs, goal_id)
            # if messaged has been obtained, don't care about hitting goal
            if avatar_id == self.WITH_MESSAGE_ID:
                g_pos = None
            # choose action that takes avatar furthest from the enemy
            return self.get_best_action_for_surviving(a_pos, e_pos, g_pos)

        if intention == 'get_message':
            avatar_id = self.get_avatar_id(obs)
            # if message has been obtained, act randomly
            if avatar_id == self.WITH_MESSAGE_ID:
                return self.compute_action(obs, parsed_manuals, 'random')
            a_pos = self.get_position_by_id(obs, avatar_id)
            message_id = self.get_entity_id_by_role(parsed_manuals, 'message')
            t_pos = self.get_position_by_id(obs, message_id)
            # choose action that takes avatar closest to the goal
            return self.get_best_action_for_chasing(a_pos, t_pos)

        if intention == 'go_to_goal':
            avatar_id = self.get_avatar_id(obs)
            a_pos = self.get_position_by_id(obs, avatar_id)
            # if message has been obtained, go to goal
            if avatar_id == self.WITH_MESSAGE_ID:
                goal_id = self.get_entity_id_by_role(parsed_manuals, 'goal')
                t_pos = self.get_position_by_id(obs, goal_id)
            # else go to message
            else:
                message_id = self.get_entity_id_by_role(parsed_manuals, 'message')
                t_pos = self.get_position_by_id(obs, message_id)
            # choose action that takes avatar closest to the goal or message
            return self.get_best_action_for_chasing(a_pos, t_pos)

        return None

    def get_best_action_for_chasing(self, a_pos, t_pos):
        best_d = 1e9
        best_a = None
        # shuffle action order to randomize choice
        for a, dr, dc in random.sample(self.all_actions, len(self.all_actions)):
            na_pos = (a_pos[0] + dr, a_pos[1] + dc)
            if self.out_of_bounds(na_pos):
                continue
            d = self.get_distance(na_pos, t_pos)
            if d < best_d:
                best_d = d
                best_a = a
        return best_a

    def get_best_action_for_surviving(self, a_pos, e_pos, g_pos):
        distance_to_enemy = self.get_distance(a_pos, e_pos)
        if g_pos is not None:
            distance_to_goal = self.get_distance(a_pos, g_pos)
        else:
            distance_to_goal = 1e9
        # if far enough from enemy and goal just act randomly
        SAFE_DISTANCE = 6
        if distance_to_enemy >= SAFE_DISTANCE and distance_to_goal >= SAFE_DISTANCE:
            return random.choice(range(len(self.all_actions)))
        # otherwise, stay further from both
        best_d = -1e9
        best_a = None
        # shuffle action order to randomize choice
        for a, dr, dc in random.sample(self.all_actions, len(self.all_actions)):
            na_pos = (a_pos[0] + dr, a_pos[1] + dc)
            if self.out_of_bounds(na_pos):
                continue
            d = self.get_distance(na_pos, e_pos)
            if g_pos is not None:
                d = min(d, self.get_distance(na_pos, g_pos))
            if d >= SAFE_DISTANCE / 2 or d > best_d:
                best_d = d
                best_a = a

        return best_a

    def get_avatar_id(self, obs):
        return obs[..., -1].max()

    def get_entity_id_by_role(self, parsed_manuals, role):
        for e in parsed_manuals:
            if e[2] == role:
                return self.ENTITY_IDS[self.CUSTOM_TO_MESSENGER_ENTITY[e[0]]]
        return None

    def get_position_by_id(self, obs, id):
        entity_ids = obs.reshape(100, -1).max(0).tolist()
        c = entity_ids.index(id)
        pos = obs.reshape(100, -1)[:, c].tolist().index(id)
        row = pos // 10
        col = pos % 10
        return row, col

    def out_of_bounds(self, x):
        return x[0] < 0 or x[0] >= 10 or x[1] < 0 or x[1] >= 10

    def get_distance(self, x, y):
        return abs(x[0] - y[0]) + abs(x[1] - y[1])

