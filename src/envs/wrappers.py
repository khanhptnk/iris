"""
Credits to https://github.com/openai/baselines/blob/master/baselines/common/atari_wrappers.py
"""

import os
from typing import Tuple, Optional
import json
import random
import numpy as np

import messenger
from messenger.envs.config import NPCS
import gym
import numpy as np
from PIL import Image
from hydra.utils import get_original_cwd, to_absolute_path
from pathlib import Path


def make_messenger(id, max_episode_steps=None, split=None, seed=None):
    env = gym.make(id, shuffle_obs=False)
    if max_episode_steps is not None:
        env = TimeLimitEnv(env, max_episode_steps=max_episode_steps)
    env = MessengerSplitEnv(env, split, seed=seed)
    return env


def make_atari(
    id,
    size=64,
    max_episode_steps=None,
    noop_max=30,
    frame_skip=4,
    done_on_life_loss=False,
    clip_reward=False,
):
    env = gym.make(id)
    assert "NoFrameskip" in env.spec.id or "Frameskip" not in env.spec
    env = ResizeObsWrapper(env, (size, size))
    if clip_reward:
        env = RewardClippingWrapper(env)
    if max_episode_steps is not None:
        env = gym.wrappers.TimeLimit(env, max_episode_steps=max_episode_steps)
    if noop_max is not None:
        env = NoopResetEnv(env, noop_max=noop_max)
    env = MaxAndSkipEnv(env, skip=frame_skip)
    if done_on_life_loss:
        env = EpisodicLifeEnv(env)
    return env


class ResizeObsWrapper(gym.ObservationWrapper):
    def __init__(self, env: gym.Env, size: Tuple[int, int]) -> None:
        gym.ObservationWrapper.__init__(self, env)
        self.size = tuple(size)
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(size[0], size[1], 3), dtype=np.uint8
        )
        self.unwrapped.original_obs = None

    def resize(self, obs: np.ndarray):
        img = Image.fromarray(obs)
        img = img.resize(self.size, Image.BILINEAR)
        return np.array(img)

    def observation(self, observation: np.ndarray) -> np.ndarray:
        self.unwrapped.original_obs = observation
        return self.resize(observation)


class RewardClippingWrapper(gym.RewardWrapper):
    def reward(self, reward):
        return np.sign(reward)


class NoopResetEnv(gym.Wrapper):
    def __init__(self, env, noop_max=30):
        """Sample initial states by taking random number of no-ops on reset.
        No-op is assumed to be action 0.
        """
        gym.Wrapper.__init__(self, env)
        self.noop_max = noop_max
        self.override_num_noops = None
        self.noop_action = 0
        assert env.unwrapped.get_action_meanings()[0] == "NOOP"

    def reset(self, **kwargs):
        """Do no-op action for a number of steps in [1, noop_max]."""
        self.env.reset(**kwargs)
        if self.override_num_noops is not None:
            noops = self.override_num_noops
        else:
            noops = self.unwrapped.np_random.randint(1, self.noop_max + 1)
        assert noops > 0
        obs = None
        for _ in range(noops):
            obs, _, done, _ = self.env.step(self.noop_action)
            if done:
                obs = self.env.reset(**kwargs)
        return obs

    def step(self, action):
        return self.env.step(action)


class EpisodicLifeEnv(gym.Wrapper):
    def __init__(self, env):
        """Make end-of-life == end-of-episode, but only reset on true game over.
        Done by DeepMind for the DQN and co. since it helps value estimation.
        """
        gym.Wrapper.__init__(self, env)
        self.lives = 0
        self.was_real_done = True

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        self.was_real_done = done
        # check current lives, make loss of life terminal,
        # then update lives to handle bonus lives
        lives = self.env.unwrapped.ale.lives()
        if lives < self.lives and lives > 0:
            # for Qbert sometimes we stay in lives == 0 condition for a few frames
            # so it's important to keep lives > 0, so that we only reset once
            # the environment advertises done.
            done = True
        self.lives = lives
        return obs, reward, done, info

    def reset(self, **kwargs):
        """Reset only when lives are exhausted.
        This way all states are still reachable even though lives are episodic,
        and the learner need not know about any of this behind-the-scenes.
        """
        if self.was_real_done:
            obs = self.env.reset(**kwargs)
        else:
            # no-op step to advance from terminal/lost life state
            obs, _, _, _ = self.env.step(0)
        self.lives = self.env.unwrapped.ale.lives()
        return obs


class MaxAndSkipEnv(gym.Wrapper):
    def __init__(self, env, skip=4):
        """Return only every `skip`-th frame"""
        gym.Wrapper.__init__(self, env)
        assert skip > 0
        # most recent raw observations (for max pooling across time steps)
        self._obs_buffer = np.zeros((2,) + env.observation_space.shape, dtype=np.uint8)
        self._skip = skip
        self.max_frame = np.zeros(env.observation_space.shape, dtype=np.uint8)

    def step(self, action):
        """Repeat action, sum reward, and max over last observations."""
        total_reward = 0.0
        done = None
        for i in range(self._skip):
            obs, reward, done, info = self.env.step(action)
            if i == self._skip - 2:
                self._obs_buffer[0] = obs
            if i == self._skip - 1:
                self._obs_buffer[1] = obs
            total_reward += reward
            if done:
                break
        # Note that the observation on the done=True frame
        # doesn't matter
        self.max_frame = self._obs_buffer.max(axis=0)

        return self.max_frame, total_reward, done, info

    def reset(self, **kwargs):
        return self.env.reset(**kwargs)


class TimeLimitEnv(gym.Wrapper):
    def __init__(
        self,
        env: gym.Env,
        max_episode_steps: Optional[int] = None,
    ):
        super().__init__(env)
        if max_episode_steps is None and self.env.spec is not None:
            max_episode_steps = env.spec.max_episode_steps
        if self.env.spec is not None:
            self.env.spec.max_episode_steps = max_episode_steps
        self._max_episode_steps = max_episode_steps
        self._elapsed_steps = None

    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        self._elapsed_steps += 1

        if self._elapsed_steps >= self._max_episode_steps:
            self.truncated = True

        return observation, reward, done, info

    def reset(self, **kwargs):
        self._elapsed_steps = 0
        self.truncated = False
        return self.env.reset(**kwargs)


class MessengerSplitEnv(gym.Wrapper):
    INTENTIONS = ["random", "suicide", "survive", "get_message", "go_to_goal"]

    def __init__(self, env, split, seed=None):
        gym.Wrapper.__init__(self, env)
        with open(Path(get_original_cwd()) / "src/envs/messenger_splits.json") as f:
            splits = json.load(f)
        self.split = split
        self.games = splits[split]
        self.entity_ids = {entity.name: entity.id for entity in NPCS}
        self.random = np.random.RandomState(seed + 543)

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        obs = self.wrap_obs(obs, self.entity_order)
        return obs, reward, done, info

    def reset(self):
        # This is just for setting a goal for the rule-based policy
        self.intention = self.random.choice(self.INTENTIONS)

        # entities = self.games[self.random.randint(0, len(self.games))]
        entities = self.games[0]
        obs, self.manual, self.ground_truth = self.env.reset(
            split=self.split, entities=entities
        )
        self.entity_order = self.random.permutation(3)
        obs = self.wrap_obs(obs, self.entity_order)
        return obs

    def wrap_obs(self, obs, entity_order):
        obs["entities"] = obs["entities"][..., entity_order]
        return np.concatenate((obs["entities"], obs["avatar"]), axis=-1)
