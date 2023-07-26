#!/bin/bash

#rm outputs/single_env_diverse_intentions && python src/main.py exp_dir=single_env_diverse_intentions_run2 env=messenger tokenizer=messenger actor_critic=messenger world_model=messenger_dummy common.device=cuda:0 collection.train.config.num_steps=1000 training.world_model.start_after_epochs=5 training.world_model.batch_num_samples=8 training.tokenizer.start_after_epochs=1000 training.world_model.grad_acc_steps=4 training.actor_critic.start_after_epochs=1000 wandb.mode=online agent._target_=agent.MessengerRuleBasedAgent

cd ..

rm outputs/single_env_diverse_intentions && \
  python src/main.py \
    exp_dir=single_env_diverse_intentions \
    env=messenger \
    tokenizer=messenger \
    actor_critic=messenger \
    world_model=messenger_dummy \
    common.device=cuda:0 \
    collection.train.config.num_steps=1000 \
    training.world_model.start_after_epochs=5 \
    training.world_model.batch_num_samples=8 \
    training.tokenizer.start_after_epochs=1000 \
    training.world_model.grad_acc_steps=4 \
    training.actor_critic.start_after_epochs=1000 \
    wandb.mode=online \
    agent._target_=agent.MessengerRuleBasedAgent
