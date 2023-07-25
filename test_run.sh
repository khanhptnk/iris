#!/bin/bash

rm outputs/fun && python src/main.py exp_dir=fun env=messenger tokenizer=messenger actor_critic=messenger world_model=messenger_dummy common.device=cuda:0 actor_critic.use_original_obs=True training.world_model.start_after_epochs=5 training.world_model.batch_num_samples=1 training.actor_critic.start_after_epochs=5 wandb.mode=online
