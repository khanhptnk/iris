#!/usr/bin/bash

fps=15
header=1
reconstruction=0
save_mode=0
mode="agent_in_env"

case $1 in
    -f | --fps )
        shift
        fps=$1
        ;;
    -h | --header )
        header=1
        ;;
    -r | --reconstruction )
        reconstruction=1
        ;;
    -s | --save-mode )
        save_mode=1
        ;;
    -a | --agent-world-model )
        mode="agent_in_world_model"
        ;;
    -e | --episode )
        mode="episode_replay"
        ;;
    -w | --world-model )
        mode="play_in_world_model"
        ;;
    -p | --world-model )
        mode="play_in_env"
        ;;
    * )
        echo Invalid usage : $1
        exit 1
esac

exp_dir=$2

if [ "$3" != ""  ]; then
  seed=$3
else
  seed=0
fi


env=messenger

echo $exp_dir
echo $seed
echo $mode

python3 src/play.py exp_dir=${exp_dir} +mode="${mode}" +fps="${fps}" +header="${header}" +reconstruction="${reconstruction}" +save_mode="${save_mode}" env=${env} tokenizer=${env} actor_critic=${env} world_model=${env}_dummy common.seed=${seed}
