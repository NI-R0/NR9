#!/bin/bash
   

python main.py --env_domain=hipp_walker --env_task=stand --run_name=hipp_stand_baseline \
    --episodes=1000 --eval_frequency=20 --num_envs=5 \
    --capacity=1000000 --warmup=5000 --lr=1e-4 --target_update_period=100

python main.py --env_domain=hipp_walker --env_task=stand --run_name=hipp_stand_sgd16 \
    --episodes=1000 --eval_frequency=20 --num_envs=5 \
    --capacity=1000000 --warmup=5000 --lr=1e-4 --target_update_period=100 \
    --sgd_steps_per_learner_step=16

python main.py --env_domain=hipp_walker --env_task=stand --run_name=hipp_stand_nstep3 \
    --episodes=1000 --eval_frequency=20 --num_envs=5 \
    --capacity=1000000 --warmup=5000 --lr=1e-4 --target_update_period=100 \
    --n_step=3

python main.py --env_domain=hipp_walker --env_task=stand --run_name=hipp_stand_eps005 \
    --episodes=1000 --eval_frequency=20 --num_envs=5 \
    --capacity=1000000 --warmup=5000 --lr=1e-4 --target_update_period=100 \
    --epsilon=0.05