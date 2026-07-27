#!/bin/bash

# python main.py --env_domain=walker_3D_ball --env_task=stand --run_name=w3d_stand_baseline \
#     --episodes=1000 --eval_frequency=20 --num_envs=5 \
#     --capacity=1000000 --warmup=5000 --lr=1e-4 --target_update_period=100 --curriculum \
#     --phase1_threshold=600 \
#     --phase2_threshold=700

python main.py --env_domain=walker_3D_ball --env_task=stand --run_name=w3d_stand_sgd16 \
    --episodes=1000 --eval_frequency=20 --num_envs=5 \
    --capacity=1000000 --warmup=5000 --lr=1e-4 --target_update_period=100 \
    --sgd_steps_per_learner_step=16 --curriculum \
    --phase1_threshold=600 \
    --phase2_threshold=700

python main.py --env_domain=walker_3D_ball --env_task=stand --run_name=w3d_stand_nstep3 \
    --episodes=1000 --eval_frequency=20 --num_envs=5 \
    --capacity=1000000 --warmup=5000 --lr=1e-4 --target_update_period=100 \
    --n_step=3 --curriculum \
    --phase1_threshold=600 \
    --phase2_threshold=700

python main.py --env_domain=walker_3D_ball --env_task=stand --run_name=w3d_stand_eps005 \
    --episodes=1000 --eval_frequency=20 --num_envs=5 \
    --capacity=1000000 --warmup=5000 --lr=1e-4 --target_update_period=100 \
    --epsilon=0.05 --curriculum \
    --phase1_threshold=600 \
    --phase2_threshold=700