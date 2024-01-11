# ORPO Implementation

This repository implements ORPO, a model-based offline RL algorithmic framework which can generate Optimistic model Rollouts for Pessimistic offline policy Optimization.
The implementation in this repositorory is used in the paper "Optimistic Model Rollouts for Pessimistic Offline Policy Optimization", which has been accepted by AAAI 2024.

## Implemented Baselines

- Model-free
    - [CQL](https://arxiv.org/abs/2006.04779)
    - [TD3+BC](https://arxiv.org/abs/2106.06860)
- Model-based
    - [MOPO](https://arxiv.org/abs/2005.13239)
    - [COMBO](https://arxiv.org/abs/2102.08363)
    - ORPO (Ours)
## Environment Setup

Recommend to run code within a [`conda`](https://docs.conda.io/projects/conda/en/latest/user-guide/install/) virtual environment. Create a virtual environment by:

```
conda create -n ORPO python=3.7
```

Activate the virtual environment by running:

```
conda activate ORPO
```
Install the following dependencies:

- MuJoCo 2.0
- Gym 0.22.0
- D4RL
- PyTorch 1.8+

Install dependencies by running the following command in the root directory of this repository (in the virtual environment):

```
cd CodeForORPO
pip install -e .
cd toy_exp/square_env
pip install -e .
```

## Training Examples

### Toy experiments
```shell
python toy_exp/run_mopo_toy.py --penalty-coef 100  --uncertainty_mode ensemble_std  --epoch 10
```

```shell
python toy_exp/run_orpo_td3bc_toy.py --penalty-coef 100 --bonus-coef 1 --uncertainty_mode ensemble_std  --epoch 10 --real-ratio-rollout 0.05 --real-ratio-final 0.05 --final-policy-rollout-ratio-final 0.45
```

### D4RL

```shell
python run_example/run_orpo.py --task halfcheetah-random-v2  --penalty-coef 6.64 --bonus-coef 0.015  --uncertainty_mode ensemble_std  --real-ratio-rollout 0.05 --real-ratio-final 0.05 --final-policy-rollout-ratio-final 0.45 --rollout-length-rollout-policy 5
```

```shell
python run_example/run_orpo.py --task walker2d-medium-replay-v2  --penalty-coef 2.48 --bonus-coef 0.015  --uncertainty_mode ensemble_std   --real-ratio-rollout 0.05 --real-ratio-final 0.05 --final-policy-rollout-ratio-final 0.45 --rollout-length 1  --rollout-length-rollout-policy 1  
```

### Tasks requiring policies to generalize

```shell
python run_example/generalization_datasets/collect_halfcheetah-jump.py
mv halfcheetah-jump.h5 run_example/generalization_datasets
python run_example/run_orpo.py --task halfcheetah-jump --dataset generalization_datasets/halfcheetah-jump.h5 --penalty-coef 1 --bonus-coef 0.1 --rollout-length 5  --rollout-length-rollout-policy 5  --real-ratio-rollout 0.05 --real-ratio-final 0.05 --final-policy-rollout-ratio-final 0 
```

```shell
python run_example/generalization_datasets/collect_halfcheetah-jump-hard.py
mv halfcheetah-jump-hard.h5 run_example/generalization_datasets
python run_example/run_orpo.py --task halfcheetah-jump-hard --dataset generalization_datasets/halfcheetah-jump-hard.h5 --penalty-coef 1 --bonus-coef 0.1 --rollout-length 5  --rollout-length-rollout-policy 5  --real-ratio-rollout 0.05 --real-ratio-final 0.05 --final-policy-rollout-ratio-final 0 
```

## Plotting Examples

```shell
python run_example/plot.py --task halfcheetah-random-v2 --algos mopo orpo
```

# References

- MOPO re-implementation: [https://github.com/yihaosun1124/OfflineRL-Kit](https://github.com/yihaosun1124/OfflineRL-Kit)
- Official d3rlpy implementation: https://github.com/takuseno/d3rlpy 
