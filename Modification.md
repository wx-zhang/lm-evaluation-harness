# Discriminative Evaluation

We follow the [HuggingFace leaderboard](https://github.com/EleutherAI/lm-evaluation-harness) for discriminative evaluation. 

Check [here](https://github.com/EleutherAI/lm-evaluation-harness/blob/main/README.md) for installation and more information. 

## Modifications
Add `hydra-core` to pass in parameters to cope with `wandb` sweep

## Sync
The original github repo [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness) is under refactorization. Keep syncing the `main` branch with the repo and use the following command to update the `private` branch
```
git checkout main
git pull
git checkout private
git cherry-pick [HASH]..main
```

## Basic Usage
Our code is in the `private` branch

Check `test.sh` for supported tasks and models