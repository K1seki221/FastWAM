Before begining, follow [FastWAM](https://github.com/yuantianyuan01/FastWAM#environment-setup) to setup environment, and prepare training and evaluating

# Strat Training
## On Remote Kernel
```
bash scripts/train_zero1.sh 8 task=libero_uncond_2cam224_1e-4
```

## Submit Pytorchjob (example)
**You need to modify certain variables in the script according to your actual code path, such as DEFAULT_REPO_ROOT.**

```
bash path_to_codebase/scripts/train_fuyao_fastwam.sh 8 task=libero_uncond_2cam224_1e-4
```
You can specify conda env by passing `CONDA_ENV=path_to_env`.

If you use wandb (default), you can specify wandb api key by passing `WANDB_API_KEY="xxx"`.

# Start Evaluating
Before evaluating, you have to install egllib:
```bash
apt install -y libegl1 libopengl0 libglvnd0 libgl1
```

Sometimes you have to set
```bash
export PYTHONPATH=$PYTHONPATH:path_to_LIBERO
```

## On Remote Kernel
```python
python experiments/libero/run_libero_manager.py task={task_name} ckpt={ckpt_path}
```

## Submit Pytorchjob
**You need to modify certain variables in the script according to your actual code path, such as DEFAULT_REPO_ROOT.**

```bash
bash path_to_codebase/scripts/eval_fuyao_libero.sh task={task_name} ckpt={ckpt_path} MULTIRUN.num_gpus=2
```
