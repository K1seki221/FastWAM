import json
import os
import re
import shlex
import shutil
import signal
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import hydra
from hydra.core.hydra_config import HydraConfig
from libero.libero import benchmark
from omegaconf import DictConfig, OmegaConf


def _find_training_run_dir(checkpoint_path: Path) -> Path | None:
    """Infer <run_dir> from <run_dir>/checkpoints/.../<checkpoint>."""
    for parent in checkpoint_path.parents:
        if parent.name == "checkpoints":
            return parent.parent
    return None


def save_checkpoint_provenance(
    *,
    checkpoint: str,
    output_dir: Path,
    task_choice: str,
) -> Path:
    """Save the exact training checkpoint provenance before workers start."""
    requested_path = str(checkpoint)
    expanded_path = Path(os.path.expanduser(os.path.expandvars(requested_path)))
    absolute_path = expanded_path.absolute()
    resolved_path = absolute_path.resolve(strict=False)

    checkpoint_stat = None
    try:
        checkpoint_stat = resolved_path.stat()
    except OSError:
        pass

    step_match = re.search(r"step_(\d+)", resolved_path.name)
    run_dir = _find_training_run_dir(resolved_path)
    training_config_path = None
    if run_dir is not None:
        config_candidates = (run_dir / "config.yaml", run_dir / ".hydra" / "config.yaml")
        training_config_path = next((path for path in config_candidates if path.is_file()), None)

    training_config_snapshot = output_dir / "training_config.yaml"
    if training_config_path is not None:
        shutil.copyfile(training_config_path, training_config_snapshot)
    elif training_config_snapshot.exists():
        training_config_snapshot.unlink()

    recorded_at = datetime.now(timezone.utc)
    evaluation_id = recorded_at.strftime("%Y%m%d_%H%M%S")
    training_summary_directory = (
        run_dir / "evaluate_results" / evaluation_id if run_dir is not None else None
    )
    checkpoint_info = {
        "recorded_at_utc": recorded_at.isoformat(),
        "task": task_choice,
        "evaluation": {
            "id": evaluation_id,
            "training_summary_directory": (
                str(training_summary_directory)
                if training_summary_directory is not None
                else None
            ),
        },
        "checkpoint": {
            "requested_path": requested_path,
            "absolute_path": str(absolute_path),
            "resolved_path": str(resolved_path),
            "filename": resolved_path.name,
            "step": int(step_match.group(1)) if step_match else None,
            "exists": resolved_path.exists(),
            "is_file": resolved_path.is_file(),
            "size_bytes": checkpoint_stat.st_size if checkpoint_stat is not None else None,
            "modified_at_utc": (
                datetime.fromtimestamp(checkpoint_stat.st_mtime, timezone.utc).isoformat()
                if checkpoint_stat is not None
                else None
            ),
        },
        "training_run": {
            "directory": str(run_dir) if run_dir is not None else None,
            "run_id": run_dir.name if run_dir is not None else None,
            "task_name": run_dir.parent.name if run_dir is not None else None,
            "config_path": (
                str(training_config_path) if training_config_path is not None else None
            ),
            "config_snapshot": (
                training_config_snapshot.name if training_config_path is not None else None
            ),
        },
    }

    output_path = output_dir / "checkpoint_info.json"
    temporary_path = output_path.with_suffix(f"{output_path.suffix}.tmp-{os.getpid()}")
    with temporary_path.open("w", encoding="utf-8") as f:
        json.dump(checkpoint_info, f, indent=2, ensure_ascii=False)
        f.write("\n")
    temporary_path.replace(output_path)
    return output_path


def create_task_file(output_file: Path, task_suite_names: list[str]) -> Path:
    benchmark_dict = benchmark.get_benchmark_dict()
    output_file.parent.mkdir(parents=True, exist_ok=True)

    total_tasks = 0
    with output_file.open("w", encoding="utf-8") as f:
        for suite_name in task_suite_names:
            task_suite = benchmark_dict[suite_name]()
            n_tasks = int(task_suite.n_tasks)
            print(f"\n{suite_name}:")
            print(f"- Number of tasks: {n_tasks}")
            for task_id in range(n_tasks):
                f.write(f"{suite_name},{task_id}\n")
                total_tasks += 1

    print(f"\nTask list created: {output_file}")
    print(f"Total tasks: {total_tasks}")
    return output_file


def _is_blocked_override(raw_override: str) -> bool:
    key = raw_override.split("=", 1)[0].lstrip("+~")
    blocked_exact = {
        "task",
        "ckpt",
        "gpu_id",
        "EVALUATION.task_suite_name",
        "EVALUATION.task_id",
    }
    if key in blocked_exact:
        return True
    return key.startswith("MULTIRUN.") or key.startswith("hydra.")


def collect_worker_overrides() -> list[str]:
    hydra_overrides = list(HydraConfig.get().overrides.task)
    return [ov for ov in hydra_overrides if not _is_blocked_override(ov)]


def _resolve_worker_task_choice() -> str:
    task_choice = HydraConfig.get().runtime.choices.get("task")
    if task_choice is None or str(task_choice).strip() == "":
        raise ValueError(
            "Hydra task choice is empty. Please pass task=... (e.g., task=world_action_model_forward_224)."
        )
    return str(task_choice)


def _print_worker_log_excerpts(output_dir: Path, max_logs: int = 4, max_lines: int = 200) -> None:
    failed_tasks = output_dir / "failed_tasks.txt"
    log_paths: list[Path] = []
    if failed_tasks.exists():
        for line in failed_tasks.read_text(encoding="utf-8", errors="replace").splitlines():
            marker = ",log="
            if marker in line:
                log_paths.append(Path(line.rsplit(marker, 1)[1]))

    if not log_paths:
        task_log_dir = output_dir / "task_logs"
        if task_log_dir.exists():
            log_paths = sorted(
                task_log_dir.glob("*.log"),
                key=lambda path: path.stat().st_mtime,
                reverse=True,
            )[:max_logs]

    seen: set[Path] = set()
    for log_path in log_paths[:max_logs]:
        if log_path in seen:
            continue
        seen.add(log_path)
        print(f"----- worker log: {log_path} (last {max_lines} lines) -----", flush=True)
        if not log_path.exists():
            print("[worker log does not exist; the tmux worker may not have started]", flush=True)
        elif log_path.stat().st_size == 0:
            print("[worker log is empty; the worker ended before Python produced output]", flush=True)
        else:
            lines = log_path.read_text(encoding="utf-8", errors="replace").splitlines()
            print("\n".join(lines[-max_lines:]), flush=True)
        print("----- end worker log -----", flush=True)


def run_evaluation(
    *,
    task_file: Path,
    task_choice: str,
    ckpt: str,
    num_gpus: int,
    num_trials: int,
    max_tasks_per_gpu: int,
    output_dir: Path,
    extra_overrides: list[str],
) -> None:
    script_path = Path("experiments/libero/run_libero_parallel_test.sh")
    if not script_path.exists():
        raise FileNotFoundError(f"Evaluation script not found: {script_path}")

    root_dir = os.getcwd()
    output_dir.mkdir(parents=True, exist_ok=True)
    extra_args = shlex.join(extra_overrides) if extra_overrides else ""
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    libero_pythonpath = str(Path(root_dir) / "third_party" / "LIBERO")
    existing_pythonpath = os.environ.get("PYTHONPATH", "")
    worker_pythonpath = (
        f"{libero_pythonpath}{os.pathsep}{existing_pythonpath}"
        if existing_pythonpath
        else libero_pythonpath
    )

    env = os.environ.copy()
    env.update(
        {
            "CONFIG": task_choice,
            "CKPT": ckpt,
            "NUM_GPUS": str(num_gpus),
            "NUM_TRIALS": str(num_trials),
            "MAX_TASKS_PER_GPU": str(max_tasks_per_gpu),
            "ROOT_DIR": root_dir,
            "RUN_ID": run_id,
            "OUTPUT_DIR": str(output_dir),
            "EXTRA_ARGS": extra_args,
            "EXP_NAME": os.environ.get("EXP_NAME", ""),
            "PYTHON_EXECUTABLE": sys.executable,
            "PYTHONPATH": worker_pythonpath,
        }
    )

    print("\nStarting evaluation (Hydra manager)...")
    print(f"task: {task_choice}")
    print(f"Checkpoint: {ckpt}")
    print(f"Number of GPUs: {num_gpus}")
    print(f"Trials per task: {num_trials}")
    print(f"Max tasks per GPU: {max_tasks_per_gpu}")
    print(f"Output directory: {output_dir}")
    if extra_args:
        print(f"Forwarded overrides: {extra_args}")

    process = subprocess.Popen(
        ["bash", str(script_path), str(task_file)],
        env=env,
        text=True,
    )
    received_signal: int | None = None
    previous_sigterm_handler = signal.getsignal(signal.SIGTERM)

    def _handle_sigterm(signum, _frame):
        nonlocal received_signal
        received_signal = signum
        print(
            f"\nEvaluation manager received {signal.Signals(signum).name}; "
            "forwarding it to the scheduler for tmux worker cleanup.",
            flush=True,
        )
        if process.poll() is None:
            process.send_signal(signum)

    signal.signal(signal.SIGTERM, _handle_sigterm)
    try:
        return_code = process.wait()
        if received_signal is not None:
            _print_worker_log_excerpts(output_dir)
            raise SystemExit(128 + received_signal)
        if return_code != 0:
            raise subprocess.CalledProcessError(return_code, process.args)
    except KeyboardInterrupt:
        print("\nEvaluation interrupted; forwarding SIGINT to the scheduler.", flush=True)
        if process.poll() is None:
            process.send_signal(signal.SIGINT)
            try:
                process.wait(timeout=15)
            except subprocess.TimeoutExpired:
                process.terminate()
        raise SystemExit(130) from None
    except subprocess.CalledProcessError as e:
        print(f"Evaluation script failed with return code: {e.returncode}", flush=True)
        failed_tasks = output_dir / "failed_tasks.txt"
        if failed_tasks.exists() and failed_tasks.stat().st_size > 0:
            print(f"Failed subtask list: {failed_tasks}", flush=True)
            print(failed_tasks.read_text(encoding="utf-8", errors="replace"), flush=True)
        _print_worker_log_excerpts(output_dir)
        raise
    finally:
        signal.signal(signal.SIGTERM, previous_sigterm_handler)


@hydra.main(version_base="1.3", config_path="../../configs", config_name="sim_libero.yaml")
def main(cfg: DictConfig):
    if cfg.ckpt is None:
        raise ValueError("ckpt must not be None.")
    if cfg.EVALUATION.output_dir is None:
        raise ValueError("EVALUATION.output_dir must not be None.")

    task_choice = _resolve_worker_task_choice()
    manager = cfg.MULTIRUN

    output_dir = Path(os.path.expanduser(os.path.expandvars(str(cfg.EVALUATION.output_dir))))
    output_dir.mkdir(parents=True, exist_ok=True)

    task_file_cfg = manager.get("task_file")
    if task_file_cfg:
        task_file = Path(os.path.expanduser(os.path.expandvars(str(task_file_cfg))))
    else:
        task_file = output_dir / "tasks.txt"
    task_file = create_task_file(task_file, list(manager.task_suite_names))

    OmegaConf.save(config=cfg, f=str(output_dir / "manager_config.yaml"))
    checkpoint_info_path = save_checkpoint_provenance(
        checkpoint=str(cfg.ckpt),
        output_dir=output_dir,
        task_choice=task_choice,
    )
    print(f"Checkpoint provenance saved: {checkpoint_info_path}")

    if bool(manager.get("create_only", False)):
        print("create_only=True, only create the task list and exit.")
        return

    run_evaluation(
        task_file=task_file,
        task_choice=task_choice,
        ckpt=str(cfg.ckpt),
        num_gpus=int(manager.num_gpus),
        num_trials=int(cfg.EVALUATION.num_trials),
        max_tasks_per_gpu=int(manager.max_tasks_per_gpu),
        output_dir=output_dir,
        extra_overrides=collect_worker_overrides(),
    )


if __name__ == "__main__":
    main()
