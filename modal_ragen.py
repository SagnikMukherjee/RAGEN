import os
import shlex
import shutil
import subprocess
import time
from pathlib import Path

import modal


APP_NAME = "ragen-experiments"
REMOTE_ROOT = Path("/home/sagnikm3/RAGEN")
BFEX_ROOT = Path("/shared/storage-01/users/sagnikm3/bfex")
ARTIFACT_VOLUME_NAME = "ragen-artifacts"
SECRET_NAME = "ragen-secrets"

GPU = os.environ.get("RAGEN_MODAL_GPU", "H100:8")
TIMEOUT_SECONDS = 24 * 60 * 60

app = modal.App(APP_NAME)
artifact_volume = modal.Volume.from_name(ARTIFACT_VOLUME_NAME, create_if_missing=True)

image = (
    modal.Image.from_registry(
        "nvidia/cuda:12.8.1-devel-ubuntu22.04",
        add_python="3.12",
    )
    .entrypoint([])
    .apt_install(
        "build-essential",
        "cmake",
        "curl",
        "git",
        "libglib2.0-0",
        "libgl1",
        "ninja-build",
        "wget",
    )
    .add_local_file(
        "verl/scripts/install_vllm_sglang_mcore.sh",
        "/tmp/install_vllm_sglang_mcore.sh",
        copy=True,
    )
    .run_commands(
        "USE_MEGATRON=0 USE_SGLANG=1 bash /tmp/install_vllm_sglang_mcore.sh",
        "pip install --no-cache-dir IPython matplotlib gym gym_sokoban gymnasium 'gymnasium[toy-text]' debugpy together anthropic faiss-cpu==1.11.0 numpy==1.26.4",
        "pip install --no-cache-dir 'setuptools<70.0.0'",
    )
    .add_local_dir(
        ".",
        str(REMOTE_ROOT),
        copy=True,
        ignore=[
            ".git",
            ".git/**",
            "__pycache__",
            "**/__pycache__/**",
            "*.pyc",
            "checkpoints/**",
            "results/**",
            "outputs/**",
            "logs/**",
            "model_saving/**",
            "wandb/**",
            "verl.broken/**",
        ],
    )
    .run_commands(
        f"cd {REMOTE_ROOT} && pip install -e . --no-deps",
        f"cd {REMOTE_ROOT / 'verl'} && pip install --no-deps -e .",
    )
    .env(
        {
            "PYTHONPATH": f"{REMOTE_ROOT / 'verl'}:{REMOTE_ROOT}",
            "HF_HOME": str(BFEX_ROOT / "hf_cache"),
            "HUGGINGFACE_HUB_CACHE": str(BFEX_ROOT / "hf_cache" / "hub"),
            "TRANSFORMERS_CACHE": str(BFEX_ROOT / "hf_cache" / "transformers"),
            "HF_DATASETS_CACHE": str(BFEX_ROOT / "hf_cache" / "datasets"),
            "HF_HUB_ENABLE_HF_TRANSFER": "1",
            "WANDB_DIR": str(BFEX_ROOT / "wandb_logs"),
            "WANDB_CACHE_DIR": str(BFEX_ROOT / "wandb_cache"),
            "WANDB_ARTIFACT_DIR": str(BFEX_ROOT / "wandb_artifacts"),
            "TMPDIR": "/tmp/ragen",
            "RAY_TMPDIR": "/tmp/ragen/ray",
            "XDG_CACHE_HOME": str(BFEX_ROOT / ".cache"),
            "RAGEN_MODAL_GPU": GPU,
        }
    )
    .workdir(str(REMOTE_ROOT))
)


def _copy_if_present(path: Path, destination: Path) -> None:
    if not path.exists():
        return
    target = destination / path.name
    if target.exists():
        if target.is_dir():
            shutil.rmtree(target)
        else:
            target.unlink()
    shutil.copytree(path, target) if path.is_dir() else shutil.copy2(path, target)


@app.function(
    image=image,
    gpu=GPU,
    timeout=TIMEOUT_SECONDS,
    startup_timeout=60 * 60,
    volumes={str(BFEX_ROOT): artifact_volume},
    secrets=[modal.Secret.from_name(SECRET_NAME)],
)
def run_ragen_script(script: str, script_args: str = "", run_name: str = "") -> dict:
    run_name = run_name or f"{Path(script).stem}-{int(time.time())}"
    run_dir = BFEX_ROOT / "modal_runs" / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    for path in [
        Path(os.environ["TMPDIR"]),
        Path(os.environ["RAY_TMPDIR"]),
        Path(os.environ["HUGGINGFACE_HUB_CACHE"]),
        Path(os.environ["TRANSFORMERS_CACHE"]),
        Path(os.environ["HF_DATASETS_CACHE"]),
        Path(os.environ["WANDB_DIR"]),
        Path(os.environ["WANDB_CACHE_DIR"]),
        Path(os.environ["WANDB_ARTIFACT_DIR"]),
        Path(os.environ["XDG_CACHE_HOME"]),
    ]:
        path.mkdir(parents=True, exist_ok=True)

    script_path = REMOTE_ROOT / script
    if not script_path.exists():
        raise FileNotFoundError(f"Could not find {script_path}")

    print(f"Modal GPU request: {GPU}")
    subprocess.run(["nvidia-smi"], check=False)

    argv = ["bash", str(script_path), *shlex.split(script_args)]
    print(f"Running: {shlex.join(argv)}")
    started = time.time()
    try:
        subprocess.run(argv, cwd=REMOTE_ROOT, check=True)
    finally:
        for relative in ["logs", "outputs", "model_saving"]:
            _copy_if_present(REMOTE_ROOT / relative, run_dir)
        artifact_volume.commit()

    elapsed_seconds = int(time.time() - started)
    return {
        "script": script,
        "script_args": script_args,
        "run_name": run_name,
        "elapsed_seconds": elapsed_seconds,
        "artifacts": str(run_dir),
        "volume": ARTIFACT_VOLUME_NAME,
    }


@app.local_entrypoint()
def main(
    script: str = "train_lookahead_baseline.sh",
    script_args: str = "",
    run_name: str = "",
    wait: bool = False,
) -> None:
    if wait:
        result = run_ragen_script.remote(script, script_args, run_name)
        print(result)
        return

    function_call = run_ragen_script.spawn(script, script_args, run_name)
    print("Spawned run_ragen_script asynchronously.")
    print(f"Function call ID: {function_call.object_id}")
    print(f"Dashboard: {function_call.get_dashboard_url()}")
    print(f"Stream logs: modal app logs {APP_NAME} --function-call {function_call.object_id} -f")
    print(f"Fetch logs: modal app logs {APP_NAME} --function-call {function_call.object_id} --tail 200")
