#!/usr/bin/env python
from __future__ import annotations

import argparse
import importlib
import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path


REQUIRED_IMPORTS = [
    "packaging",
    "numpy",
    "yaml",
    "torch",
    "datasets",
    "transformers",
    "accelerate",
    "peft",
    "evaluate",
    "wandb",
    "safetensors",
    "sentencepiece",
    "tqdm",
    "torch_geometric",
    "torch_scatter",
    "rouge_score",
    "glot_lclm",
]

OPTIONAL_IMPORTS = [
    "bitsandbytes",
    "kvpress",
]


def status(ok: bool) -> str:
    return "OK" if ok else "FAIL"


def run(cmd: list[str], timeout: int = 20) -> tuple[int, str]:
    try:
        proc = subprocess.run(
            cmd,
            check=False,
            text=True,
            capture_output=True,
            timeout=timeout,
        )
        return proc.returncode, (proc.stdout + proc.stderr).strip()
    except Exception as exc:  # pragma: no cover
        return 1, repr(exc)


def import_check(name: str) -> tuple[bool, str]:
    try:
        module = importlib.import_module(name)
        version = getattr(module, "__version__", "unknown")
        location = getattr(module, "__file__", "built-in")
        return True, f"{version} | {location}"
    except Exception as exc:
        return False, f"{type(exc).__name__}: {exc}"


def check_imports(names: list[str], required: bool) -> bool:
    print("\n== Python Imports ==" if required else "\n== Optional Imports ==")
    all_ok = True
    for name in names:
        ok, detail = import_check(name)
        all_ok = all_ok and (ok or not required)
        print(f"[{status(ok)}] {name}: {detail}")
    return all_ok


def check_torch() -> bool:
    print("\n== Torch / GPU ==")
    try:
        import torch
    except Exception as exc:
        print(f"[FAIL] torch import: {exc}")
        return False

    print(f"[OK] torch version: {torch.__version__}")
    cuda_ok = torch.cuda.is_available()
    print(f"[{status(cuda_ok)}] torch.cuda.is_available(): {cuda_ok}")
    if cuda_ok:
        print(f"[OK] CUDA version: {torch.version.cuda}")
        print(f"[OK] GPU count: {torch.cuda.device_count()}")
        for idx in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(idx)
            gb = props.total_memory / (1024**3)
            print(f"[OK] GPU {idx}: {props.name} | {gb:.1f} GB")
    return cuda_ok


def check_nvidia_smi() -> bool:
    print("\n== nvidia-smi ==")
    if shutil.which("nvidia-smi") is None:
        print("[FAIL] nvidia-smi not found on PATH")
        return False
    code, output = run(["nvidia-smi"], timeout=20)
    ok = code == 0
    print(f"[{status(ok)}] nvidia-smi exit code: {code}")
    print("\n".join(output.splitlines()[:20]))
    return ok


def check_repo_paths() -> bool:
    print("\n== Repo Path ==")
    cwd = Path.cwd()
    src = cwd / "src"
    print(f"[OK] cwd: {cwd}")
    print(f"[{status(src.exists())}] src exists: {src}")
    py_path = os.environ.get("PYTHONPATH", "")
    has_src = str(src) in py_path.split(":")
    print(f"[{status(has_src)}] PYTHONPATH contains repo src: {has_src}")
    if not has_src:
        print("      Fix: export PYTHONPATH=\"$PWD/src:${PYTHONPATH:-}\"")

    ok, detail = import_check("glot_lclm")
    print(f"[{status(ok)}] glot_lclm import: {detail}")
    return src.exists() and ok


def check_auth() -> None:
    print("\n== Auth Hints ==")
    hf_token = bool(os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN"))
    print(f"[{'OK' if hf_token else 'WARN'}] HF token env present: {hf_token}")
    if not hf_token:
        print("      Public downloads still work, but login avoids rate limits: huggingface-cli login")

    code, output = run([sys.executable, "-m", "wandb", "status"], timeout=20)
    ok = code == 0 and "api_key" in output.lower()
    print(f"[{'OK' if ok else 'WARN'}] wandb status command exit code: {code}")
    for line in output.splitlines()[:12]:
        print(f"      {line}")


def check_dataset_smoke() -> bool:
    print("\n== Dataset Smoke ==")
    try:
        from datasets import load_dataset

        ds = load_dataset("json", data_files={"train": "data/smoke_qa.jsonl"})
        n = len(ds["train"])
        ok = n == 4
        print(f"[{status(ok)}] local smoke dataset rows: {n}")
        return ok
    except Exception as exc:
        print(f"[FAIL] dataset smoke: {type(exc).__name__}: {exc}")
        return False


def check_config_smoke() -> bool:
    print("\n== Config Smoke ==")
    try:
        from glot_lclm.utils.config import apply_overrides, load_config

        cfg = load_config("configs/glot_squad_qwen05_8k_r4_sft.yaml")
        cfg = apply_overrides(
            cfg,
            [
                "dataset.train_limit=2",
                "training.stages.0.steps=1",
                "training.stages.1.steps=1",
            ],
        )
        ok = cfg["training"]["stages"][0]["steps"] == 1
        print(f"[{status(ok)}] config load + list override")
        return ok
    except Exception as exc:
        print(f"[FAIL] config smoke: {type(exc).__name__}: {exc}")
        return False


def check_model_smoke(device: str) -> bool:
    print("\n== Tiny Model Smoke ==")
    try:
        import torch

        from glot_lclm.data.qa_examples import QAExample
        from glot_lclm.models.compressor_qa import CompressedQAModel, set_trainability
        from glot_lclm.utils.config import load_config

        cfg = load_config("configs/smoke_tiny_glot.yaml")
        cfg["experiment"]["wandb_mode"] = "disabled"
        model = CompressedQAModel(cfg)
        if device == "cuda" and not torch.cuda.is_available():
            print("[WARN] CUDA requested but unavailable; using CPU")
            device = "cpu"
        model.to(device)
        set_trainability(
            model,
            train_pooler=True,
            train_adapter=True,
            train_encoder_lora=False,
            train_decoder_lora=False,
        )
        example = QAExample(
            qid="preflight",
            question="What color is the sky?",
            context="The sky is blue.",
            answers=["blue"],
            support_indices=[],
        )
        out = model.forward_compressed([example])
        ok = bool(torch.isfinite(out.loss).item())
        print(f"[{status(ok)}] one compressed forward loss: {float(out.loss.detach().cpu()):.4f}")
        return ok
    except Exception as exc:
        print(f"[FAIL] tiny model smoke: {type(exc).__name__}: {exc}")
        return False


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--deep", action="store_true", help="Run tiny model forward pass")
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    args = parser.parse_args()

    print("== System ==")
    print(f"[OK] executable: {sys.executable}")
    print(f"[OK] python: {sys.version.split()[0]}")
    print(f"[OK] platform: {platform.platform()}")

    py_ok = sys.version_info >= (3, 10)
    print(f"[{status(py_ok)}] python >= 3.10: {py_ok}")
    if not py_ok:
        print("      This project targets Python >= 3.10. Python 3.9 may fail later.")

    checks = [
        py_ok,
        check_repo_paths(),
        check_imports(REQUIRED_IMPORTS, required=True),
        check_torch(),
        check_nvidia_smi(),
        check_dataset_smoke(),
        check_config_smoke(),
    ]
    check_imports(OPTIONAL_IMPORTS, required=False)
    check_auth()
    if args.deep:
        checks.append(check_model_smoke(args.device))

    print("\n== Summary ==")
    ok = all(checks)
    print(f"[{status(ok)}] preflight {'passed' if ok else 'failed'}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
