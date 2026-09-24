from __future__ import annotations

import importlib.util
import json
import os
import uuid
from pathlib import Path

import runpod
import torch


ROOT = Path(__file__).resolve().parent
INFERENCE_SCRIPT = ROOT / "scripts" / "19_inference_pipeline.py"
VOLUME_ROOT = Path(os.getenv("RUNPOD_VOLUME_PATH", "/runpod-volume"))
JOBS_ROOT = VOLUME_ROOT / "neuroxvisionai_data" / "jobs"


spec = importlib.util.spec_from_file_location("neuroxvisionai_inference", INFERENCE_SCRIPT)
if spec is None or spec.loader is None:
    raise RuntimeError("Unable to load NeuroXVisionAI inference module.")

inference = importlib.util.module_from_spec(spec)
spec.loader.exec_module(inference)


def _valid_job_id(value: str) -> str:
    try:
        return str(uuid.UUID(str(value)))
    except (ValueError, TypeError) as exc:
        raise ValueError("Invalid job_id.") from exc


def _find_input(input_dir: Path, stem: str) -> Path:
    candidates = [input_dir / f"{stem}.nii.gz", input_dir / f"{stem}.nii"]
    for path in candidates:
        if path.exists() and path.is_file():
            return path
    raise FileNotFoundError(f"Missing {stem} MRI volume in {input_dir}")


def _health() -> dict:
    cuda = torch.cuda.is_available()
    return {
        "status": "healthy",
        "cuda_available": cuda,
        "gpu": torch.cuda.get_device_name(0) if cuda else None,
        "worker": "NeuroXVisionAI 3D MRI segmentation",
        "network_volume": str(VOLUME_ROOT),
    }


def _run_full_study(job: dict, job_input: dict) -> dict:
    job_id = _valid_job_id(job_input.get("job_id"))
    job_dir = JOBS_ROOT / job_id
    input_dir = job_dir / "inputs"
    result_dir = job_dir / "results"

    if not input_dir.exists():
        raise FileNotFoundError(
            f"Shared job input directory not found: {input_dir}. "
            "Attach the same RunPod Network Volume to the CPU host and Serverless endpoint."
        )

    t1 = _find_input(input_dir, "t1")
    t1ce = _find_input(input_dir, "t1ce")
    t2 = _find_input(input_dir, "t2")
    flair = _find_input(input_dir, "flair")

    result_dir.mkdir(parents=True, exist_ok=True)

    runpod.serverless.progress_update(job, "MRI inputs found; starting 3D segmentation")

    inference.run_inference(
        t1,
        t1ce,
        t2,
        flair,
        result_dir,
    )

    report_path = result_dir / "inference_report.json"
    if not report_path.exists():
        raise RuntimeError("Segmentation completed but inference_report.json was not created.")

    with report_path.open("r", encoding="utf-8") as fh:
        report = json.load(fh)

    runpod.serverless.progress_update(job, "3D segmentation completed")

    return {
        "status": "completed",
        "mode": "full_study",
        "job_id": job_id,
        "inference_time_seconds": report.get("inference_time_seconds"),
        "regions": report.get("regions"),
        "result_files": [
            "WT_mask.nii.gz",
            "TC_mask.nii.gz",
            "ET_mask.nii.gz",
            "tumor_regions.nii.gz",
            "inference_report.json",
        ],
    }


def handler(job):
    job_input = job.get("input") or {}
    mode = str(job_input.get("mode", "")).strip().lower()

    if mode == "health":
        return _health()

    if mode == "full_study":
        return _run_full_study(job, job_input)

    raise ValueError("Unsupported mode. Use 'health' or 'full_study'.")


runpod.serverless.start({"handler": handler})
