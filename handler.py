from __future__ import annotations

import importlib.util
import json
import os
import shutil
import tempfile
import uuid
from pathlib import Path

import boto3
import runpod
import torch


ROOT = Path(__file__).resolve().parent
INFERENCE_SCRIPT = ROOT / "scripts" / "19_inference_pipeline.py"

R2_BUCKET = os.environ["R2_BUCKET"]
R2_ENDPOINT_URL = os.environ["R2_ENDPOINT_URL"]
R2_ACCESS_KEY_ID = os.environ["R2_ACCESS_KEY_ID"]
R2_SECRET_ACCESS_KEY = os.environ["R2_SECRET_ACCESS_KEY"]


r2 = boto3.client(
    "s3",
    endpoint_url=R2_ENDPOINT_URL,
    aws_access_key_id=R2_ACCESS_KEY_ID,
    aws_secret_access_key=R2_SECRET_ACCESS_KEY,
    region_name="auto",
)


spec = importlib.util.spec_from_file_location(
    "neuroxvisionai_inference",
    INFERENCE_SCRIPT,
)

if spec is None or spec.loader is None:
    raise RuntimeError("Unable to load NeuroXVisionAI inference module.")

inference = importlib.util.module_from_spec(spec)
spec.loader.exec_module(inference)


def _valid_job_id(value: str) -> str:
    try:
        return str(uuid.UUID(str(value)))
    except (ValueError, TypeError) as exc:
        raise ValueError("Invalid job_id.") from exc


def _health() -> dict:
    cuda = torch.cuda.is_available()

    return {
        "status": "healthy",
        "cuda_available": cuda,
        "gpu": torch.cuda.get_device_name(0) if cuda else None,
        "worker": "NeuroXVisionAI 3D MRI segmentation",
        "storage": "Cloudflare R2",
        "bucket": R2_BUCKET,
    }


def _object_exists(key: str) -> bool:
    try:
        r2.head_object(
            Bucket=R2_BUCKET,
            Key=key,
        )
        return True
    except Exception as exc:
        response = getattr(exc, "response", {})
        error = response.get("Error", {})
        code = str(error.get("Code", ""))

        if code in {"404", "NoSuchKey", "NotFound"}:
            return False

        raise


def _find_r2_input(job_id: str, modality: str) -> tuple[str, str]:
    candidates = [
        (
            f"jobs/{job_id}/inputs/{modality}.nii.gz",
            f"{modality}.nii.gz",
        ),
        (
            f"jobs/{job_id}/inputs/{modality}.nii",
            f"{modality}.nii",
        ),
    ]

    for key, filename in candidates:
        if _object_exists(key):
            return key, filename

    raise FileNotFoundError(
        f"Missing {modality} MRI volume for job {job_id}."
    )


def _download_input(
    job_id: str,
    modality: str,
    input_dir: Path,
) -> Path:
    key, filename = _find_r2_input(job_id, modality)

    destination = input_dir / filename

    r2.download_file(
        R2_BUCKET,
        key,
        str(destination),
    )

    if not destination.exists() or destination.stat().st_size == 0:
        raise RuntimeError(
            f"Downloaded {modality} MRI file is empty."
        )

    return destination


def _upload_results(
    job_id: str,
    result_dir: Path,
) -> dict:
    uploaded = {}

    for path in sorted(result_dir.iterdir()):
        if not path.is_file():
            continue

        key = f"jobs/{job_id}/results/{path.name}"

        r2.upload_file(
            str(path),
            R2_BUCKET,
            key,
        )

        uploaded[path.name] = key

    return uploaded


def _run_full_study(job: dict, job_input: dict) -> dict:
    job_id = _valid_job_id(job_input.get("job_id"))

    work_root = Path(
        tempfile.mkdtemp(
            prefix=f"neuroxvisionai-{job_id}-"
        )
    )

    input_dir = work_root / "inputs"
    result_dir = work_root / "results"

    input_dir.mkdir(parents=True, exist_ok=True)
    result_dir.mkdir(parents=True, exist_ok=True)

    try:
        runpod.serverless.progress_update(
            job,
            "Downloading MRI inputs from secure storage",
        )

        t1 = _download_input(
            job_id,
            "t1",
            input_dir,
        )

        t1ce = _download_input(
            job_id,
            "t1ce",
            input_dir,
        )

        t2 = _download_input(
            job_id,
            "t2",
            input_dir,
        )

        flair = _download_input(
            job_id,
            "flair",
            input_dir,
        )

        runpod.serverless.progress_update(
            job,
            "MRI inputs downloaded; starting 3D segmentation",
        )

        inference.run_inference(
            t1,
            t1ce,
            t2,
            flair,
            result_dir,
        )

        report_path = result_dir / "inference_report.json"

        if not report_path.exists():
            raise RuntimeError(
                "Segmentation completed but "
                "inference_report.json was not created."
            )

        with report_path.open(
            "r",
            encoding="utf-8",
        ) as fh:
            report = json.load(fh)

        runpod.serverless.progress_update(
            job,
            "Uploading analysis results",
        )

        uploaded = _upload_results(
            job_id,
            result_dir,
        )

        required_outputs = [
            "WT_mask.nii.gz",
            "TC_mask.nii.gz",
            "ET_mask.nii.gz",
            "tumor_regions.nii.gz",
            "inference_report.json",
        ]

        missing_outputs = [
            name
            for name in required_outputs
            if name not in uploaded
        ]

        if missing_outputs:
            raise RuntimeError(
                "Missing expected inference outputs: "
                + ", ".join(missing_outputs)
            )

        runpod.serverless.progress_update(
            job,
            "3D segmentation completed",
        )

        return {
            "status": "completed",
            "mode": "full_study",
            "job_id": job_id,
            "inference_time_seconds": report.get(
                "inference_time_seconds"
            ),
            "regions": report.get("regions"),
            "result_files": uploaded,
        }

    finally:
        shutil.rmtree(
            work_root,
            ignore_errors=True,
        )


def handler(job):
    job_input = job.get("input") or {}

    mode = str(
        job_input.get("mode", "")
    ).strip().lower()

    if mode == "health":
        return _health()

    if mode == "full_study":
        return _run_full_study(
            job,
            job_input,
        )

    raise ValueError(
        "Unsupported mode. Use 'health' or 'full_study'."
    )


runpod.serverless.start(
    {"handler": handler}
)
