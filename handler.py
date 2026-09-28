from __future__ import annotations

import importlib.util
import json
import os
import shutil
import tempfile
import time
import uuid
from pathlib import Path

import boto3
import runpod
import torch
from PIL import Image, UnidentifiedImageError

ROOT = Path(__file__).resolve().parent
INFERENCE_SCRIPT = ROOT / "scripts" / "19_inference_pipeline.py"

R2_BUCKET = os.environ["R2_BUCKET"]
R2_ENDPOINT_URL = os.environ["R2_ENDPOINT_URL"]
R2_ACCESS_KEY_ID = os.environ["R2_ACCESS_KEY_ID"]
R2_SECRET_ACCESS_KEY = os.environ["R2_SECRET_ACCESS_KEY"]

API_VERSION = "1.4.0"
MODEL_DISPLAY_NAME = "MRI Deployment Model"

r2 = boto3.client(
    "s3",
    endpoint_url=R2_ENDPOINT_URL,
    aws_access_key_id=R2_ACCESS_KEY_ID,
    aws_secret_access_key=R2_SECRET_ACCESS_KEY,
    region_name="auto",
)

spec = importlib.util.spec_from_file_location(
    "neuroxvisionai_inference", INFERENCE_SCRIPT
)
if spec is None or spec.loader is None:
    raise RuntimeError(f"Unable to load inference pipeline: {INFERENCE_SCRIPT}")

inference = importlib.util.module_from_spec(spec)
spec.loader.exec_module(inference)

# Reuse the already-working application model code.
from api.classifier import classify_image
from api.job_grade import compute_grade_risk_for_job
from api.grade_report import build_report


def _valid_job_id(value: str) -> str:
    return str(uuid.UUID(str(value)))


def _health():
    cuda = torch.cuda.is_available()
    return {
        "status": "healthy",
        "cuda_available": cuda,
        "gpu": torch.cuda.get_device_name(0) if cuda else None,
        "worker": "NeuroXVisionAI",
        "storage": "Cloudflare R2",
        "bucket": R2_BUCKET,
        "api_version": API_VERSION,
        "capabilities": ["single_image", "full_study"],
    }


def _object_exists(key: str) -> bool:
    try:
        r2.head_object(Bucket=R2_BUCKET, Key=key)
        return True
    except Exception as exc:
        response = getattr(exc, "response", {}) or {}
        code = str(response.get("Error", {}).get("Code", ""))
        status = response.get("ResponseMetadata", {}).get("HTTPStatusCode")
        if code in {"404", "NoSuchKey", "NotFound"} or status == 404:
            return False
        raise


def _find_r2_input(job_id: str, modality: str) -> str:
    for suffix in (".nii.gz", ".nii"):
        key = f"jobs/{job_id}/inputs/{modality}{suffix}"
        if _object_exists(key):
            return key
    raise FileNotFoundError(f"Missing {modality} MRI input in R2.")


def _download(key: str, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    r2.download_file(R2_BUCKET, key, str(path))
    if not path.exists() or path.stat().st_size <= 0:
        raise RuntimeError(f"Downloaded object is empty: {key}")


def _upload_results(job_id: str, result_dir: Path) -> dict[str, str]:
    uploaded = {}
    for path in sorted(result_dir.iterdir()):
        if not path.is_file():
            continue
        key = f"jobs/{job_id}/results/{path.name}"
        r2.upload_file(str(path), R2_BUCKET, key)
        uploaded[path.name] = key
    return uploaded


def _run_single_image(job, job_input):
    job_id = _valid_job_id(job_input.get("job_id"))
    key = str(job_input.get("image_key") or "")
    expected_prefix = f"jobs/{job_id}/inputs/single_image."

    if not key.startswith(expected_prefix):
        raise ValueError("Invalid single-image object key.")

    suffix = Path(key).suffix.lower()
    if suffix not in {".jpg", ".jpeg", ".png", ".bmp"}:
        raise ValueError("Unsupported image format. Use JPG, JPEG, PNG or BMP.")

    tmpdir = Path(tempfile.mkdtemp(prefix="neuroxvisionai_single_"))
    local_path = tmpdir / f"image{suffix}"

    try:
        runpod.serverless.progress_update(job, "Downloading MRI image")
        _download(key, local_path)

        try:
            with Image.open(local_path) as image:
                image.load()
                runpod.serverless.progress_update(job, "Running MRI classification")
                result = classify_image(image)
        except UnidentifiedImageError as exc:
            raise ValueError("Uploaded file is not a readable image.") from exc

        result["status"] = "completed"
        result["mode"] = "single_image"
        result["job_id"] = job_id
        result["api_version"] = API_VERSION
        return result

    finally:
        # Single-image uploads are not needed after classification.
        try:
            r2.delete_object(Bucket=R2_BUCKET, Key=key)
        except Exception:
            pass
        shutil.rmtree(tmpdir, ignore_errors=True)


def _run_full_study(job, job_input):
    job_id = _valid_job_id(job_input.get("job_id"))

    tmp_root = Path(tempfile.mkdtemp(prefix="neuroxvisionai_study_"))
    job_dir = tmp_root / job_id
    input_dir = job_dir / "inputs"
    result_dir = job_dir / "results"
    input_dir.mkdir(parents=True, exist_ok=True)
    result_dir.mkdir(parents=True, exist_ok=True)

    try:
        runpod.serverless.progress_update(job, "Downloading MRI study from R2")

        local_inputs = {}
        for modality in ("t1", "t1ce", "t2", "flair"):
            key = _find_r2_input(job_id, modality)
            suffix = ".nii.gz" if key.endswith(".nii.gz") else ".nii"
            local_path = input_dir / f"{modality}{suffix}"
            _download(key, local_path)
            local_inputs[modality] = local_path

        runpod.serverless.progress_update(job, "Running 3D MRI segmentation")
        inference.run_inference(
            local_inputs["t1"],
            local_inputs["t1ce"],
            local_inputs["t2"],
            local_inputs["flair"],
            result_dir,
        )

        inference_json = result_dir / "inference_report.json"
        if not inference_json.exists():
            raise RuntimeError("Inference completed but inference_report.json is missing.")

        with inference_json.open("r", encoding="utf-8") as fh:
            report = json.load(fh)

        runpod.serverless.progress_update(job, "Running experimental grade-risk analysis")
        try:
            grade_risk = compute_grade_risk_for_job(job_dir)
        except Exception as exc:
            grade_risk = {
                "status": "unavailable",
                "prediction": None,
                "error": str(exc),
                "warning": "Experimental grade-risk estimation was unavailable for this study.",
            }
            with (result_dir / "grade_risk.json").open("w", encoding="utf-8") as fh:
                json.dump(grade_risk, fh, indent=2)

        summary = {
            "job_id": job_id,
            "study_id": job_input.get("study_id"),
            "model": MODEL_DISPLAY_NAME,
            "api_version": API_VERSION,
            "regions": report["regions"],
            "inference_time_seconds": report["inference_time_seconds"],
        }
        with (result_dir / "analysis_summary.json").open("w", encoding="utf-8") as fh:
            json.dump(summary, fh, indent=2)

        runpod.serverless.progress_update(job, "Generating analysis report")
        try:
            build_report(job_id, job_dir)
        except Exception as exc:
            # Segmentation remains usable even if PDF generation fails.
            with (result_dir / "report_generation_error.txt").open("w", encoding="utf-8") as fh:
                fh.write(str(exc))

        runpod.serverless.progress_update(job, "Uploading results to R2")
        uploaded = _upload_results(job_id, result_dir)

        required = {
            "WT_mask.nii.gz",
            "TC_mask.nii.gz",
            "ET_mask.nii.gz",
            "tumor_regions.nii.gz",
            "inference_report.json",
        }
        missing = sorted(required - set(uploaded))
        if missing:
            raise RuntimeError("Missing required result files: " + ", ".join(missing))

        return {
            "status": "completed",
            "mode": "full_study",
            "job_id": job_id,
            "model": MODEL_DISPLAY_NAME,
            "api_version": API_VERSION,
            "inference_time_seconds": report["inference_time_seconds"],
            "regions": report["regions"],
            "grade_risk": grade_risk,
            "result_files": uploaded,
            "retention": "temporary; delete after use",
        }

    finally:
        shutil.rmtree(tmp_root, ignore_errors=True)


def handler(job):
    job_input = job.get("input") or {}
    mode = str(job_input.get("mode") or "").strip().lower()

    if mode == "health":
        return _health()
    if mode == "single_image":
        return _run_single_image(job, job_input)
    if mode == "full_study":
        return _run_full_study(job, job_input)

    raise ValueError(f"Unsupported mode: {mode!r}")


runpod.serverless.start({"handler": handler})
