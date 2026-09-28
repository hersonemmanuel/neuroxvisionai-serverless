from pathlib import Path
import json

import nibabel as nib
import numpy as np

from api.glioma_grade import predict_grade


# ============================================================
# MODALITY DETECTION
# ============================================================

def strip_nifti_suffix(name: str) -> str:

    name = name.lower()

    if name.endswith(".nii.gz"):
        return name[:-7]

    if name.endswith(".nii"):
        return name[:-4]

    return name


def detect_modality(path: Path):

    name = strip_nifti_suffix(
        path.name
    )

    # T1 contrast enhanced
    if (
        "t1ce" in name
        or name.endswith("_t1c")
        or name.endswith("-t1c")
        or name == "t1c"
    ):
        return "t1ce"

    # FLAIR
    if (
        "flair" in name
        or name.endswith("_t2f")
        or name.endswith("-t2f")
        or name == "t2f"
    ):
        return "flair"

    # T2
    if (
        name.endswith("_t2")
        or name.endswith("-t2")
        or name.endswith("_t2w")
        or name.endswith("-t2w")
        or name in {"t2", "t2w"}
    ):
        return "t2"

    return None


def find_required_inputs(
    input_dir: Path,
):

    found = {}

    files = list(
        input_dir.glob("*.nii")
    ) + list(
        input_dir.glob("*.nii.gz")
    )

    for path in files:

        modality = detect_modality(
            path
        )

        if (
            modality is not None
            and modality not in found
        ):
            found[
                modality
            ] = path

    missing = [
        modality
        for modality in [
            "t1ce",
            "t2",
            "flair",
        ]
        if modality not in found
    ]

    if missing:

        raise RuntimeError(
            "Missing MRI modality for grade-risk inference: "
            + ", ".join(missing)
        )

    return found


# ============================================================
# NIFTI LOADER
# ============================================================

def load_volume(
    path: Path,
):

    image = nib.load(
        str(path)
    )

    data = image.get_fdata(
        dtype=np.float32
    )

    return data


# ============================================================
# AUTOMATIC JOB GRADING
# ============================================================

def compute_grade_risk_for_job(
    job_dir: Path,
):

    job_dir = Path(
        job_dir
    )

    input_dir = (
        job_dir
        / "inputs"
    )

    result_dir = (
        job_dir
        / "results"
    )

    if not input_dir.exists():

        raise RuntimeError(
            f"Input directory not found: {input_dir}"
        )

    if not result_dir.exists():

        raise RuntimeError(
            f"Result directory not found: {result_dir}"
        )


    # --------------------------------------------------------
    # Locate MRI sequences
    # --------------------------------------------------------

    modalities = find_required_inputs(
        input_dir
    )


    # --------------------------------------------------------
    # Use model-generated segmentation, NOT ground truth
    # --------------------------------------------------------

    mask_path = (
        result_dir
        / "tumor_regions.nii.gz"
    )

    if not mask_path.exists():

        # Fallback to WT mask if required.

        mask_path = (
            result_dir
            / "WT_mask.nii.gz"
        )


    if not mask_path.exists():

        raise RuntimeError(
            "Predicted tumor segmentation was not found."
        )


    # --------------------------------------------------------
    # Load images
    # --------------------------------------------------------

    t1ce = load_volume(
        modalities["t1ce"]
    )

    t2 = load_volume(
        modalities["t2"]
    )

    flair = load_volume(
        modalities["flair"]
    )

    tumor_mask = load_volume(
        mask_path
    )


    # --------------------------------------------------------
    # Geometry check
    # --------------------------------------------------------

    if not (
        t1ce.shape
        ==
        t2.shape
        ==
        flair.shape
        ==
        tumor_mask.shape
    ):

        raise RuntimeError(
            "MRI and predicted tumor-mask shapes do not match."
        )


    # --------------------------------------------------------
    # If model predicted no tumor region
    # --------------------------------------------------------

    if not np.any(
        tumor_mask > 0
    ):

        result = {

            "status": "not_run",

            "prediction": None,

            "reason": (
                "No predicted tumor voxels were available "
                "for tumor-centered grade-risk estimation."
            ),

            "warning": (
                "Grade-risk estimation is only applicable "
                "when a tumor region is identified."
            ),
        }


    else:

        result = predict_grade(
            t1ce=t1ce,
            t2=t2,
            flair=flair,
            tumor_mask=tumor_mask,
        )

        result[
            "status"
        ] = "completed"

        result[
            "mask_source"
        ] = (
            "MRI Deployment Model predicted tumor segmentation"
        )

        result[
            "automatic"
        ] = True


    # --------------------------------------------------------
    # Save independently
    # --------------------------------------------------------

    output_path = (
        result_dir
        / "grade_risk.json"
    )

    with open(
        output_path,
        "w",
    ) as f:

        json.dump(
            result,
            f,
            indent=4,
        )


    return result
