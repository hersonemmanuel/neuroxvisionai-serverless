from pathlib import Path
import tempfile
import threading

import nibabel as nib
import numpy as np
from PIL import Image

import torch
import torch.nn as nn
from torchvision import transforms
from torchvision.models import efficientnet_b0

from fastapi import APIRouter, File, UploadFile, HTTPException


ROOT = Path(__file__).resolve().parents[1]

MODEL_PATH = (
    ROOT
    / "models/glioma_grade/best_glioma_grade_model.pt"
)

router = APIRouter(
    tags=["Glioma Grade Risk"]
)


DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

_model = None
_transform = None
_lock = threading.Lock()


# ============================================================
# MODEL
# ============================================================

def load_model():

    global _model
    global _transform

    if _model is not None:
        return

    with _lock:

        if _model is not None:
            return

        if not MODEL_PATH.exists():
            raise RuntimeError(
                f"Grade model not found: {MODEL_PATH}"
            )

        checkpoint = torch.load(
            MODEL_PATH,
            map_location=DEVICE,
            weights_only=False,
        )

        model = efficientnet_b0(
            weights=None
        )

        in_features = (
            model.classifier[1].in_features
        )

        model.classifier[1] = nn.Linear(
            in_features,
            2,
        )

        model.load_state_dict(
            checkpoint["state_dict"]
        )

        model = model.to(
            DEVICE
        )

        model.eval()

        image_size = checkpoint.get(
            "image_size",
            224,
        )

        mean = checkpoint.get(
            "mean",
            [0.485, 0.456, 0.406],
        )

        std = checkpoint.get(
            "std",
            [0.229, 0.224, 0.225],
        )

        _transform = transforms.Compose([
            transforms.Resize(
                (image_size, image_size)
            ),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=mean,
                std=std,
            ),
        ])

        _model = model


# ============================================================
# NIFTI
# ============================================================

def load_nifti_bytes(
    contents,
    filename,
):

    filename = (
        filename
        or ""
    ).lower()

    if filename.endswith(".nii.gz"):
        suffix = ".nii.gz"

    elif filename.endswith(".nii"):
        suffix = ".nii"

    else:
        raise ValueError(
            "Expected .nii or .nii.gz file."
        )

    with tempfile.NamedTemporaryFile(
        suffix=suffix,
        delete=True,
    ) as tmp:

        tmp.write(
            contents
        )

        tmp.flush()

        image = nib.load(
            tmp.name
        )

        data = image.get_fdata(
            dtype=np.float32
        )

    return data


# ============================================================
# NORMALIZATION
# ============================================================

def normalize_slice(
    image,
):

    image = image.astype(
        np.float32
    )

    valid = (
        np.isfinite(image)
        &
        (image != 0)
    )

    if not valid.any():

        return np.zeros_like(
            image,
            dtype=np.uint8,
        )

    values = image[
        valid
    ]

    low = np.percentile(
        values,
        1
    )

    high = np.percentile(
        values,
        99
    )

    if high <= low:
        high = low + 1.0

    image = np.clip(
        image,
        low,
        high,
    )

    image = (
        image - low
    ) / (
        high - low
    )

    image = (
        image * 255.0
    ).clip(
        0,
        255
    )

    return image.astype(
        np.uint8
    )


# ============================================================
# SLICE SELECTION
# ============================================================

def choose_slices(
    flair,
    tumor_mask=None,
    count=5,
):

    # --------------------------------------------------------
    # Preferred mode:
    # use tumor segmentation to identify most informative slices
    # --------------------------------------------------------

    if tumor_mask is not None:

        if tumor_mask.shape != flair.shape:

            raise ValueError(
                "Tumor mask and MRI volumes must have matching shapes."
            )

        tumor_area = (
            tumor_mask > 0
        ).sum(
            axis=(0, 1)
        )

        candidate_slices = np.argsort(
            tumor_area
        )[::-1]

        candidate_slices = [
            int(z)
            for z in candidate_slices
            if tumor_area[z] > 0
        ]

        if candidate_slices:

            selected = []

            for z in candidate_slices:

                if all(
                    abs(z - previous) >= 2
                    for previous in selected
                ):

                    selected.append(
                        z
                    )

                if len(selected) == count:
                    break

            while len(selected) < count:
                selected.append(
                    selected[-1]
                )

            return (
                sorted(selected),
                "tumor_centered",
            )


    # --------------------------------------------------------
    # Fallback:
    # use central brain slices
    # --------------------------------------------------------

    nonzero_area = (
        flair != 0
    ).sum(
        axis=(0, 1)
    )

    valid = np.where(
        nonzero_area > 0
    )[0]

    center = (
        flair.shape[2]
        // 2
    )

    if len(valid) == 0:

        return (
            [center] * count,
            "central_fallback",
        )

    distances = np.abs(
        valid - center
    )

    order = np.argsort(
        distances
    )

    selected = []

    for index in order:

        z = int(
            valid[index]
        )

        if all(
            abs(z - previous) >= 2
            for previous in selected
        ):

            selected.append(
                z
            )

        if len(selected) == count:
            break

    while len(selected) < count:
        selected.append(
            selected[-1]
        )

    return (
        sorted(selected),
        "central_fallback",
    )


# ============================================================
# INFERENCE
# ============================================================

def predict_grade(
    t1ce,
    t2,
    flair,
    tumor_mask=None,
):

    load_model()


    if not (
        t1ce.shape
        ==
        t2.shape
        ==
        flair.shape
    ):

        raise ValueError(
            "T1ce, T2 and FLAIR volumes must have matching shapes."
        )


    selected_slices, selection_mode = choose_slices(
        flair,
        tumor_mask=tumor_mask,
        count=5,
    )


    probabilities = []


    for z in selected_slices:

        # Same channel construction used during training:
        #
        # R = T1ce
        # G = T2
        # B = FLAIR

        red = normalize_slice(
            t1ce[:, :, z]
        )

        green = normalize_slice(
            t2[:, :, z]
        )

        blue = normalize_slice(
            flair[:, :, z]
        )


        rgb = np.stack(
            [
                red,
                green,
                blue,
            ],
            axis=-1,
        )


        image = Image.fromarray(
            rgb
        )


        tensor = _transform(
            image
        ).unsqueeze(
            0
        ).to(
            DEVICE
        )


        with torch.no_grad():

            if DEVICE.type == "cuda":

                with torch.amp.autocast(
                    device_type="cuda"
                ):

                    logits = _model(
                        tensor
                    )

            else:

                logits = _model(
                    tensor
                )


            probability_high = torch.softmax(
                logits,
                dim=1,
            )[0, 1].item()


        probabilities.append(
            float(
                probability_high
            )
        )


    probability_high = float(
        np.mean(
            probabilities
        )
    )

    probability_low = (
        1.0
        - probability_high
    )


    if probability_high >= 0.5:

        prediction = (
            "Higher-grade pattern"
        )

        confidence = (
            probability_high
        )

    else:

        prediction = (
            "Lower-grade pattern"
        )

        confidence = (
            probability_low
        )


    # Conservative confidence wording because this model
    # is experimental and has moderate balanced accuracy.

    if confidence >= 0.80:

        confidence_level = "high"

    elif confidence >= 0.65:

        confidence_level = "moderate"

    else:

        confidence_level = "low"


    return {

        "prediction": prediction,

        "probability_higher_grade": (
            probability_high
        ),

        "probability_lower_grade": (
            probability_low
        ),

        "confidence": confidence,

        "confidence_percent": round(
            confidence * 100,
            2,
        ),

        "confidence_level": (
            confidence_level
        ),

        "slices_used": (
            selected_slices
        ),

        "slice_selection": (
            selection_mode
        ),

        "task": (
            "Experimental MRI-based glioma grade-risk estimation"
        ),

        "model_performance": {

            "test_accuracy": 0.8571,

            "test_balanced_accuracy": 0.7273,

            "test_macro_f1": 0.7565,

            "test_roc_auc": 0.7216,
        },

        "warning": (
            "Experimental MRI-pattern estimate only. "
            "This output is not a pathology-confirmed WHO tumor grade."
        ),
    }


# ============================================================
# API
# ============================================================

@router.post(
    "/glioma-grade-risk"
)
async def glioma_grade_risk(

    t1ce: UploadFile = File(...),

    t2: UploadFile = File(...),

    flair: UploadFile = File(...),

    tumor_mask: UploadFile | None = File(
        None
    ),
):


    uploads = {
        "t1ce": t1ce,
        "t2": t2,
        "flair": flair,
    }


    contents = {}


    for name, upload in uploads.items():

        filename = (
            upload.filename
            or ""
        ).lower()


        if not (
            filename.endswith(".nii")
            or
            filename.endswith(".nii.gz")
        ):

            raise HTTPException(
                status_code=400,
                detail=(
                    f"{name} must be .nii or .nii.gz"
                ),
            )


        data = await upload.read()


        if not data:

            raise HTTPException(
                status_code=400,
                detail=(
                    f"{name} file is empty."
                ),
            )


        contents[
            name
        ] = data


    # --------------------------------------------------------
    # Optional tumor segmentation
    # --------------------------------------------------------

    tumor_mask_data = None


    if tumor_mask is not None:

        mask_filename = (
            tumor_mask.filename
            or ""
        ).lower()


        if not (
            mask_filename.endswith(".nii")
            or
            mask_filename.endswith(".nii.gz")
        ):

            raise HTTPException(
                status_code=400,
                detail=(
                    "Tumor mask must be .nii or .nii.gz"
                ),
            )


        mask_contents = await tumor_mask.read()


        if not mask_contents:

            raise HTTPException(
                status_code=400,
                detail=(
                    "Tumor mask is empty."
                ),
            )


        try:

            tumor_mask_data = load_nifti_bytes(
                mask_contents,
                tumor_mask.filename,
            )

        except Exception as exc:

            raise HTTPException(
                status_code=400,
                detail=(
                    f"Invalid tumor mask: {exc}"
                ),
            )


    try:

        t1ce_data = load_nifti_bytes(
            contents["t1ce"],
            t1ce.filename,
        )

        t2_data = load_nifti_bytes(
            contents["t2"],
            t2.filename,
        )

        flair_data = load_nifti_bytes(
            contents["flair"],
            flair.filename,
        )


        result = predict_grade(
            t1ce=t1ce_data,
            t2=t2_data,
            flair=flair_data,
            tumor_mask=tumor_mask_data,
        )


    except Exception as exc:

        raise HTTPException(
            status_code=500,
            detail=(
                f"Grade-risk inference failed: {exc}"
            ),
        )


    return result


# ============================================================
# MODEL INFO
# ============================================================

@router.get(
    "/glioma-grade-info"
)
def glioma_grade_info():

    return {

        "status": "available",

        "model": "EfficientNet-B0",

        "task": (
            "Experimental glioma grade-risk estimation"
        ),

        "classes": [
            "Lower-grade pattern",
            "Higher-grade pattern",
        ],

        "test_accuracy": 0.8571,

        "test_balanced_accuracy": 0.7273,

        "test_macro_f1": 0.7565,

        "test_roc_auc": 0.7216,

        "preferred_slice_selection": (
            "tumor-centered using segmentation mask"
        ),

        "warning": (
            "Not equivalent to pathology-confirmed WHO grading."
        ),
    }
