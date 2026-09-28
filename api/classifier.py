from pathlib import Path
from io import BytesIO
import threading

import torch
import torch.nn as nn

from fastapi import (
    APIRouter,
    File,
    HTTPException,
    UploadFile,
)

from PIL import (
    Image,
    UnidentifiedImageError,
)

from torchvision import transforms
from torchvision.models import efficientnet_b0


# ============================================================
# CONFIG
# ============================================================

ROOT = Path(__file__).resolve().parents[1]

MODEL_PATH = (
    ROOT
    / "models/unified_classifier/best_unified_classifier.pt"
)


MAX_IMAGE_BYTES = (
    20
    * 1024
    * 1024
)


ALLOWED_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
}


DISPLAY_NAMES = {

    "glioma":
        "Glioma",

    "meningioma":
        "Meningioma",

    "neurocytoma":
        "Neurocytoma",

    "no_tumor":
        "No Tumor",

    "other_uncertain":
        "Other / Uncertain",

    "pituitary":
        "Pituitary Tumor",

    "schwannoma":
        "Schwannoma",
}


router = APIRouter(
    tags=[
        "Single MRI Classification"
    ]
)


DEVICE = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)


_model = None
_preprocess = None
_idx_to_class = None

_model_lock = threading.Lock()


# ============================================================
# MODEL LOADING
# ============================================================

def load_classifier():

    global _model
    global _preprocess
    global _idx_to_class


    if _model is not None:
        return


    with _model_lock:

        if _model is not None:
            return


        if not MODEL_PATH.exists():

            raise RuntimeError(
                f"Unified classifier checkpoint "
                f"not found: {MODEL_PATH}"
            )


        checkpoint = torch.load(
            MODEL_PATH,
            map_location=DEVICE,
            weights_only=False,
        )


        class_names = checkpoint[
            "class_names"
        ]


        _idx_to_class = {
            index: name
            for index, name
            in enumerate(
                class_names
            )
        }


        model = efficientnet_b0(
            weights=None
        )


        in_features = (
            model.classifier[1]
            .in_features
        )


        model.classifier[1] = nn.Linear(
            in_features,
            len(
                class_names
            ),
        )


        model.load_state_dict(
            checkpoint[
                "state_dict"
            ]
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
            [
                0.485,
                0.456,
                0.406,
            ],
        )


        std = checkpoint.get(
            "std",
            [
                0.229,
                0.224,
                0.225,
            ],
        )


        _preprocess = transforms.Compose([

            transforms.Resize(
                (
                    image_size,
                    image_size,
                )
            ),

            transforms.ToTensor(),

            transforms.Normalize(
                mean=mean,
                std=std,
            ),
        ])


        _model = model


# ============================================================
# INFERENCE
# ============================================================

def classify_image(
    image: Image.Image,
):

    load_classifier()


    tensor = _preprocess(
        image.convert(
            "RGB"
        )
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


        probabilities = torch.softmax(
            logits,
            dim=1,
        )[0]


    predicted_index = int(
        probabilities.argmax().item()
    )


    raw_prediction = (
        _idx_to_class[
            predicted_index
        ]
    )


    confidence = float(
        probabilities[
            predicted_index
        ].item()
    )


    probability_results = {}


    for index in range(
        len(
            _idx_to_class
        )
    ):

        raw_name = (
            _idx_to_class[
                index
            ]
        )


        display_name = DISPLAY_NAMES.get(
            raw_name,
            raw_name,
        )


        probability_results[
            display_name
        ] = float(
            probabilities[
                index
            ].item()
        )


    # ========================================================
    # CONFIDENCE HANDLING
    # ========================================================

    if confidence < 0.50:

        confidence_level = (
            "very low"
        )

        interpretation = (
            "Very-low-confidence prediction. "
            "The image cannot be classified reliably "
            "by the current model."
        )


    elif confidence < 0.65:

        confidence_level = (
            "low"
        )

        interpretation = (
            "Low-confidence prediction. "
            "Treat this result as uncertain."
        )


    elif confidence < 0.80:

        confidence_level = (
            "moderate"
        )

        interpretation = (
            "Moderate-confidence model prediction."
        )


    else:

        confidence_level = (
            "high"
        )

        interpretation = (
            "High-confidence model prediction."
        )


    # ========================================================
    # TOP 3 PREDICTIONS
    # ========================================================

    ranked = sorted(
        probability_results.items(),
        key=lambda item: item[1],
        reverse=True,
    )


    top_predictions = [
        {
            "class": name,
            "probability": probability,
            "percent": round(
                probability * 100,
                2,
            ),
        }
        for name, probability
        in ranked[:3]
    ]


    # ========================================================
    # RETURN
    # ========================================================

    return {

        "prediction":
            DISPLAY_NAMES.get(
                raw_prediction,
                raw_prediction,
            ),

        "raw_prediction":
            raw_prediction,

        "confidence":
            confidence,

        "confidence_percent":
            round(
                confidence * 100,
                2,
            ),

        "confidence_level":
            confidence_level,

        "probabilities":
            probability_results,

        "top_predictions":
            top_predictions,

        "interpretation":
            interpretation,

        "model":
            "Unified EfficientNet-B0",

        "task":
            (
                "Single MRI image brain tumor "
                "screening and tumor-family classification"
            ),

        "supported_classes": [
            "Glioma",
            "Meningioma",
            "Pituitary Tumor",
            "Neurocytoma",
            "Schwannoma",
            "No Tumor",
            "Other / Uncertain",
        ],

        "research_performance": {

            "test_accuracy":
                0.9576,

            "test_balanced_accuracy":
                0.9732,

            "test_macro_f1":
                0.9584,
        },

        "limitations": [

            (
                "The model classifies a single "
                "2D MRI image."
            ),

            (
                "The result depends on the MRI "
                "appearance represented in the "
                "training datasets."
            ),

            (
                "Other / Uncertain is a learned "
                "miscellaneous-lesion class and "
                "must not be interpreted as a "
                "complete open-set detector of "
                "every possible brain abnormality."
            ),

            (
                "A single-image prediction does "
                "not provide 3D tumor segmentation."
            ),

            (
                "Definitive tumor diagnosis and "
                "grading may require complete MRI, "
                "clinical information, pathology "
                "and molecular testing."
            ),

            (
                "The result is an AI imaging "
                "prediction and not an independent "
                "clinical diagnosis."
            ),
        ],
    }


# ============================================================
# API ENDPOINT
# ============================================================

@router.post(
    "/classify"
)
async def classify_single_mri(

    image: UploadFile = File(...),

):

    filename = (
        image.filename
        or
        "uploaded_image"
    )


    suffix = Path(
        filename
    ).suffix.lower()


    if suffix not in ALLOWED_EXTENSIONS:

        raise HTTPException(
            status_code=400,
            detail=(
                "Unsupported image format. "
                "Use JPG, JPEG, PNG or BMP."
            ),
        )


    contents = await image.read()


    if not contents:

        raise HTTPException(
            status_code=400,
            detail=(
                "Uploaded image is empty."
            ),
        )


    if len(contents) > MAX_IMAGE_BYTES:

        raise HTTPException(
            status_code=413,
            detail=(
                "Image exceeds the "
                "20 MB upload limit."
            ),
        )


    try:

        pil_image = Image.open(
            BytesIO(
                contents
            )
        )

        pil_image.load()


    except (
        UnidentifiedImageError,
        OSError,
        ValueError,
    ):

        raise HTTPException(
            status_code=400,
            detail=(
                "The uploaded file is "
                "not a valid image."
            ),
        )


    try:

        result = classify_image(
            pil_image
        )


    except Exception as exc:

        raise HTTPException(
            status_code=500,
            detail=(
                f"Classification failed: "
                f"{exc}"
            ),
        )


    result[
        "filename"
    ] = filename


    return result


# ============================================================
# MODEL INFO
# ============================================================

@router.get(
    "/classifier-info"
)
def classifier_info():

    return {

        "status":
            "available",

        "model":
            "Unified EfficientNet-B0",

        "device":
            str(
                DEVICE
            ),

        "gpu":
            (
                torch.cuda.get_device_name(
                    0
                )
                if torch.cuda.is_available()
                else None
            ),

        "classes": [
            "Glioma",
            "Meningioma",
            "Pituitary Tumor",
            "Neurocytoma",
            "Schwannoma",
            "No Tumor",
            "Other / Uncertain",
        ],

        "test_accuracy":
            0.9576,

        "test_balanced_accuracy":
            0.9732,

        "test_macro_f1":
            0.9584,

        "important_note": (
            "These are research-development "
            "test results and are not independent "
            "clinical validation."
        ),
    }
