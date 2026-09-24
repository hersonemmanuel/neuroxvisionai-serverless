from pathlib import Path
import argparse
import json
import time

import nibabel as nib
import numpy as np
import torch

from monai.inferers import sliding_window_inference
from monai.networks.nets import UNet


ROOT = Path(__file__).resolve().parents[1]

CHECKPOINT = ROOT / "models/tpr_student/best_model.pt"

ROI_SIZE = (96, 96, 96)

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# ============================================================
# MODEL
# ============================================================

def create_model():

    return UNet(
        spatial_dims=3,
        in_channels=4,
        out_channels=3,
        channels=(16, 32, 64, 128, 256),
        strides=(2, 2, 2, 2),
        num_res_units=2,
        norm="INSTANCE"
    )


# ============================================================
# IMAGE UTILITIES
# ============================================================

def load_nifti(path):

    nii = nib.load(str(path))

    data = np.asarray(
        nii.dataobj,
        dtype=np.float32
    )

    return nii, data


def normalize_nonzero(volume):

    mask = volume != 0

    output = np.zeros_like(
        volume,
        dtype=np.float32
    )

    if not mask.any():
        return output

    values = volume[mask]

    mean = values.mean()
    std = values.std()

    if std < 1e-8:
        std = 1.0

    output[mask] = (
        volume[mask] - mean
    ) / std

    return output


def check_geometry(niftis, arrays):

    reference_shape = arrays[0].shape
    reference_affine = niftis[0].affine

    for i in range(1, 4):

        if arrays[i].shape != reference_shape:

            raise ValueError(
                "MRI volumes do not have matching shapes: "
                f"{reference_shape} vs {arrays[i].shape}"
            )

        if not np.allclose(
            niftis[i].affine,
            reference_affine,
            atol=1e-4
        ):

            raise ValueError(
                "MRI volumes do not share the same spatial affine."
            )


# ============================================================
# SEGMENTATION
# ============================================================

def enforce_tumor_hierarchy(prediction):

    wt = prediction[0].astype(bool)
    tc = prediction[1].astype(bool)
    et = prediction[2].astype(bool)

    # Enforce:
    # ET subset TC subset WT

    tc = np.logical_or(tc, et)
    wt = np.logical_or(wt, tc)

    return np.stack(
        [wt, tc, et],
        axis=0
    ).astype(np.uint8)


def create_visual_label_map(masks):

    wt, tc, et = masks

    label_map = np.zeros(
        wt.shape,
        dtype=np.uint8
    )

    # Visualization labels only:
    # 0 = background
    # 1 = WT-only
    # 2 = TC
    # 3 = ET

    label_map[wt == 1] = 1
    label_map[tc == 1] = 2
    label_map[et == 1] = 3

    return label_map


# ============================================================
# SAVE
# ============================================================

def save_nifti(
    data,
    reference,
    path
):

    image = nib.Nifti1Image(
        data,
        affine=reference.affine,
        header=reference.header.copy()
    )

    image.set_data_dtype(
        data.dtype
    )

    nib.save(
        image,
        str(path)
    )


# ============================================================
# INFERENCE
# ============================================================

def run_inference(
    t1_path,
    t1ce_path,
    t2_path,
    flair_path,
    output_dir
):

    output_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    paths = [
        t1_path,
        t1ce_path,
        t2_path,
        flair_path
    ]

    names = [
        "T1",
        "T1ce",
        "T2",
        "FLAIR"
    ]

    niftis = []
    arrays = []

    print("\nLoading MRI volumes...")

    for name, path in zip(
        names,
        paths
    ):

        if not path.exists():

            raise FileNotFoundError(
                f"{name} not found: {path}"
            )

        nii, data = load_nifti(
            path
        )

        niftis.append(nii)
        arrays.append(data)

        print(
            f"{name:6s}: "
            f"{data.shape}"
        )

    check_geometry(
        niftis,
        arrays
    )

    print("MRI geometry check: PASSED")

    image = np.stack(
        [
            normalize_nonzero(x)
            for x in arrays
        ],
        axis=0
    )

    image = torch.from_numpy(
        np.ascontiguousarray(image)
    ).unsqueeze(0).to(
        DEVICE
    )

    # --------------------------------------------------------
    # MODEL
    # --------------------------------------------------------

    model = create_model().to(
        DEVICE
    )

    checkpoint = torch.load(
        CHECKPOINT,
        map_location=DEVICE
    )

    model.load_state_dict(
        checkpoint["state_dict"]
    )

    model.eval()

    print(
        f"\nModel loaded: "
        f"{CHECKPOINT.name}"
    )

    print(
        f"Device: {DEVICE}"
    )

    start = time.time()

    with torch.no_grad():

        if DEVICE.type == "cuda":

            with torch.autocast(
                device_type="cuda",
                dtype=torch.float16
            ):

                logits = sliding_window_inference(
                    inputs=image,
                    roi_size=ROI_SIZE,
                    sw_batch_size=1,
                    predictor=model,
                    overlap=0.5
                )

        else:

            logits = sliding_window_inference(
                inputs=image,
                roi_size=ROI_SIZE,
                sw_batch_size=1,
                predictor=model,
                overlap=0.5
            )

    probabilities = torch.sigmoid(
        logits
    )

    prediction = (
        probabilities > 0.5
    ).cpu().numpy()[0].astype(
        np.uint8
    )

    prediction = enforce_tumor_hierarchy(
        prediction
    )

    elapsed = time.time() - start

    wt, tc, et = prediction

    visual_map = create_visual_label_map(
        prediction
    )

    # --------------------------------------------------------
    # VOLUME
    # --------------------------------------------------------

    voxel_sizes = niftis[0].header.get_zooms()[:3]

    voxel_volume_mm3 = float(
        np.prod(voxel_sizes)
    )

    volumes = {}

    for name, mask in zip(
        ["WT", "TC", "ET"],
        [wt, tc, et]
    ):

        voxels = int(
            mask.sum()
        )

        volume_mm3 = (
            voxels * voxel_volume_mm3
        )

        volumes[name] = {
            "voxels": voxels,
            "volume_mm3": volume_mm3,
            "volume_cm3": volume_mm3 / 1000.0
        }

    # --------------------------------------------------------
    # SAVE MASKS
    # --------------------------------------------------------

    save_nifti(
        wt,
        niftis[0],
        output_dir / "WT_mask.nii.gz"
    )

    save_nifti(
        tc,
        niftis[0],
        output_dir / "TC_mask.nii.gz"
    )

    save_nifti(
        et,
        niftis[0],
        output_dir / "ET_mask.nii.gz"
    )

    save_nifti(
        visual_map,
        niftis[0],
        output_dir / "tumor_regions.nii.gz"
    )

    np.savez_compressed(
        output_dir / "prediction_probabilities.npz",
        probabilities=probabilities
        .cpu()
        .numpy()[0]
        .astype(np.float32)
    )

    report = {
        "model": "TPR-BraTS MRI Deployment Model",
        "checkpoint": str(CHECKPOINT),
        "device": str(DEVICE),
        "inference_time_seconds": elapsed,
        "input_shape": list(arrays[0].shape),
        "voxel_spacing_mm": list(
            map(float, voxel_sizes)
        ),
        "regions": volumes,
        "visualization_labels": {
            "0": "Background",
            "1": "Whole Tumor only",
            "2": "Tumor Core",
            "3": "Enhancing Tumor"
        }
    }

    with open(
        output_dir / "inference_report.json",
        "w"
    ) as f:

        json.dump(
            report,
            f,
            indent=4
        )

    print(
        "\n"
        + "=" * 80
    )

    print(
        "TPR-BRATS INFERENCE COMPLETE"
    )

    print(
        "=" * 80
    )

    print(
        f"WT volume : "
        f"{volumes['WT']['volume_cm3']:.2f} cm³"
    )

    print(
        f"TC volume : "
        f"{volumes['TC']['volume_cm3']:.2f} cm³"
    )

    print(
        f"ET volume : "
        f"{volumes['ET']['volume_cm3']:.2f} cm³"
    )

    print(
        f"Inference time: "
        f"{elapsed:.2f} s"
    )

    print(
        f"Outputs: {output_dir}"
    )

    print(
        "=" * 80
    )


# ============================================================
# CLI
# ============================================================

if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--t1",
        required=True,
        type=Path
    )

    parser.add_argument(
        "--t1ce",
        required=True,
        type=Path
    )

    parser.add_argument(
        "--t2",
        required=True,
        type=Path
    )

    parser.add_argument(
        "--flair",
        required=True,
        type=Path
    )

    parser.add_argument(
        "--output",
        required=True,
        type=Path
    )

    args = parser.parse_args()

    run_inference(
        t1_path=args.t1,
        t1ce_path=args.t1ce,
        t2_path=args.t2,
        flair_path=args.flair,
        output_dir=args.output
    )
