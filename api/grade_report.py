from pathlib import Path
import json
import re
import uuid

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import cm
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
)


ROOT = Path(__file__).resolve().parents[1]

API_OUTPUT = ROOT / "outputs/web_api"

router = APIRouter(
    tags=["Analysis Report"]
)


def load_json(path):

    if not path.exists():
        return {}

    try:
        with open(path, "r") as f:
            return json.load(f)

    except Exception:
        return {}


def guess_study_id(
    job_dir,
    summary,
    inference,
):

    for source in [
        summary,
        inference,
    ]:

        for key in [
            "study_id",
            "patient_id",
            "subject_id",
        ]:

            value = source.get(
                key
            )

            if value:
                return str(value)


    input_dir = (
        job_dir
        / "inputs"
    )

    if input_dir.exists():

        files = (
            list(
                input_dir.glob("*.nii")
            )
            +
            list(
                input_dir.glob("*.nii.gz")
            )
        )

        if files:

            name = files[0].name

            name = re.sub(
                r"\.nii(\.gz)?$",
                "",
                name,
                flags=re.IGNORECASE,
            )

            name = re.sub(
                r"[_-](t1ce|t1c|t1n|t1|t2w|t2|t2f|flair)$",
                "",
                name,
                flags=re.IGNORECASE,
            )

            return name

    return "Unknown"


def format_volume(
    region,
):

    value = region.get(
        "volume_cm3"
    )

    if value is None:
        return "N/A"

    try:
        return f"{float(value):.2f}"

    except Exception:
        return str(value)


def build_report(
    job_id,
    job_dir,
):

    result_dir = (
        job_dir
        / "results"
    )

    summary = load_json(
        result_dir
        / "analysis_summary.json"
    )

    inference = load_json(
        result_dir
        / "inference_report.json"
    )

    grade = load_json(
        result_dir
        / "grade_risk.json"
    )


    output_path = (
        result_dir
        / "analysis_report_with_grade.pdf"
    )


    study_id = guess_study_id(
        job_dir,
        summary,
        inference,
    )


    regions = inference.get(
        "regions",
        summary.get(
            "regions",
            {}
        ),
    )


    styles = getSampleStyleSheet()


    title_style = ParagraphStyle(
        "TitleCustom",
        parent=styles["Title"],
        alignment=TA_CENTER,
        fontSize=20,
        leading=24,
        spaceAfter=16,
    )


    heading_style = ParagraphStyle(
        "HeadingCustom",
        parent=styles["Heading2"],
        fontSize=13,
        leading=16,
        spaceBefore=10,
        spaceAfter=8,
    )


    body_style = ParagraphStyle(
        "BodyCustom",
        parent=styles["BodyText"],
        fontSize=9.5,
        leading=14,
    )


    warning_style = ParagraphStyle(
        "WarningCustom",
        parent=body_style,
        backColor=colors.HexColor(
            "#FFF7DB"
        ),
        borderColor=colors.HexColor(
            "#D69E00"
        ),
        borderWidth=1,
        borderPadding=8,
        spaceBefore=8,
        spaceAfter=8,
    )


    doc = SimpleDocTemplate(
        str(output_path),
        pagesize=A4,
        rightMargin=1.7 * cm,
        leftMargin=1.7 * cm,
        topMargin=1.7 * cm,
        bottomMargin=1.7 * cm,
    )


    story = []


    story.append(
        Paragraph(
            "Brain Tumor AI Analysis Report",
            title_style,
        )
    )


    story.append(
        Paragraph(
            "Research and decision-support output",
            ParagraphStyle(
                "subtitle",
                parent=body_style,
                alignment=TA_CENTER,
                textColor=colors.HexColor(
                    "#64748B"
                ),
            ),
        )
    )

    story.append(
        Spacer(
            1,
            12,
        )
    )


    # ========================================================
    # STUDY INFORMATION
    # ========================================================

    story.append(
        Paragraph(
            "Study Information",
            heading_style,
        )
    )


    info_table = Table(
        [
            [
                "Study ID",
                study_id,
            ],
            [
                "Analysis ID",
                job_id,
            ],
            [
                "Segmentation model",
                "MRI Deployment Model",
            ],
        ],
        colWidths=[
            5 * cm,
            11 * cm,
        ],
    )


    info_table.setStyle(
        TableStyle([
            (
                "BACKGROUND",
                (0, 0),
                (0, -1),
                colors.HexColor(
                    "#EEF4F8"
                ),
            ),
            (
                "GRID",
                (0, 0),
                (-1, -1),
                0.4,
                colors.HexColor(
                    "#CBD5E1"
                ),
            ),
            (
                "FONTNAME",
                (0, 0),
                (0, -1),
                "Helvetica-Bold",
            ),
            (
                "FONTSIZE",
                (0, 0),
                (-1, -1),
                9,
            ),
            (
                "VALIGN",
                (0, 0),
                (-1, -1),
                "MIDDLE",
            ),
            (
                "TOPPADDING",
                (0, 0),
                (-1, -1),
                7,
            ),
            (
                "BOTTOMPADDING",
                (0, 0),
                (-1, -1),
                7,
            ),
        ])
    )


    story.append(
        info_table
    )


    # ========================================================
    # SEGMENTATION
    # ========================================================

    story.append(
        Paragraph(
            "Predicted Tumor Regions",
            heading_style,
        )
    )


    segmentation_data = [
        [
            "Region",
            "Volume (cm³)",
            "Voxels",
        ]
    ]


    for code, name in [
        (
            "WT",
            "Whole Tumor",
        ),
        (
            "TC",
            "Tumor Core",
        ),
        (
            "ET",
            "Enhancing Tumor",
        ),
    ]:

        region = regions.get(
            code,
            {}
        )

        segmentation_data.append([
            f"{name} ({code})",
            format_volume(
                region
            ),
            str(
                region.get(
                    "voxels",
                    "N/A",
                )
            ),
        ])


    segmentation_table = Table(
        segmentation_data,
        colWidths=[
            8 * cm,
            4 * cm,
            4 * cm,
        ],
    )


    segmentation_table.setStyle(
        TableStyle([
            (
                "BACKGROUND",
                (0, 0),
                (-1, 0),
                colors.HexColor(
                    "#0B6FA4"
                ),
            ),
            (
                "TEXTCOLOR",
                (0, 0),
                (-1, 0),
                colors.white,
            ),
            (
                "FONTNAME",
                (0, 0),
                (-1, 0),
                "Helvetica-Bold",
            ),
            (
                "GRID",
                (0, 0),
                (-1, -1),
                0.4,
                colors.HexColor(
                    "#CBD5E1"
                ),
            ),
            (
                "FONTSIZE",
                (0, 0),
                (-1, -1),
                9,
            ),
            (
                "ALIGN",
                (1, 1),
                (-1, -1),
                "CENTER",
            ),
            (
                "TOPPADDING",
                (0, 0),
                (-1, -1),
                7,
            ),
            (
                "BOTTOMPADDING",
                (0, 0),
                (-1, -1),
                7,
            ),
        ])
    )


    story.append(
        segmentation_table
    )


    # ========================================================
    # GRADE RISK
    # ========================================================

    story.append(
        Paragraph(
            "Experimental Glioma Grade-Risk Estimate",
            heading_style,
        )
    )


    if (
        grade
        and
        grade.get("status")
        == "completed"
    ):

        prediction = grade.get(
            "prediction",
            "Unavailable",
        )

        confidence = grade.get(
            "confidence_percent",
            0,
        )

        level = grade.get(
            "confidence_level",
            "unknown",
        )

        probability_high = (
            float(
                grade.get(
                    "probability_higher_grade",
                    0,
                )
            )
            * 100
        )

        probability_low = (
            float(
                grade.get(
                    "probability_lower_grade",
                    0,
                )
            )
            * 100
        )


        grade_table = Table(
            [
                [
                    "Grade-risk pattern",
                    prediction,
                ],
                [
                    "Model confidence",
                    f"{confidence:.2f}% ({level})",
                ],
                [
                    "Higher-grade probability",
                    f"{probability_high:.2f}%",
                ],
                [
                    "Lower-grade probability",
                    f"{probability_low:.2f}%",
                ],
                [
                    "Slice selection",
                    grade.get(
                        "slice_selection",
                        "N/A",
                    ),
                ],
            ],
            colWidths=[
                7 * cm,
                9 * cm,
            ],
        )


        grade_table.setStyle(
            TableStyle([
                (
                    "BACKGROUND",
                    (0, 0),
                    (0, -1),
                    colors.HexColor(
                        "#F1F5F9"
                    ),
                ),
                (
                    "GRID",
                    (0, 0),
                    (-1, -1),
                    0.4,
                    colors.HexColor(
                        "#CBD5E1"
                    ),
                ),
                (
                    "FONTNAME",
                    (0, 0),
                    (0, -1),
                    "Helvetica-Bold",
                ),
                (
                    "FONTSIZE",
                    (0, 0),
                    (-1, -1),
                    9,
                ),
                (
                    "TOPPADDING",
                    (0, 0),
                    (-1, -1),
                    7,
                ),
                (
                    "BOTTOMPADDING",
                    (0, 0),
                    (-1, -1),
                    7,
                ),
            ])
        )


        story.append(
            grade_table
        )


        distance = abs(
            probability_high
            - 50.0
        )


        if distance < 10:

            story.append(
                Paragraph(
                    "<b>Uncertain result:</b> "
                    "The grade-risk estimate is close to the "
                    "model decision boundary and should be "
                    "interpreted with particular caution.",
                    warning_style,
                )
            )


        story.append(
            Paragraph(
                (
                    grade.get(
                        "warning"
                    )
                    or
                    "This is an experimental MRI-pattern estimate "
                    "and is not pathology-confirmed WHO grading."
                ),
                warning_style,
            )
        )


    else:

        story.append(
            Paragraph(
                "Grade-risk estimation was not available for this analysis.",
                body_style,
            )
        )


    # ========================================================
    # GRADE MODEL PERFORMANCE
    # ========================================================

    if grade.get(
        "model_performance"
    ):

        perf = grade[
            "model_performance"
        ]


        story.append(
            Paragraph(
                "Grade-Risk Model Research Performance",
                heading_style,
            )
        )


        perf_table = Table(
            [
                [
                    "Metric",
                    "Held-out test result",
                ],
                [
                    "Accuracy",
                    f"{float(perf.get('test_accuracy', 0)) * 100:.2f}%",
                ],
                [
                    "Balanced accuracy",
                    f"{float(perf.get('test_balanced_accuracy', 0)) * 100:.2f}%",
                ],
                [
                    "Macro F1",
                    f"{float(perf.get('test_macro_f1', 0)) * 100:.2f}%",
                ],
                [
                    "ROC-AUC",
                    f"{float(perf.get('test_roc_auc', 0)):.4f}",
                ],
            ],
            colWidths=[
                8 * cm,
                8 * cm,
            ],
        )


        perf_table.setStyle(
            TableStyle([
                (
                    "BACKGROUND",
                    (0, 0),
                    (-1, 0),
                    colors.HexColor(
                        "#475569"
                    ),
                ),
                (
                    "TEXTCOLOR",
                    (0, 0),
                    (-1, 0),
                    colors.white,
                ),
                (
                    "GRID",
                    (0, 0),
                    (-1, -1),
                    0.4,
                    colors.HexColor(
                        "#CBD5E1"
                    ),
                ),
                (
                    "FONTSIZE",
                    (0, 0),
                    (-1, -1),
                    9,
                ),
                (
                    "TOPPADDING",
                    (0, 0),
                    (-1, -1),
                    6,
                ),
                (
                    "BOTTOMPADDING",
                    (0, 0),
                    (-1, -1),
                    6,
                ),
            ])
        )


        story.append(
            perf_table
        )


    # ========================================================
    # LIMITATIONS
    # ========================================================

    story.append(
        Paragraph(
            "Interpretation and Limitations",
            heading_style,
        )
    )


    limitations = [
        (
            "Tumor-region masks and volumes are automated "
            "model predictions."
        ),
        (
            "The grade-risk component estimates a lower-grade "
            "or higher-grade MRI pattern only."
        ),
        (
            "The grade-risk output is not a pathology-confirmed "
            "WHO tumor grade."
        ),
        (
            "Histological and molecular testing remain necessary "
            "for definitive tumor classification and grading."
        ),
        (
            "The system is intended for research and "
            "decision-support evaluation and is not an "
            "independent clinical diagnosis."
        ),
    ]


    for item in limitations:

        story.append(
            Paragraph(
                f"• {item}",
                body_style,
            )
        )

        story.append(
            Spacer(
                1,
                3,
            )
        )


    story.append(
        Spacer(
            1,
            8,
        )
    )


    story.append(
        Paragraph(
            (
                "<b>Temporary data handling:</b> "
                "Uploaded MRI volumes and generated outputs are "
                "temporarily retained by the research platform "
                "and may be deleted using the analysis-deletion control."
            ),
            body_style,
        )
    )


    doc.build(
        story
    )

    return output_path


@router.get(
    "/grade-report/{job_id}"
)
def grade_report(
    job_id: str,
):

    try:

        uuid.UUID(
            job_id
        )

    except ValueError:

        raise HTTPException(
            status_code=400,
            detail="Invalid analysis ID.",
        )


    job_dir = (
        API_OUTPUT
        / job_id
    )


    if not job_dir.exists():

        raise HTTPException(
            status_code=404,
            detail="Analysis not found or expired.",
        )


    try:

        pdf_path = build_report(
            job_id,
            job_dir,
        )

    except Exception as exc:

        raise HTTPException(
            status_code=500,
            detail=(
                f"Unable to generate report: {exc}"
            ),
        )


    return FileResponse(
        pdf_path,
        media_type="application/pdf",
        filename=(
            f"brain_tumor_analysis_{job_id}.pdf"
        ),
        headers={
            "Cache-Control": "no-store",
        },
    )
