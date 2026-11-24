"""
Streamlit dashboard for prediction, model health, and dataset insights.
"""

import os
from pathlib import Path

import requests
import streamlit as st
from PIL import Image


PROJECT_DIRECTORY = Path(__file__).resolve().parents[1]
API_URL = os.environ.get("PLANT_DISEASE_API_URL", "http://127.0.0.1:8000")
VISUALIZATIONS_DIRECTORY = PROJECT_DIRECTORY / "visualizations"

st.set_page_config(
    page_title="Plant Disease Classifier",
    layout="wide"
)

st.title("Plant Disease Classifier")
st.caption(
    "Upload a leaf photo for a model prediction. Results are screening "
    "assistance, not a substitute for expert agricultural diagnosis."
)

st.subheader("Model status")
try:
    health_response = requests.get(f"{API_URL}/health", timeout=5)
    health_response.raise_for_status()
    health = health_response.json()
    if health["model_loaded"]:
        st.success(
            f"API online — model loaded for {health['class_count']} classes."
        )
    else:
        st.warning("API is responding, but the model is still starting.")
except requests.RequestException as error:
    st.error(f"Cannot reach the prediction API at {API_URL}: {error}")
    health = None

st.subheader("Predict from an image")
uploaded_image = st.file_uploader(
    "Choose a leaf image",
    type=("jpg", "jpeg", "png"),
    help="Upload one JPEG or PNG image, up to 10 MB."
)

if uploaded_image is not None:
    image_bytes = uploaded_image.getvalue()
    image_column, results_column = st.columns(2)
    with image_column:
        st.image(
            Image.open(uploaded_image),
            caption=uploaded_image.name,
            use_container_width=True
        )
    with results_column:
        if st.button("Predict disease", type="primary", disabled=health is None):
            try:
                response = requests.post(
                    f"{API_URL}/predict",
                    files={
                        "image": (
                            uploaded_image.name,
                            image_bytes,
                            uploaded_image.type
                        )
                    },
                    params={"top_k": 3},
                    timeout=60
                )
                response.raise_for_status()
                results = response.json()["predictions"]
                st.success(
                    f"Top prediction: {results[0]['class_name']} "
                    f"({results[0]['confidence']:.1%})"
                )
                st.dataframe(
                    [
                        {
                            "Class": item["class_name"],
                            "Confidence": f"{item['confidence']:.1%}"
                        }
                        for item in results
                    ],
                    hide_index=True,
                    use_container_width=True
                )
            except requests.RequestException as error:
                st.error(f"Prediction request failed: {error}")

st.subheader("Dataset insights")
visualization_files = (
    ("Class distribution", "class_distribution.png"),
    ("Sample image from each class", "class_samples.png"),
    ("Image feature comparisons", "image_feature_comparisons.png")
)
for caption, filename in visualization_files:
    image_path = VISUALIZATIONS_DIRECTORY / filename
    if image_path.is_file():
        st.image(
            str(image_path),
            caption=caption,
            use_container_width=True
        )
    else:
        st.info(
            f"{filename} is not available yet. Generate visualizations "
            "with `python -m src.visualization`."
        )

interpretations_path = (
    VISUALIZATIONS_DIRECTORY / "feature_interpretations.txt"
)
if interpretations_path.is_file():
    with st.expander("Three measured image features"):
        st.text(interpretations_path.read_text(encoding="utf-8"))

st.subheader("Upload labelled data and retrain")
st.caption(
    "ZIP layout: one known PlantVillage class folder containing JPEG/PNG "
    "images. New classes are not accepted."
)
admin_token = st.text_input(
    "Retraining admin token",
    type="password",
    help="Set PLANT_DISEASE_ADMIN_TOKEN in the API terminal before starting it."
)
uploaded_archive = st.file_uploader(
    "Choose a labelled ZIP archive",
    type=("zip",),
    key="retraining_archive"
)

if st.button(
    "Upload data and start retraining",
    disabled=not admin_token or uploaded_archive is None
):
    try:
        upload_response = requests.post(
            f"{API_URL}/retraining/uploads",
            headers={"X-Admin-Token": admin_token},
            files={
                "file": (
                    uploaded_archive.name,
                    uploaded_archive.getvalue(),
                    "application/zip"
                )
            },
            timeout=120
        )
        upload_response.raise_for_status()
        upload_result = upload_response.json()
        trigger_response = requests.post(
            f"{API_URL}/retraining/uploads/"
            f"{upload_result['upload_id']}/trigger",
            headers={"X-Admin-Token": admin_token},
            timeout=10
        )
        trigger_response.raise_for_status()
        st.session_state["retraining_upload_id"] = upload_result["upload_id"]
        st.success(
            f"Saved {upload_result['image_count']} labelled images from "
            f"{len(upload_result['class_counts'])} classes. Retraining queued."
        )
    except requests.RequestException as error:
        st.error(f"Could not upload data or queue retraining: {error}")

upload_id = st.session_state.get("retraining_upload_id")
if upload_id:
    if st.button("Refresh retraining status"):
        st.rerun()
    try:
        status_response = requests.get(
            f"{API_URL}/retraining/uploads/{upload_id}",
            timeout=10
        )
        status_response.raise_for_status()
        retraining_status = status_response.json()
        st.info(f"Retraining status: {retraining_status['status']}")
        if retraining_status["result"]:
            st.json(retraining_status["result"])
        if retraining_status["error"]:
            st.error(retraining_status["error"])
    except requests.RequestException as error:
        st.error(f"Could not read retraining status: {error}")
