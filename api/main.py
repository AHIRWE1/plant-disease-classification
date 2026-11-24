"""
FastAPI service for plant disease image predictions and model health.
"""

from contextlib import asynccontextmanager
import hmac
import os
from pathlib import Path
import tempfile
from typing import Callable

from fastapi import (
    BackgroundTasks,
    FastAPI,
    File,
    Header,
    HTTPException,
    Query,
    Request,
    UploadFile
)

from src.prediction import MAX_IMAGE_BYTES, load_predictor
from src.retraining import (
    MAX_ARCHIVE_BYTES,
    get_upload_record,
    retrain_uploaded_dataset,
    save_uploaded_dataset,
    trigger_retraining,
    update_upload_record
)


PROJECT_DIRECTORY = Path(__file__).resolve().parents[1]


def _require_admin_token(token):
    configured_token = os.environ.get("PLANT_DISEASE_ADMIN_TOKEN")
    if not configured_token:
        raise HTTPException(
            status_code=503,
            detail="Retraining is disabled until an admin token is configured."
        )
    if token is None or not hmac.compare_digest(token, configured_token):
        raise HTTPException(status_code=401, detail="Invalid admin token.")


def create_app(predictor_loader: Callable = load_predictor):
    """Create the API app; loader injection keeps endpoint tests lightweight."""

    @asynccontextmanager
    async def lifespan(app):
        app.state.predictor = predictor_loader(PROJECT_DIRECTORY)
        yield

    app = FastAPI(
        title="Plant Disease Classification API",
        description="Predict a PlantVillage class from a leaf image.",
        version="1.0.0",
        lifespan=lifespan
    )

    @app.get("/health")
    async def health(request: Request):
        predictor = getattr(request.app.state, "predictor", None)
        return {
            "status": "ok" if predictor is not None else "starting",
            "model_loaded": predictor is not None,
            "class_count": (
                len(predictor.class_names)
                if predictor is not None
                else 0
            )
        }

    @app.post("/predict")
    async def predict(
        request: Request,
        image: UploadFile = File(...),
        top_k: int = Query(default=3, ge=1, le=10)
    ):
        predictor = getattr(request.app.state, "predictor", None)
        if predictor is None:
            raise HTTPException(
                status_code=503,
                detail="Model is not loaded."
            )
        if image.content_type not in ("image/jpeg", "image/png"):
            raise HTTPException(
                status_code=415,
                detail="Upload a JPEG or PNG image."
            )

        image_bytes = await image.read(MAX_IMAGE_BYTES + 1)
        if len(image_bytes) > MAX_IMAGE_BYTES:
            raise HTTPException(
                status_code=413,
                detail="Image exceeds the 10 MB size limit."
            )

        try:
            predictions = predictor.predict(image_bytes, top_k=top_k)
        except ValueError as error:
            raise HTTPException(status_code=400, detail=str(error)) from error

        return {
            "filename": image.filename,
            "predictions": predictions
        }

    @app.post("/retraining/uploads", status_code=201)
    async def upload_retraining_data(
        file: UploadFile = File(...),
        x_admin_token: str = Header(default=None)
    ):
        _require_admin_token(x_admin_token)
        if not file.filename or Path(file.filename).suffix.lower() != ".zip":
            raise HTTPException(
                status_code=415,
                detail="Upload a labelled ZIP archive."
            )

        staging_directory = PROJECT_DIRECTORY / "data" / "upload_staging"
        staging_directory.mkdir(parents=True, exist_ok=True)
        temporary_path = None
        archive_size = 0
        try:
            with tempfile.NamedTemporaryFile(
                mode="wb",
                suffix=".zip",
                dir=staging_directory,
                delete=False
            ) as temporary_file:
                temporary_path = Path(temporary_file.name)
                while True:
                    chunk = await file.read(1024 * 1024)
                    if not chunk:
                        break
                    archive_size += len(chunk)
                    if archive_size > MAX_ARCHIVE_BYTES:
                        raise HTTPException(
                            status_code=413,
                            detail="The uploaded ZIP exceeds the 100 MB limit."
                        )
                    temporary_file.write(chunk)

            result = save_uploaded_dataset(
                PROJECT_DIRECTORY,
                temporary_path
            )
        except ValueError as error:
            raise HTTPException(status_code=400, detail=str(error)) from error
        finally:
            if temporary_path is not None:
                temporary_path.unlink(missing_ok=True)
            await file.close()

        return result

    @app.get("/retraining/uploads/{upload_id}")
    async def retraining_status(upload_id: str):
        try:
            record = get_upload_record(PROJECT_DIRECTORY, upload_id)
        except KeyError as error:
            raise HTTPException(status_code=404, detail=str(error)) from error

        return {
            "upload_id": record["upload_id"],
            "status": record["status"],
            "image_count": record["image_count"],
            "class_counts": record["class_counts"],
            "result": record.get("result"),
            "error": record.get("error")
        }

    @app.post("/retraining/uploads/{upload_id}/trigger", status_code=202)
    async def trigger_retraining_job(
        upload_id: str,
        request: Request,
        background_tasks: BackgroundTasks,
        x_admin_token: str = Header(default=None)
    ):
        _require_admin_token(x_admin_token)
        try:
            trigger_retraining(PROJECT_DIRECTORY, upload_id)
        except KeyError as error:
            raise HTTPException(status_code=404, detail=str(error)) from error
        except ValueError as error:
            raise HTTPException(status_code=409, detail=str(error)) from error

        def run_retraining_job():
            import tensorflow as tf

            try:
                result = retrain_uploaded_dataset(
                    PROJECT_DIRECTORY,
                    upload_id
                )
            except (
                OSError,
                ValueError,
                RuntimeError,
                KeyError,
                tf.errors.OpError
            ) as error:
                update_upload_record(
                    PROJECT_DIRECTORY,
                    upload_id,
                    status="failed",
                    error=f"{type(error).__name__}: {error}"
                )
                raise

            if result["promoted"]:
                request.app.state.predictor = predictor_loader(
                    PROJECT_DIRECTORY
                )

        background_tasks.add_task(run_retraining_job)
        return {
            "upload_id": upload_id,
            "status": "queued"
        }

    return app


app = create_app()
