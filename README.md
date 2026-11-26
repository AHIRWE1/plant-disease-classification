# Plant Disease Classification

An end-to-end image-classification project for identifying plant leaf
conditions using the PlantVillage dataset, built for the African Leadership
University BSE Machine Learning Pipeline summative assignment.

## Dataset acquisition

The acquisition script loads PlantVillage through TensorFlow Datasets and
exports a reproducible subset into class-named folders. By default, it keeps
at most 200 images per each of the 38 classes (up to 7,600 images total).
Classes with fewer examples keep all their available images. A fixed random
seed makes the per-class sample reproducible. The archive's background-only
folder is excluded, leaving the 38 disease/healthy classes.

On Windows, from the project directory:

```powershell
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python -m src.data_acquisition --max-images-per-class 200 --seed 42
```

TensorFlow Datasets downloads the source dataset into `data/tfds_cache/`.
Exported JPEG images and `dataset_metadata.json` are written under
`data/raw/`. The metadata records the selected and available image counts for
each class, the per-class cap, and the sampling seed. Acquisition validates
that exported images can be opened.

The output directory must be empty before acquisition. This prevents a second
run from mixing new images with files from an earlier run. If a run is
interrupted, inspect `data/raw/` before clearing or moving its contents and
retrying.

## Notebook walkthrough

Open [notebook/plant_disease_classification.ipynb](./notebook/plant_disease_classification.ipynb)
in Google Colab and run its cells from top to bottom. The setup cell clones
this repository and installs missing notebook dependencies; the acquisition
cell downloads PlantVillage through TensorFlow Datasets into the temporary
Colab runtime, so you do not need to upload dataset files. Select a GPU
runtime when available. The notebook covers preprocessing, reproducible
splits, visualizations, train-only augmentation, a frozen-versus-fine-tuned
comparison, held-out evaluation, class-level error analysis, and one
prediction example. Colab storage is temporary; copy or download the model
and evaluation results before the runtime is deleted.

## Create train, validation, and test splits

After acquisition, create a deterministic, stratified split manifest:

```powershell
python -m src.preprocessing --seed 42
```

This writes `data/processed/split_manifest.csv` and
`data/processed/split_metadata.json`. The CSV assigns each image to one split
without copying the source images. The default split is 70% training, 15%
validation, and 15% testing. The same seed reproduces the same assignments.
Training code can load MobileNetV2-preprocessed TensorFlow datasets with
`load_split_datasets` from `src.preprocessing`.

## Train and evaluate the classifier

After creating the split manifest, run:

```powershell
python -m src.training --head-epochs 5 --fine-tune-epochs 3 --seed 42
```

By default, training uses an ImageNet-pretrained MobileNetV2 feature extractor,
trains a custom 38-class head, and then fine-tunes the last 20 eligible layers.
Early stopping monitors validation loss. The held-out test split is reserved
for final evaluation. The trained model and metrics are saved under `models/`,
and the confusion matrix is saved under `visualizations/`.

The evaluation report includes test loss, accuracy, macro- and weighted-
averaged precision, recall, and F1-score, plus per-class metrics. The optional
`--weights none` setting is intended for local architecture checks; use the
default ImageNet weights for the transfer-learning assignment workflow.

### Saved model results

The saved local model report in `models/evaluation_metrics.json` records
accuracy **0.9303**, macro precision **0.9315**, macro recall **0.9302**,
macro F1 **0.9286**, and test loss **0.2010** on 1,133 held-out images from a
seed-42 70/15/15 split. The executed Colab notebook currently reports a
separate run at **0.6390 accuracy** and **0.5896 macro F1**. These are
different model runs and must not be presented as one result; deploy a model
only with the evaluation report produced for that same model and split. The
model binary is excluded from Git and must be transferred separately.

## Docker and local smoke test

The API and Streamlit dashboard have separate images. The API image includes
the saved model and class labels, while local data/model directories are
mounted into the API container. Before building, make sure
`models/plant_disease_classifier.keras` exists locally. Create a local token
file and start both services:

```powershell
Copy-Item .env.example .env
# Edit .env and replace the example token before using retraining.
docker compose config
docker compose up --build
```

In another terminal, confirm the API responds:

```powershell
Invoke-RestMethod http://localhost:8000/health
```

The dashboard opens at `http://localhost:8501`. Stop the services with
Ctrl+C, then run `docker compose down`. Never commit `.env`.

Install Locust separately and run it against the API:

```powershell
python -m pip install -r requirements-loadtest.txt
locust -f locustfile.py --host http://127.0.0.1:8000
```

Open `http://127.0.0.1:8089`, set the user count and spawn rate, and run a
flood test. Record p50/p95/p99 latency, mean/median response time,
requests per second, and failures for one container and additional container
counts. The Locust script submits one synthetic 224 x 224 JPEG per request.

## AWS EC2 deployment preparation

The current deployment target is one Ubuntu 26.04 EC2 instance running the
API and Streamlit dashboard with Docker Compose. This setup uses the
instance's EBS-backed project directory for persistent `data/` and `models/`;
it does not use ECS or EFS and does not provide automatic scaling. Review
cost alerts, enable MFA, and restrict SSH/dashboard access before launching
the public demo. The API port is bound to the EC2 host's loopback interface;
only the dashboard port is published. See [aws/README.md](./aws/README.md)
for artifact transfer, SSH, Docker installation, security-group, smoke-test,
and cleanup instructions.

## Dataset visualizations

After acquiring the subset, run:

```powershell
python -m src.visualization --seed 42
```

The script writes a class-count chart, a labelled image sample grid, and a
comparison of three measurable image properties: mean brightness,
green-channel ratio, and edge contrast. It also writes class-level feature
means to `visualizations/image_feature_summary.csv` and computes a short
healthy-versus-diseased interpretation in
`visualizations/feature_interpretations.txt`. These are descriptive dataset
statistics, not causal explanations of plant disease or the model.

## Run the prediction API and dashboard

Install dependencies, then start the API in one PowerShell terminal:

```powershell
.\.venv\Scripts\Activate.ps1
uvicorn api.main:app --reload
```

In another terminal, start the dashboard:

```powershell
.\.venv\Scripts\Activate.ps1
streamlit run dashboard/app.py
```

The API health page is available at `http://127.0.0.1:8000/health` and the
interactive API documentation at `http://127.0.0.1:8000/docs`. The dashboard
opens in the browser, checks API health, accepts a JPEG/PNG leaf image, and
displays the top three predictions and confidence values.

For the labelled ZIP upload and retraining controls, configure an admin token
in the API terminal before starting Uvicorn:

```powershell
$env:PLANT_DISEASE_ADMIN_TOKEN = "choose-a-local-demo-token"
uvicorn api.main:app --reload
```

Enter the same token in the dashboard. The ZIP must contain folders matching
the known class names, with JPEG/PNG images inside each folder. Valid images
are saved under `data/uploaded/`, and upload status/class counts are recorded
in `data/retraining.sqlite3`. Retraining fine-tunes the current model and
promotes it only if validation accuracy does not decrease and validation loss
is no more than 0.05 above the baseline. The test split is not used for this
decision.

## Current status

- [x] Python 3.11 environment and dependency list
- [x] PlantVillage acquisition with a reproducible per-class image cap
- [x] Stratified train/validation/test manifest and image preprocessing loader
- [x] Class, sample-image, and three image-feature visualizations
- [x] MobileNetV2 training, fine-tuning, and held-out evaluation pipeline
- [x] FastAPI image-prediction service and Streamlit dashboard
- [x] Protected labelled ZIP upload, persisted upload tracking, and guarded retraining
- [ ] Container deployment and Locust load testing

Do not commit dataset images, TFDS cache, uploaded retraining data, local
secrets, or candidate model versions. The active `.keras` model is ignored
because of its size; make it available locally before building the API image.
The small class-label metadata, evaluation report, and selected dashboard
charts are kept as project artifacts. See the dataset source and terms at the
[TensorFlow Datasets PlantVillage page](https://www.tensorflow.org/datasets/catalog/plant_village).