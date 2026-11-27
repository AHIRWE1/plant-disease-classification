# Plant Disease Classification

An end-to-end image-classification project for identifying plant leaf
conditions using the PlantVillage dataset, built for the African Leadership
University BSE Machine Learning Pipeline summative assignment.

## Live deployment

The API and Streamlit dashboard are deployed with Docker Compose on the
`plant-disease-server` EC2 instance in `eu-north-1`. The current public IPv4
address is `13.60.40.48` (it can change after the instance is stopped and
started because no Elastic IP is attached).

- Dashboard: [http://13.60.40.48:8501](http://13.60.40.48:8501/)
- API documentation: [http://13.60.40.48:8000/docs](http://13.60.40.48:8000/docs)
- API health: [http://13.60.40.48:8000/health](http://13.60.40.48:8000/health)
- Demo video: [Google Docs demo link](https://docs.google.com/document/d/1wiL5CXuFe5ELSJ_9Hl7KA3hce7KeLq6vHsSGi71PwIU/edit?usp=sharing)

These endpoints currently use plain HTTP. The EC2 security group allows
ports 8000 and 8501 from the internet for the demo. This is not a hardened
production configuration; see [AWS EC2 deployment](#aws-ec2-deployment)
below for the exact rules, safer SSH-tunnel access, and security limitations.

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

The original saved model report in `models/evaluation_metrics.json` records
accuracy **0.9303**, macro precision **0.9315**, macro recall **0.9302**,
macro F1 **0.9286**, and test loss **0.2010** on 1,133 held-out images from a
seed-42 70/15/15 split. That report predates the latest retraining. The
promoted candidate reached **0.9312 validation accuracy** and **0.2589
validation loss**, compared with baseline validation accuracy **0.9241** and
loss **0.2676**. These are validation results, not test results; evaluate the
promoted model on the held-out test split before reporting its test metrics.
The executed Colab notebook reports a separate run at **0.6390 accuracy** and
**0.5896 macro F1**. Keep the results for these different runs distinct.
`models/plant_disease_classifier.keras` (about 22 MB) is currently tracked in
Git so a fresh clone can build the API image.

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
To target the deployed EC2 API instead of the local API, use:

```powershell
locust -f locustfile.py --host http://13.60.40.48:8000
```

Run a controlled test against this project only. The deployed API is publicly
reachable over plain HTTP and runs on a single instance. Report measured
latency, throughput, failures, and container count; do not claim results for
multiple container counts unless those configurations were actually tested.

## AWS EC2 deployment

### Deployed environment and URLs

The API and dashboard run with Docker Compose on one Ubuntu 26.04
`m7i-flex.large` EC2 instance named `plant-disease-server` in `eu-north-1`.
The instance's EBS-backed project directory persists `data/` and `models/`;
the deployment does not use ECS or EFS and has no automatic scaling.

The current public IPv4 address and application/video links are listed under
[Live deployment](#live-deployment). The address can change if the instance
is stopped and started because no Elastic IP is attached.

These endpoints use plain HTTP. The deployed security group `plant-sg`
currently has these inbound rules:

| Protocol/port | Source | Purpose |
| --- | --- | --- |
| TCP 22 | EC2 Instance Connect prefix list for `eu-north-1` | Browser-based console connection |
| TCP 8000 | `0.0.0.0/0` | Public API, Swagger UI, and health check |
| TCP 8501 | `0.0.0.0/0` | Public dashboard |

This is a permissive demo configuration, not a production security setup.
Port 8000 exposes unauthenticated prediction requests to the internet, and
plain HTTP does not encrypt dashboard traffic. Do not enter the retraining
token through the public dashboard. For safer use, remove public ingress on
8000 and access the API through an SSH tunnel; restrict the dashboard or put
it behind HTTPS and appropriate access controls. A local SSH tunnel requires
a temporary TCP 22 security-group rule for your current public IP; the EC2
Instance Connect prefix-list rule alone only allows the browser-based console
connection. Remove the temporary rule when finished.

If replacing an existing SSH rule that uses an IP address, delete it and add
a new rule with the EC2 Instance Connect prefix list; the source type cannot
be changed in place.

### AWS account and instance preparation

- Set an AWS Budget and actual/forecasted cost alerts. Credits do not
  guarantee that every resource is free, alerts can be delayed, and an alert
  does not stop resources. See
  [AWS Budgets](https://docs.aws.amazon.com/cost-management/latest/userguide/budgets-managing-costs.html).
- Enable MFA for the AWS identity used to administer the account. See
  [AWS MFA guidance](https://docs.aws.amazon.com/IAM/latest/UserGuide/id_credentials_mfa.html).
- Check EBS free space before building or downloading the dataset. The API
  image is about 1.85 GB; build layers, the dataset, and TFDS cache need more
  space. Dataset acquisition and retraining also use CPU, memory, and disk.
  An instance with about 1 GB of memory may need swap or a larger instance
  before a TensorFlow build.

The active model `models/plant_disease_classifier.keras` (about 22 MB) is
tracked in Git and is included in a fresh clone. The original
`models/evaluation_metrics.json` reports 93.03% test accuracy, but it predates
the latest retraining. The promoted candidate reached 93.12% validation
accuracy (baseline 92.41%) and validation loss 0.2589 (baseline 0.2676).
These are validation results, not test results; evaluate the promoted model
on the held-out test split before claiming its test performance. The Colab
notebook's 63.90% accuracy is from a separate model run.

### Connect to EC2 and install Docker

The instance was managed from the AWS console using EC2 Instance Connect:
open **EC2 → Instances → Connect → EC2 Instance Connect**, select
`plant-disease-server`, and use username `ubuntu`. The SSH security-group
source is the `com.amazonaws.eu-north-1.ec2-instance-connect` prefix list,
not a personal IP address.

Install Docker Engine and the Compose plugin from Docker's official
[Ubuntu installation guide](https://docs.docker.com/engine/install/ubuntu/)
using its APT repository (Ubuntu 26.04 is supported). Keep using `sudo docker`
unless you have deliberately reviewed the security implications of granting
a user Docker access.

### Clone the repository and start the services

The EC2 clone only receives committed files. Commit and push the deployment
code before cloning:

```bash
git clone https://github.com/AHIRWE1/plant-disease-classification.git
cd plant-disease-classification
mkdir -p data models
ls -lh models/plant_disease_classifier.keras
```

The committed `docker-compose.yml` binds API port 8000 to loopback. On the
deployed instance, the API was changed to a public host binding so the API
docs and health URL are directly reachable. Reproduce that change with:

```bash
sed -i 's/127.0.0.1:8000:8000/8000:8000/' docker-compose.yml
```

This publishes the API on all host interfaces. The security-group rule also
allows public access, so use it only for the demo and see the warning above.
Create the admin-token file once, then build and start both services:

```bash
sudo chown -R 10001:10001 data models
printf 'PLANT_DISEASE_ADMIN_TOKEN=%s\n' "$(openssl rand -hex 32)" > .env
chmod 600 .env
sudo docker compose config --quiet
sudo docker compose up -d --build
sudo docker compose ps
curl http://127.0.0.1:8000/health
```

The health response should be
`{"status":"ok","model_loaded":true,"class_count":38}`. If startup fails,
inspect the service logs:

```bash
sudo docker compose logs --tail 100 api dashboard
```

The generated admin token is stored in the EC2 checkout's `.env`. Retrieve
it in the private EC2 terminal when you need the retraining UI:

```bash
sed -n 's/^PLANT_DISEASE_ADMIN_TOKEN=//p' .env
```

Copy only the printed value into the dashboard's **Retraining admin token**
field. Do not paste it into chat, screenshots, or GitHub, and never commit
`.env`. If rotating the token, write a new value and recreate the API:

```bash
printf 'PLANT_DISEASE_ADMIN_TOKEN=%s\n' "$(openssl rand -hex 32)" > .env
chmod 600 .env
sudo docker compose up -d --force-recreate api
```

### Safer dashboard and API access

For private access, allow TCP 22 temporarily from your current public IP in
addition to the EC2 Instance Connect prefix-list rule. From PowerShell, open
an SSH tunnel:

```powershell
ssh -i "C:\path\to\your-key.pem" `
  -L 8501:127.0.0.1:8501 `
  -L 8000:127.0.0.1:8000 `
  ubuntu@<EC2_PUBLIC_IPV4>
```

While the tunnel is open, browse to `http://127.0.0.1:8501` for the dashboard
or `http://127.0.0.1:8000/docs` for the API docs. The dashboard calls the API
over the internal Compose address `http://api:8000`. Remove the temporary SSH
security-group rule when finished. Do not enter the retraining token through
the public plain-HTTP dashboard.

### Retraining, load testing, and cleanup

Retraining also requires the PlantVillage source dataset and processed split
manifest. If they are absent, check free disk space with `df -h /` before
acquiring the dataset and preparing splits:

```bash
sudo docker compose exec api python -m src.data_acquisition --max-images-per-class 200 --seed 42
sudo docker compose exec api python -m src.preprocessing --seed 42
```

Keep only one API instance during retraining. This deployment runs one API
container; comparing multiple container counts requires a controlled setup
with a reverse proxy/load balancer. Use Locust against the deployed API only
for a controlled test of this project:

```powershell
locust -f locustfile.py --host http://13.60.40.48:8000
```

Record actual latency, throughput, failures, and container counts. Do not
claim measurements for configurations that were not tested. The Locust
script submits synthetic 224 x 224 JPEG requests.

Stop the containers when finished:

```bash
sudo docker compose down
```

This does not stop the EC2 instance or remove EBS storage. Stop or terminate
the instance in the AWS console when the demo is over, and review billing;
retained EBS storage can continue to incur charges. Take an EBS snapshot
before deleting the instance if you need to preserve its data.

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
- [x] Docker Compose deployment on AWS EC2 (API and dashboard)
- [ ] Locust latency and throughput results by container count

Do not commit dataset images, TFDS cache, uploaded retraining data, local
secrets, or candidate model versions. The active
`models/plant_disease_classifier.keras` is currently tracked; new candidate
and versioned models are ignored. The small class-label metadata, evaluation
report, and selected dashboard charts are kept as project artifacts. See the
dataset source and terms at the
[TensorFlow Datasets PlantVillage page](https://www.tensorflow.org/datasets/catalog/plant_village).
