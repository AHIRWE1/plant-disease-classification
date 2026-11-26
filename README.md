Plant Disease Classification
An end-to-end image-classification project for identifying plant leaf conditions using the PlantVillage dataset, built for the African Leadership University BSE Machine Learning Pipeline summative assignment.
Live deployment
The API and dashboard are deployed on an AWS EC2 instance (plant-disease-server, region eu-north-1) with Docker Compose:
Service
Link to googledoc that contains the demo video: https://docs.google.com/document/d/1wiL5CXuFe5ELSJ_9Hl7KA3hce7KeLq6vHsSGi71PwIU/edit?usp=sharing
URL
Streamlit dashboard
http://13.60.40.48:8501
API documentation (Swagger UI)
http://13.60.40.48:8000/docs
API health check
http://13.60.40.48:8000/health

The links use plain HTTP (no TLS). The public IP changes if the instance is stopped and started again, unless an Elastic IP is attached. See AWS EC2 deployment for how it was set up.
Dataset acquisition
The acquisition script loads PlantVillage through TensorFlow Datasets and exports a reproducible subset into class-named folders. By default, it keeps at most 200 images per each of the 38 classes (up to 7,600 images total). Classes with fewer examples keep all their available images. A fixed random seed makes the per-class sample reproducible. The archive's background-only folder is excluded, leaving the 38 disease/healthy classes.
On Windows, from the project directory:
```powershell
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python -m src.data_acquisition --max-images-per-class 200 --seed 42
```

TensorFlow Datasets downloads the source dataset into data/tfds_cache/. Exported JPEG images and dataset_metadata.json are written under data/raw/. The metadata records the selected and available image counts for each class, the per-class cap, and the sampling seed. Acquisition validates that exported images can be opened.
The output directory must be empty before acquisition. This prevents a second run from mixing new images with files from an earlier run. If a run is interrupted, inspect data/raw/ before clearing or moving its contents and retrying.
Notebook walkthrough
Open notebook/plant_disease_classification.ipynb in Google Colab and run its cells from top to bottom. The setup cell clones this repository and installs missing notebook dependencies; the acquisition cell downloads PlantVillage through TensorFlow Datasets into the temporary Colab runtime, so you do not need to upload dataset files. Select a GPU runtime when available. The notebook covers preprocessing, reproducible splits, visualizations, train-only augmentation, a frozen-versus-fine-tuned comparison, held-out evaluation, class-level error analysis, and one prediction example. Colab storage is temporary; copy or download the model and evaluation results before the runtime is deleted.
Create train, validation, and test splits
After acquisition, create a deterministic, stratified split manifest:
```powershell
python -m src.preprocessing --seed 42
```

This writes data/processed/split_manifest.csv and data/processed/split_metadata.json. The CSV assigns each image to one split without copying the source images. The default split is 70% training, 15% validation, and 15% testing. The same seed reproduces the same assignments. Training code can load MobileNetV2-preprocessed TensorFlow datasets with load_split_datasets from src.preprocessing.
Train and evaluate the classifier
After creating the split manifest, run:
```powershell
python -m src.training --head-epochs 5 --fine-tune-epochs 3 --seed 42
```

By default, training uses an ImageNet-pretrained MobileNetV2 feature extractor, trains a custom 38-class head, and then fine-tunes the last 20 eligible layers. Early stopping monitors validation loss. The held-out test split is reserved for final evaluation. The trained model and metrics are saved under models/, and the confusion matrix is saved under visualizations/.
The evaluation report includes test loss, accuracy, macro- and weighted- averaged precision, recall, and F1-score, plus per-class metrics. The optional --weights none setting is intended for local architecture checks; use the default ImageNet weights for the transfer-learning assignment workflow.
Saved model results
The saved local model report in models/evaluation_metrics.json records accuracy 0.9303, macro precision 0.9315, macro recall 0.9302, macro F1 0.9286, and test loss 0.2010 on 1,133 held-out images from a seed-42 70/15/15 split. The executed Colab notebook currently reports a separate run at 0.6390 accuracy and 0.5896 macro F1. These are different model runs and must not be presented as one result; deploy a model only with the evaluation report produced for that same model and split. The deployed model, models/plant_disease_classifier.keras (about 22 MB), is committed to the repository so that a fresh clone can build the API image.
Docker and local smoke test
The API and Streamlit dashboard have separate images. The API image includes the saved model and class labels, while local data/model directories are mounted into the API container. Before building, make sure models/plant_disease_classifier.keras exists locally. Create a local token file and start both services:
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

The dashboard opens at http://localhost:8501. Stop the services with Ctrl+C, then run docker compose down. Never commit .env.
Install Locust separately and run it against the API:
```powershell
python -m pip install -r requirements-loadtest.txt
locust -f locustfile.py --host http://127.0.0.1:8000
```

Open http://127.0.0.1:8089, set the user count and spawn rate, and run a flood test. Record p50/p95/p99 latency, mean/median response time, requests per second, and failures for one container and additional container counts. The Locust script submits one synthetic 224 x 224 JPEG per request.
AWS EC2 deployment
The project runs on a single Ubuntu EC2 instance with Docker Compose. Both containers run on that instance; the dashboard reaches the API over the internal Compose network at http://api:8000. The instance's EBS volume holds the persistent data/ and models/ directories. There is no ECS, EFS, or automatic scaling.
Security group
Inbound rules on the instance's security group (plant-sg):
Type
Port
Source
Purpose
SSH
22
com.amazonaws.eu-north-1.ec2-instance-connect prefix list
Browser-based EC2 Instance Connect only
Custom TCP
8000
0.0.0.0/0
FastAPI service
Custom TCP
8501
0.0.0.0/0
Streamlit dashboard

SSH is limited to the EC2 Instance Connect prefix list, so the server is managed from the AWS console rather than from a personal IP address. A rule that already uses an IP address cannot be switched to a prefix list. Delete that rule and add a new SSH rule instead.
Deployment steps
Connect from the AWS console (EC2 → Instances → Connect → EC2 Instance Connect, username ubuntu). Then:
Install Docker Engine and the Compose plugin from Docker's Ubuntu APT repository.
Clone the repository. The trained model is included in the clone:
git clone https://github.com/AHIRWE1/plant-disease-classification.git
cd plant-disease-classification
mkdir -p data models
Publish the API port. In the committed docker-compose.yml the API is bound to 127.0.0.1:8000, so only the instance itself can reach it. The deployed server changes this to a public binding:
sed -i 's/127.0.0.1:8000:8000/8000:8000/' docker-compose.yml
Create the admin token, then build and start both services:
sudo chown -R 10001:10001 data models
printf 'PLANT_DISEASE_ADMIN_TOKEN=%s\n' "$(openssl rand -hex 32)" > .env
chmod 600 .env
sudo docker compose up -d --build
docker compose build only builds images; up -d --build builds them and starts the containers.
Verify the deployment:
sudo docker compose ps
curl http://127.0.0.1:8000/health
The API container should be healthy, and the health response should be {"status":"ok","model_loaded":true,"class_count":38}.
Useful commands on the instance:
sudo docker compose logs --tail 100 api dashboard   # inspect errors
sudo docker compose down                            # stop the services
On instances with about 1 GB of memory, add a swap file before building, or TensorFlow may run out of memory. Retraining also needs the dataset and split manifest. Create them on the instance only after checking free space with df -h /:
sudo docker compose exec api python -m src.data_acquisition --max-images-per-class 200 --seed 42
sudo docker compose exec api python -m src.preprocessing --seed 42
Stopping the containers does not stop the EC2 instance or its storage. Stop or terminate the instance in the AWS console when the demo is over, so it stops incurring charges. aws/README.md has the original deployment guide, including the SSH-tunnel alternative for a private dashboard.
Dataset visualizations
After acquiring the subset, run:
```powershell
python -m src.visualization --seed 42
```

The script writes a class-count chart, a labelled image sample grid, and a comparison of three measurable image properties: mean brightness, green-channel ratio, and edge contrast. It also writes class-level feature means to visualizations/image_feature_summary.csv and computes a short healthy-versus-diseased interpretation in visualizations/feature_interpretations.txt. These are descriptive dataset statistics, not causal explanations of plant disease or the model.
Run the prediction API and dashboard
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

The API health page is available at http://127.0.0.1:8000/health and the interactive API documentation at http://127.0.0.1:8000/docs. The dashboard opens in the browser, checks API health, accepts a JPEG/PNG leaf image, and displays the top three predictions and confidence values.
For the labelled ZIP upload and retraining controls, configure an admin token in the API terminal before starting Uvicorn:
```powershell
$env:PLANT_DISEASE_ADMIN_TOKEN = "choose-a-local-demo-token"
uvicorn api.main:app --reload
```

Enter the same token in the dashboard. The ZIP must contain folders matching the known class names, with JPEG/PNG images inside each folder. Valid images are saved under data/uploaded/, and upload status/class counts are recorded in data/retraining.sqlite3. Retraining fine-tunes the current model and promotes it only if validation accuracy does not decrease and validation loss is no more than 0.05 above the baseline. The test split is not used for this decision.
Current status
Python 3.11 environment and dependency list
PlantVillage acquisition with a reproducible per-class image cap
Stratified train/validation/test manifest and image preprocessing loader
Class, sample-image, and three image-feature visualizations
MobileNetV2 training, fine-tuning, and held-out evaluation pipeline
FastAPI image-prediction service and Streamlit dashboard
Protected labelled ZIP upload, persisted upload tracking, and guarded retraining
Docker Compose deployment on AWS EC2 (API and dashboard)
Locust load testing against the deployed API
Do not commit dataset images, TFDS cache, uploaded retraining data, local secrets, or candidate model versions. .gitignore excludes .keras files, so the deployed model was added with git add -f; retrained candidate and versioned models stay out of Git. The small class-label metadata, evaluation report, and selected dashboard charts are kept as project artifacts. See the dataset source and terms at the TensorFlow Datasets PlantVillage page.
