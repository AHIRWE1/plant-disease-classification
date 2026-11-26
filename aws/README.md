# AWS EC2 deployment guide

This guide targets the single Ubuntu 26.04 EC2 instance described for this
project, using Docker Compose. It does not create AWS resources or deploy
anything. No EFS is used: `data/` and `models/` are bind-mounted from the
instance's EBS-backed project directory.

The API is bound to the EC2 host's loopback interface on port 8000. The
dashboard is published on port 8501. This is a single-instance demo, not a
high-availability or horizontally scaled deployment.

## Before deployment

1. **Set a cost budget and alerts.** Create an AWS Budget with actual and
   forecasted cost notifications at thresholds you choose. Credits do not
   guarantee that every resource is free, budget alerts are delayed, and an
   alert does not stop resources. Check costs after testing and stop/delete
   resources you no longer need. See
   [AWS Budgets](https://docs.aws.amazon.com/cost-management/latest/userguide/budgets-managing-costs.html).
2. **Enable MFA** for the AWS identity you use to administer the account.
   AWS recommends passkeys or security keys where available. See
   [AWS MFA guidance](https://docs.aws.amazon.com/IAM/latest/UserGuide/id_credentials_mfa.html).
3. **Review the EC2 security group.** Allow SSH (TCP 22) only from your
   current public IP. Do not add a public inbound rule for API port 8000.
   For initial dashboard testing, keep port 8501 closed and use the SSH
   tunnel below. If you later need a public demo URL, allow 8501 only for
   the required period and audience; it is plain HTTP and has no TLS.
4. **Check EBS free space before building or downloading data.** The API
   image is about 1.85 GB, and Docker build layers, the dataset, and TFDS
   cache require additional disk space. Dataset acquisition and retraining
   also consume CPU, memory, and disk. Do not start those jobs until the
   instance has sufficient free space.
5. **Use a verified model/report pair.** The local
   `models/evaluation_metrics.json` reports 93.0% accuracy and names
   `models/plant_disease_classifier.keras` as its model path. That model
   exists locally, so it is the recommended deployment candidate. The
   executed Colab notebook reports 63.9% for a separate run; do not replace
   the recommended model with the Colab export or claim its metrics for the
   local model. For submission consistency, evaluate the exact deployment
   model in the notebook on the matching test split. The API Dockerfile
   requires the model at `models/plant_disease_classifier.keras`; the model
   binary is excluded from Git and must be transferred separately.

## Prepare the repository and artifacts

The EC2 clone only receives committed files. Before cloning, commit and push
the deployment code, including `Dockerfile`, `Dockerfile.dashboard`,
`docker-compose.yml`, `docker/`, the split requirements files,
`models/class_names.json`, and the three dashboard assets:

- `visualizations/class_distribution.png`
- `visualizations/image_feature_comparisons.png`
- `visualizations/feature_interpretations.txt`

Keep `.env`, SSH keys, dataset/cache files, and model binaries out of Git.
The model is deliberately ignored; transfer the chosen model to the EC2
checkout after cloning. Do not upload both model files under the same name
without first deciding which one is intended for deployment.

## Connect and install Docker

Use the SSH key downloaded for the instance and the Ubuntu account name:

```powershell
ssh -i "C:\path\to\your-key.pem" ubuntu@<EC2_PUBLIC_IPV4>
```

Install Docker Engine and the Compose plugin using Docker's official
[Ubuntu installation guide](https://docs.docker.com/engine/install/ubuntu/),
which lists Ubuntu 26.04 as supported. Use Docker's APT repository, not the
convenience script for a long-running server. Keep using `sudo docker`
commands unless you have deliberately reviewed the security implications of
granting a user Docker access.

## Copy the model and start the services

On the EC2 instance, clone the latest pushed branch:

```bash
git clone https://github.com/AHIRWE1/plant-disease-classification.git
cd plant-disease-classification
mkdir -p data models
```

From a separate PowerShell window on your PC, copy the selected model into
that checkout. Replace the key path, public IP, and local model path:

```powershell
scp -i "C:\path\to\your-key.pem" `
  "C:\path\to\plant-disease-classification\models\plant_disease_classifier.keras" `
  ubuntu@<EC2_PUBLIC_IPV4>:~/plant-disease-classification/models/
```

Back on EC2, ensure the runtime user can write persistent artifacts, create a
private admin token file, and start the containers:

```bash
cd ~/plant-disease-classification
sudo chown -R 10001:10001 data models
printf 'PLANT_DISEASE_ADMIN_TOKEN=%s\n' "$(openssl rand -hex 32)" > .env
chmod 600 .env
sudo docker compose config --quiet
sudo docker compose up -d --build
sudo docker compose ps
curl http://127.0.0.1:8000/health
```

Never paste the token into chat or commit `.env`. The API health response
should report `status: ok`, `model_loaded: true`, and 38 classes. Review logs
if startup fails:

```bash
sudo docker compose logs --tail 100 api dashboard
```

## Test the dashboard safely

With inbound security-group access limited to SSH, run this from PowerShell
on your PC:

```powershell
ssh -i "C:\path\to\your-key.pem" -L 8501:127.0.0.1:8501 ubuntu@<EC2_PUBLIC_IPV4>
```

While that SSH session remains open, browse to
`http://127.0.0.1:8501`. Test prediction with a real leaf image. To provide a
public demo URL later, temporarily allow inbound TCP 8501 in the EC2 security
group and use `http://<EC2_PUBLIC_IPV4>:8501`. This is not HTTPS; do not use
it for sensitive data, and remove the public rule after the demo. The API
port 8000 should remain closed to the internet.

## Persistence, retraining, and cleanup

The Compose bind mounts keep `data/` and `models/` on the instance across
container restarts and rebuilds. They do not protect data if the instance or
its EBS volume is deleted; take an EBS snapshot if you need a backup. This
single-instance setup has no shared storage or automatic scaling.

Prediction needs the selected model and `models/class_names.json`. Retraining
also requires the source dataset and processed split manifest. If those are
not present, initialize them only after checking disk space:

```bash
sudo docker compose exec api python -m src.data_acquisition --max-images-per-class 200 --seed 42
sudo docker compose exec api python -m src.preprocessing --seed 42
```

Keep only one API instance during retraining. The assignment's comparison of
multiple container counts still needs a controlled load-test setup and a
reverse proxy/load balancer; this single-container EC2 startup does not
provide that comparison by itself.

Stop the containers when you finish:

```bash
sudo docker compose down
```

Stopping containers does not delete the EC2 instance or EBS storage. Review
the AWS billing console, and stop or terminate the instance when it is no
longer needed. Remember that attached storage and other retained resources
may continue to incur charges.
