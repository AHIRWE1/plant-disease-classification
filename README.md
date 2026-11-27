# AWS EC2 deployment guide

This guide documents the actual single-instance deployment: the
`plant-disease-server` `m7i-flex.large` instance running Ubuntu 26.04 in
`eu-north-1`, with the API and Streamlit dashboard managed by Docker Compose.
It does not use ECS or EFS. The instance's EBS-backed project directory is
bind-mounted for persistent `data/` and `models/`. This is a demo deployment,
not a high-availability or horizontally scaled service.

The current public IPv4 address is `13.60.40.48`; it may change after a
stop/start because no Elastic IP is attached:

- Dashboard: [http://13.60.40.48:8501](http://13.60.40.48:8501/)
- API documentation: [http://13.60.40.48:8000/docs](http://13.60.40.48:8000/docs)
- API health: [http://13.60.40.48:8000/health](http://13.60.40.48:8000/health)
- Demo video: [Google Docs demo link](https://docs.google.com/document/d/1wiL5CXuFe5ELSJ_9Hl7KA3hce7KeLq6vHsSGi71PwIU/edit?usp=sharing)

All three endpoints use plain HTTP. The deployed security group currently
allows ports 8000 and 8501 from the internet. This exposes the unauthenticated
prediction API to arbitrary requests and does not encrypt the dashboard
traffic. Do not enter the retraining token through the public HTTP dashboard.
Use the SSH tunnel for token-protected operations, and restrict or remove
public ingress when it is not needed.

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
3. **Review the EC2 security group.** The current `plant-sg` rules are:

   | Protocol/port | Source | Current purpose |
   | --- | --- | --- |
   | TCP 22 | EC2 Instance Connect prefix list for `eu-north-1` | Browser-based console connection |
   | TCP 8000 | `0.0.0.0/0` | Public API, Swagger UI, and health check |
   | TCP 8501 | `0.0.0.0/0` | Public dashboard |

   The public API and dashboard rules are permissive and unencrypted. For
   safer use, remove public ingress on 8000 and access it through an SSH
   tunnel; restrict the dashboard or put it behind HTTPS and appropriate
   access controls. Keep port 22 restricted to the EC2 Instance Connect
   prefix list used by the console. A local SSH tunnel from your PC also
   requires a temporary TCP 22 rule for your current public IP; the
   Instance Connect prefix-list rule alone does not permit that connection.
   Remove the temporary rule when finished.
4. **Check EBS free space before building or downloading data.** The API
   image is about 1.85 GB, and Docker build layers, the dataset, and TFDS
   cache require additional disk space. Dataset acquisition and retraining
   also consume CPU, memory, and disk. Do not start those jobs until the
   instance has sufficient free space. On an instance with about 1 GB of
   memory, add swap or use a larger instance before building; TensorFlow may
   otherwise run out of memory.
5. **Use a verified model/report pair.** The tracked
   `models/plant_disease_classifier.keras` model is about 22 MB. The original
   `models/evaluation_metrics.json` reports 93.0% accuracy and names
   `models/plant_disease_classifier.keras` as its model path. That report is
   from before the latest retraining. The promoted candidate's validation
   accuracy was 93.12% (baseline 92.41%); these are not test metrics. The
   executed Colab notebook reports 63.9% for a separate run. Keep these
   results distinct and evaluate the exact deployed model on the held-out
   test split before claiming its test performance.

## Prepare the repository and artifacts

The EC2 clone only receives committed files. The current repository tracks
the 22 MB active model, so a fresh clone contains the image required by the
API Dockerfile. Before cloning, commit and push the deployment code,
including `Dockerfile`, `Dockerfile.dashboard`, `docker-compose.yml`,
`docker/`, the split requirements files, `models/class_names.json`, the
model, and the three dashboard assets:

- `visualizations/class_distribution.png`
- `visualizations/image_feature_comparisons.png`
- `visualizations/feature_interpretations.txt`

Keep `.env`, SSH keys, dataset/cache files, and retraining candidates/version
files out of Git. The currently active model is tracked; do not force-add
candidate models or backups.

## Connect and install Docker

The actual deployment was managed from the AWS console using EC2 Instance
Connect: open **EC2 → Instances → Connect → EC2 Instance Connect**, select
the instance, and use username `ubuntu`. The security group uses the
`com.amazonaws.eu-north-1.ec2-instance-connect` prefix list for TCP 22 rather
than a personal IP address.

If connecting from a local PowerShell terminal with an SSH key instead, use:

```powershell
ssh -i "C:\path\to\your-key.pem" ubuntu@<EC2_PUBLIC_IPV4>
```

Install Docker Engine and the Compose plugin using Docker's official
[Ubuntu installation guide](https://docs.docker.com/engine/install/ubuntu/),
which lists Ubuntu 26.04 as supported. Use Docker's APT repository, not the
convenience script for a long-running server. Keep using `sudo docker`
commands unless you have deliberately reviewed the security implications of
granting a user Docker access.

## Clone the repository and start the services

On the EC2 instance, clone the latest pushed branch. The model is included in
the repository; no separate `scp` model transfer is required:

```bash
git clone https://github.com/AHIRWE1/plant-disease-classification.git
cd plant-disease-classification
mkdir -p data models
ls -lh models/plant_disease_classifier.keras
```

The committed Compose file binds API port 8000 to loopback. The deployed EC2
instance changed it to a public host binding so the API docs and health URL
are reachable directly. To reproduce that actual deployment change:

```bash
sed -i 's/127.0.0.1:8000:8000/8000:8000/' docker-compose.yml
```

This publishes the API on all host interfaces. The security-group rule also
allows public access, so use it only for the demo and see the security warning
above. Ensure the runtime user can write persistent artifacts, create the
admin token file, and start the containers:

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

The generated token is stored in the EC2 checkout's `.env`. When you need to
use retraining, print its value in your private EC2 terminal:

```bash
sed -n 's/^PLANT_DISEASE_ADMIN_TOKEN=//p' .env
```

Copy only the printed value (not the variable name) into the dashboard's
**Retraining admin token** field. Do not paste it into chat, screenshots, or
GitHub, and never commit `.env`. The API health response should report
`status: ok`, `model_loaded: true`, and 38 classes. Review logs if startup
fails:

```bash
sudo docker compose logs --tail 100 api dashboard
```

## Test the dashboard and API

The actual deployment exposes the dashboard and API at the public links at
the top of this guide. For safer access, use a tunnel from PowerShell instead
of entering the retraining token over public HTTP:

```powershell
ssh -i "C:\path\to\your-key.pem" `
  -L 8501:127.0.0.1:8501 `
  -L 8000:127.0.0.1:8000 `
  ubuntu@<EC2_PUBLIC_IPV4>
```

While that SSH session remains open, browse to `http://127.0.0.1:8501` for
the dashboard or `http://127.0.0.1:8000/docs` for the API docs. The dashboard
uses the Compose network address `http://api:8000` to call the API.

If you rotate the token, update `.env` and recreate the API so it reads the
new value. Retrieve it only in the private EC2 terminal and enter it through
the SSH tunnel:

```bash
printf 'PLANT_DISEASE_ADMIN_TOKEN=%s\n' "$(openssl rand -hex 32)" > .env
chmod 600 .env
sudo docker compose up -d --force-recreate api
```

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
