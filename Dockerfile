FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

RUN sed -i 's|http://deb.debian.org|https://deb.debian.org|g' /etc/apt/sources.list.d/debian.sources \
    && apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements-api.txt .
RUN python -m pip install --no-cache-dir -r requirements-api.txt

RUN groupadd --gid 10001 app \
    && useradd --uid 10001 --gid 10001 --create-home app \
    && mkdir -p /app/data /app/models /opt/plant-disease-seed/models \
    && chown -R app:app /app /opt/plant-disease-seed

COPY api ./api
COPY src ./src
COPY --chown=app:app models/class_names.json /opt/plant-disease-seed/models/class_names.json
COPY --chown=app:app models/plant_disease_classifier.keras /opt/plant-disease-seed/models/plant_disease_classifier.keras
COPY docker/api-entrypoint.sh /usr/local/bin/api-entrypoint
RUN sed -i 's/\r$//' /usr/local/bin/api-entrypoint \
    && chmod 0555 /usr/local/bin/api-entrypoint

USER app
EXPOSE 8000

ENTRYPOINT ["/usr/local/bin/api-entrypoint"]
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
