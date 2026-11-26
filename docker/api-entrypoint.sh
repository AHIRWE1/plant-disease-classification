#!/bin/sh
set -eu

mkdir -p /app/data /app/models

for artifact in class_names.json plant_disease_classifier.keras; do
    if [ ! -f "/app/models/${artifact}" ]; then
        cp "/opt/plant-disease-seed/models/${artifact}" "/app/models/${artifact}"
    fi
done

exec "$@"
