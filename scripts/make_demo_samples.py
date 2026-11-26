"""
Build demo samples for the video. Run from the project root:

    python scripts/make_demo_samples.py

Creates:
  demo_samples/predict/<class>.jpg   held-out TEST images that the current model
                                     classifies correctly (so the prediction demo
                                     shows a correct result).
  demo_samples/demo_retrain.zip      a small labelled ZIP for the upload +
                                     retraining demo (images from the TRAIN split,
                                     so the held-out test set stays untouched).
"""

import argparse
import csv
import random
import shutil
import sys
import zipfile
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))

# Classes the model gets (almost) always right on the test split.
PREDICT_CLASSES = [
    "Apple___Black_rot",
    "Blueberry___healthy",
    "Corn___Common_rust",
    "Grape___healthy",
    "Potato___Early_blight",
    "Squash___Powdery_mildew",
    "Strawberry___Leaf_scorch",
    "Tomato___Tomato_Yellow_Leaf_Curl_Virus",
]

# The model's weakest classes, which makes a good "we upload more examples" story.
RETRAIN_CLASSES = [
    "Tomato___Early_blight",
    "Tomato___Target_Spot",
    "Corn___Northern_Leaf_Blight",
]


def read_manifest():
    manifest_path = PROJECT / "data" / "processed" / "split_manifest.csv"
    if not manifest_path.is_file():
        raise SystemExit(f"Split manifest not found: {manifest_path}")
    with manifest_path.open(newline="", encoding="utf-8") as file:
        return list(csv.DictReader(file))


def main():
    parser = argparse.ArgumentParser(description="Create demo images and a retraining ZIP.")
    parser.add_argument("--min-confidence", type=float, default=0.90)
    parser.add_argument("--per-class", type=int, default=15,
                        help="Images per class in the retraining ZIP.")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    rows = read_manifest()
    by_split = {"test": {}, "train": {}}
    for row in rows:
        if row["split"] in by_split:
            by_split[row["split"]].setdefault(row["class_name"], []).append(
                row["relative_path"]
            )

    from src.prediction import load_predictor

    predictor = load_predictor(PROJECT)
    known_classes = set(predictor.class_names)

    output_directory = PROJECT / "demo_samples"
    predict_directory = output_directory / "predict"
    predict_directory.mkdir(parents=True, exist_ok=True)

    print("\n== Prediction samples (held-out test images, verified correct) ==")
    saved = 0
    for class_name in PREDICT_CLASSES:
        if class_name not in known_classes:
            print(f"  SKIP  {class_name}: not a known class")
            continue
        chosen = None
        for relative_path in sorted(by_split["test"].get(class_name, [])):
            image_path = PROJECT / relative_path
            if not image_path.is_file():
                continue
            top = predictor.predict(image_path.read_bytes(), top_k=1)[0]
            if (
                top["class_name"] == class_name
                and top["confidence"] >= args.min_confidence
            ):
                chosen = (image_path, top["confidence"])
                break
        if chosen is None:
            print(f"  SKIP  {class_name}: no test image met {args.min_confidence:.0%}")
            continue
        shutil.copy2(chosen[0], predict_directory / f"{class_name}.jpg")
        print(f"  OK    {class_name}  ({chosen[1]:.1%})")
        saved += 1

    print("\n== Retraining ZIP (train-split images, class folders) ==")
    rng = random.Random(args.seed)
    zip_path = output_directory / "demo_retrain.zip"
    total = 0
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as archive:
        for class_name in RETRAIN_CLASSES:
            if class_name not in known_classes:
                print(f"  SKIP  {class_name}: not a known class")
                continue
            paths = sorted(
                p for p in by_split["train"].get(class_name, [])
                if (PROJECT / p).is_file()
            )
            picks = rng.sample(paths, min(args.per_class, len(paths)))
            for index, relative_path in enumerate(picks):
                # Forward slashes inside the ZIP, as the API requires.
                archive.write(PROJECT / relative_path, arcname=f"{class_name}/{index:03d}.jpg")
            print(f"  OK    {class_name}: {len(picks)} images")
            total += len(picks)

    size_kb = zip_path.stat().st_size / 1024
    print(f"\nDone. {saved} prediction images, {total} images in the ZIP ({size_kb:.0f} KB).")
    print(f"Output folder: {output_directory}")


if __name__ == "__main__":
    main()