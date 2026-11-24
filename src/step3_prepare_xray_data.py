"""Step 3 - prepare the Indiana University chest X-ray data.

Input : the Kaggle dataset raddar/chest-xrays-indiana-university
        (indiana_reports.csv, indiana_projections.csv, PNG images)
Output: data/xray/xray_studies.csv      one row per study we use
        data/xray/images_512/           one smaller JPEG per study
        results/xray_data_summary.json  how many studies were kept and why

Run on Kaggle:  python src/step3_prepare_xray_data.py
Table only   :  python src/step3_prepare_xray_data.py --input <folder with the two CSVs> --skip-images
"""
import argparse
import json
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from text_metrics import clean_report_text

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "data" / "xray"
SEED = 42


def find_file(folder, name):
    """Search the input folder for a file, so no path is hard-coded."""
    matches = sorted(Path(folder).rglob(name))
    if not matches:
        raise FileNotFoundError(f"{name} not found under {folder}")
    return matches[0]


def build_study_table(reports, projections):
    """Join the two CSVs and apply the inclusion rules. Returns the table and the counts."""
    counts = {"reports_in_csv": int(len(reports)), "images_in_csv": int(len(projections))}

    frontal = projections[projections["projection"] == "Frontal"].sort_values("filename")
    frontal = frontal.drop_duplicates("uid", keep="first")            # one frontal image per study
    studies = reports.merge(frontal[["uid", "filename"]], on="uid", how="inner")
    counts["studies_with_frontal_image"] = int(len(studies))

    studies["findings"] = studies["findings"].map(clean_report_text)
    studies["impression"] = studies["impression"].map(clean_report_text)
    studies = studies[(studies["findings"] != "") & (studies["impression"] != "")]
    counts["with_findings_and_impression"] = int(len(studies))

    studies = studies[studies["Problems"].str.strip().str.lower() != "no indexing"]
    counts["after_removing_no_indexing"] = int(len(studies))

    studies = studies.copy()
    studies["label_abnormal"] = (studies["Problems"].str.strip().str.lower() != "normal").astype(int)
    studies["report_text"] = "findings: " + studies["findings"] + " impression: " + studies["impression"]
    studies["image_file"] = studies["uid"].astype(str) + ".jpg"

    # 80 / 10 / 10 split by study, same share of abnormal cases in each part
    train, rest = train_test_split(studies, test_size=0.20, random_state=SEED, stratify=studies["label_abnormal"])
    val, test = train_test_split(rest, test_size=0.50, random_state=SEED, stratify=rest["label_abnormal"])
    studies["split"] = "train"
    studies.loc[val.index, "split"] = "val"
    studies.loc[test.index, "split"] = "test"

    counts["split_sizes"] = studies["split"].value_counts().to_dict()
    counts["abnormal_share"] = round(float(studies["label_abnormal"].mean()), 3)
    counts["abnormal_share_by_split"] = studies.groupby("split")["label_abnormal"].mean().round(3).to_dict()
    counts["report_words_mean"] = round(float(studies["report_text"].str.split().str.len().mean()), 1)
    columns = ["uid", "filename", "image_file", "Problems", "findings", "impression",
               "report_text", "label_abnormal", "split"]
    return studies[columns].rename(columns={"filename": "source_png", "Problems": "problems"}), counts


def save_small_images(studies, source_dir, target_dir):
    """Shrink each PNG so its shorter side is 512 pixels and save it as JPEG."""
    from PIL import Image
    target_dir.mkdir(parents=True, exist_ok=True)
    for k, row in enumerate(studies.itertuples(), 1):
        image = Image.open(source_dir / row.source_png).convert("L")
        scale = 512 / min(image.size)
        image = image.resize((round(image.width * scale), round(image.height * scale)))
        image.save(target_dir / row.image_file, quality=90)
        if k % 500 == 0:
            print(f"  {k} / {len(studies)} images saved")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="/kaggle/input", help="folder that contains the Kaggle dataset")
    parser.add_argument("--skip-images", action="store_true", help="build the table only")
    args = parser.parse_args()

    reports = pd.read_csv(find_file(args.input, "*indiana_reports.csv"))
    projections = pd.read_csv(find_file(args.input, "*indiana_projections.csv"))
    studies, counts = build_study_table(reports, projections)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    studies.to_csv(OUT_DIR / "xray_studies.csv", index=False)
    (ROOT / "results").mkdir(exist_ok=True)
    (ROOT / "results" / "xray_data_summary.json").write_text(json.dumps(counts, indent=2))
    print(json.dumps(counts, indent=2))

    if not args.skip_images:
        source_dir = find_file(args.input, studies["source_png"].iloc[0]).parent
        print(f"Reading images from {source_dir}")
        save_small_images(studies, source_dir, OUT_DIR / "images_512")


if __name__ == "__main__":
    main()
