"""Step 4 - fine-tune the report model on the Indiana chest X-rays.

Input : data/xray/xray_studies.csv, data/xray/images_512/   (from Step 3)
Output: models/                    best weights, network descriptions, tokenizer
        results/training_log.csv   loss after every epoch
        results/fig_training_loss.png

Needs a GPU (free Kaggle notebook).
Quick check :  python src/step4_train_report_model.py --smoke     (16 studies, 1 epoch)
Full run    :  python src/step4_train_report_model.py
"""
import argparse
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, get_linear_schedule_with_warmup

from report_model import T5_NAME, XrayReportModel, pick_device, save_model
from xray_dataset import EVAL_TRANSFORM, TRAIN_TRANSFORM, XrayDataset

ROOT = Path(__file__).resolve().parents[1]
XRAY_DIR = ROOT / "data" / "xray"
SEED = 42
BATCH_SIZE = 16
MAX_EPOCHS = 15
PATIENCE = 3              # stop after this many epochs without a better validation loss
CLASS_LOSS_WEIGHT = 1.0   # how much the normal/abnormal task counts next to the report task


def run_epoch(model, loader, device, optimizer=None, scheduler=None):
    """One pass over the data. With an optimizer the model learns; without, it is only measured."""
    training = optimizer is not None
    model.train(training)
    report_total, class_total, batches = 0.0, 0.0, 0
    for pixel_values, labels, abnormal in loader:
        pixel_values, labels, abnormal = pixel_values.to(device), labels.to(device), abnormal.to(device)
        with torch.set_grad_enabled(training):
            report_loss, abnormal_logit = model(pixel_values, labels)
            class_loss = F.binary_cross_entropy_with_logits(abnormal_logit, abnormal)
            loss = report_loss + CLASS_LOSS_WEIGHT * class_loss
        if training:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)   # stops one bad batch from wrecking the weights
            optimizer.step(); scheduler.step(); optimizer.zero_grad()
        report_total += report_loss.item(); class_total += class_loss.item(); batches += 1
    return report_total / batches, class_total / batches


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true", help="tiny run to check that everything works")
    parser.add_argument("--epochs", type=int, default=MAX_EPOCHS)
    args = parser.parse_args()

    torch.manual_seed(SEED)
    device = pick_device()
    print("Device:", device)

    studies = pd.read_csv(XRAY_DIR / "xray_studies.csv")
    train, val = studies[studies["split"] == "train"], studies[studies["split"] == "val"]
    epochs = args.epochs
    if args.smoke:
        train, val, epochs = train.head(16), val.head(8), 1

    tokenizer = AutoTokenizer.from_pretrained(T5_NAME)
    image_dir = XRAY_DIR / "images_512"
    train_loader = DataLoader(XrayDataset(train, image_dir, tokenizer, TRAIN_TRANSFORM),
                              batch_size=BATCH_SIZE, shuffle=True, num_workers=2, drop_last=True)
    val_loader = DataLoader(XrayDataset(val, image_dir, tokenizer, EVAL_TRANSFORM),
                            batch_size=BATCH_SIZE, num_workers=2)

    model = XrayReportModel().to(device)
    # gentler steps for the pretrained image model, larger steps for the text model and the new layers
    optimizer = torch.optim.AdamW([
        {"params": model.vit.parameters(), "lr": 2e-5},
        {"params": list(model.t5.parameters()) + list(model.project.parameters())
                   + list(model.abnormal_head.parameters()), "lr": 1e-4},
    ], weight_decay=0.01)
    total_steps = len(train_loader) * epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, int(0.1 * total_steps), total_steps)

    log, best_val, epochs_without_gain, started = [], float("inf"), 0, time.time()
    for epoch in range(1, epochs + 1):
        train_report, train_class = run_epoch(model, train_loader, device, optimizer, scheduler)
        val_report, val_class = run_epoch(model, val_loader, device)
        log.append({"epoch": epoch, "train_report_loss": train_report, "train_class_loss": train_class,
                    "val_report_loss": val_report, "val_class_loss": val_class,
                    "minutes": round((time.time() - started) / 60, 1)})
        print(log[-1])
        if val_report < best_val:
            best_val, epochs_without_gain = val_report, 0
            save_model(model, tokenizer, ROOT / "models")            # keep only the best version
        else:
            epochs_without_gain += 1
            if epochs_without_gain >= PATIENCE:
                print(f"Stopping early: no better validation loss for {PATIENCE} epochs.")
                break

    (ROOT / "results").mkdir(exist_ok=True)
    name = "training_log_smoke.csv" if args.smoke else "training_log.csv"
    log = pd.DataFrame(log)
    log.to_csv(ROOT / "results" / name, index=False)

    if not args.smoke:
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.plot(log["epoch"], log["train_report_loss"], marker="o", label="training")
        ax.plot(log["epoch"], log["val_report_loss"], marker="o", label="validation")
        ax.set_xlabel("epoch"); ax.set_ylabel("report loss"); ax.set_title("Fine-tuning: report loss per epoch"); ax.legend()
        fig.tight_layout(); fig.savefig(ROOT / "results" / "fig_training_loss.png", dpi=150); plt.close(fig)

    # one example, to see with our own eyes that the model writes text
    model.eval()
    pixel_values, labels, _ = next(iter(val_loader))
    token_ids, score = model.generate(pixel_values[:1].to(device))
    print("Example draft :", tokenizer.decode(token_ids[0], skip_special_tokens=True))
    print("Abnormal score:", round(score[0].item(), 3))


if __name__ == "__main__":
    main()
