"""Train from scratch on the supplied splits; select using validation only."""
import argparse
import hashlib
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageOps
from torch import nn
from torch.utils.data import DataLoader, Dataset

from model import CLASSES, IMAGE_SIZE, ROOT, FruitCNN, image_transform, prepare_image


class FruitDataset(Dataset):
    def __init__(self, records, training=False):
        self.transform = image_transform(training)
        self.samples = []
        for path, label in records:
            with Image.open(path) as image:
                self.samples.append((prepare_image(image), label))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        image, label = self.samples[index]
        return self.transform(image), label


def audit_dataset(root):
    """Keep original splits, excluding repeated decoded images without deleting files."""
    seen, records = {}, {}
    report = {"counts": {}, "excluded_duplicates": [], "manifest": {}}
    for split in ("train", "validation", "test"):
        records[split] = []
        report["counts"][split] = {}
        for label, name in enumerate(CLASSES):
            directory = root / split / name
            if not directory.is_dir():
                raise ValueError(f"Missing dataset directory: {directory}")
            paths = sorted(p for p in directory.rglob("*") if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp", ".bmp"})
            kept = 0
            for path in paths:
                with Image.open(path) as image:
                    decoded = ImageOps.exif_transpose(image).convert("RGB")
                    digest = hashlib.sha256(str(decoded.size).encode() + decoded.tobytes()).hexdigest()
                relative = path.relative_to(root).as_posix()
                if digest in seen:
                    original, original_label = seen[digest]
                    if original_label != label:
                        raise ValueError(f"Conflicting labels for identical images: {original} and {relative}")
                    report["excluded_duplicates"].append({"excluded": relative, "retained": original})
                    continue
                seen[digest] = (relative, label)
                records[split].append((path, label))
                kept += 1
            if not kept:
                raise ValueError(f"No unique images in {directory}")
            report["counts"][split][name] = {"original": len(paths), "used": kept}
        report["manifest"][split] = [p.relative_to(root).as_posix() for p, _ in records[split]]
    return records, report


def run_epoch(model, loader, loss_fn, device, optimizer=None):
    model.train(optimizer is not None)
    loss_sum, correct, total = 0.0, 0, 0
    confusion = torch.zeros((2, 2), dtype=torch.int64)
    with torch.set_grad_enabled(optimizer is not None):
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            if optimizer is not None:
                optimizer.zero_grad(set_to_none=True)
            logits = model(images)
            loss = loss_fn(logits, labels)
            if optimizer is not None:
                loss.backward()
                optimizer.step()
            predicted = logits.argmax(dim=1)
            total += labels.size(0)
            loss_sum += loss.item() * labels.size(0)
            correct += (predicted == labels).sum().item()
            confusion += torch.bincount((2 * labels + predicted).cpu(), minlength=4).reshape(2, 2)
    return {"loss": loss_sum / total, "accuracy": correct / total,
            "count": total, "confusion_matrix": confusion.tolist()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=ROOT / "apple_banana_dataset")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "artifacts")
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--patience", type=int, default=7)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()
    if min(args.epochs, args.batch_size, args.patience, args.threads) < 1:
        parser.error("Epochs, batch size, patience, and threads must be positive.")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.set_num_threads(args.threads)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    records, audit = audit_dataset(args.data_dir)
    (args.output_dir / "dataset_audit.json").write_text(json.dumps(audit, indent=2), encoding="utf-8")
    print(f"Device: {device}; split counts: {audit['counts']}", flush=True)
    print(f"Excluded {len(audit['excluded_duplicates'])} exact duplicate images.", flush=True)
    # Test data is not evaluated until all training and checkpoint selection finish.
    loaders = {split: DataLoader(FruitDataset(records[split], split == "train"),
                                batch_size=args.batch_size, shuffle=split == "train", num_workers=0)
               for split in ("train", "validation")}
    model = FruitCNN().to(device)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.0001)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=2, factor=0.5)
    history, best_loss, stale, best_epoch = [], float("inf"), 0, 0
    started = time.perf_counter()
    checkpoint_path = args.output_dir / "fruit_cnn.pt"
    for epoch in range(1, args.epochs + 1):
        train = run_epoch(model, loaders["train"], loss_fn, device, optimizer)
        validation = run_epoch(model, loaders["validation"], loss_fn, device)
        scheduler.step(validation["loss"])
        history.append({"epoch": epoch, "train_loss": train["loss"], "train_accuracy": train["accuracy"],
                        "validation_loss": validation["loss"], "validation_accuracy": validation["accuracy"]})
        print(f"Epoch {epoch:02d}/{args.epochs}: train {train['accuracy']:.1%}, "
              f"validation {validation['accuracy']:.1%}, validation loss {validation['loss']:.4f}", flush=True)
        if validation["loss"] < best_loss:
            best_loss, stale, best_epoch = validation["loss"], 0, epoch
            torch.save({"state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()},
                        "classes": list(CLASSES), "image_size": IMAGE_SIZE,
                        "epoch": epoch, "seed": args.seed, "pretrained": False}, checkpoint_path)
        else:
            stale += 1
        (args.output_dir / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
        if stale >= args.patience:
            print("Early stopping: validation loss has stopped improving.", flush=True)
            break
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint["state_dict"])
    test_loader = DataLoader(FruitDataset(records["test"]), batch_size=args.batch_size)
    test = run_epoch(model, test_loader, loss_fn, device)
    metrics = {"architecture": "FruitCNN", "trained_from_scratch": True, "classes": list(CLASSES),
               "parameters": sum(p.numel() for p in model.parameters()), "seed": args.seed,
               "best_epoch": best_epoch, "epochs_completed": len(history), "device": str(device),
               "training_seconds": round(time.perf_counter() - started, 2),
               "validation": run_epoch(model, loaders["validation"], loss_fn, device),
               "test": test, "split_counts": audit["counts"],
               "excluded_duplicates": len(audit["excluded_duplicates"]),
               "confusion_matrix_axes": "rows=true, columns=predicted; apple, banana"}
    (args.output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(f"Best epoch: {best_epoch}. Held-out test accuracy: {test['accuracy']:.2%} ({test['count']} images).", flush=True)
    print(f"Saved model to {checkpoint_path}", flush=True)


if __name__ == "__main__":
    main()
