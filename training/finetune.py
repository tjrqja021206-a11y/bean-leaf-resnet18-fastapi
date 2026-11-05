"""예: python -m training.finetune --data dataset --epochs 3 --output artifacts/beans_resnet18"""
import argparse
import json
import random
from pathlib import Path
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from transformers import AutoImageProcessor, AutoModelForImageClassification

BASE_MODEL = "microsoft/resnet-18"
EXTS = {".jpg", ".jpeg", ".png", ".webp"}

class FolderImages(Dataset):
    def __init__(self, root: Path, classes: list[str], processor):
        self.records = []
        self.processor = processor
        for idx, name in enumerate(classes):
            folder = root / name
            if not folder.is_dir():
                raise ValueError(f"필요한 클래스 폴더가 없습니다: {folder}")
            self.records.extend((path, idx) for path in sorted(folder.iterdir())
                                if path.suffix.lower() in EXTS and path.is_file())
        if not self.records:
            raise ValueError(f"이미지가 없습니다: {root}")

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        path, label = self.records[index]
        with Image.open(path) as raw:
            image = raw.convert("RGB")
            tensor = self.processor(images=image, return_tensors="pt")["pixel_values"][0]
        return {"pixel_values": tensor, "labels": torch.tensor(label, dtype=torch.long)}

def run(args):
    random.seed(42)
    torch.manual_seed(42)

    data = Path(args.data)
    train_root = data / "train"
    if not train_root.is_dir():
        raise ValueError("dataset/train/ 폴더를 먼저 준비하세요.")

    classes = sorted(p.name for p in train_root.iterdir() if p.is_dir())
    if len(classes) < 2:
        raise ValueError("클래스 폴더가 2개 이상 필요합니다.")

    processor = AutoImageProcessor.from_pretrained(BASE_MODEL)
    train_set = FolderImages(train_root, classes, processor)
    val_set = FolderImages(data / "val", classes, processor)

    train_counts = {n: sum(label == i for _, label in train_set.records) for i, n in enumerate(classes)}
    val_counts = {n: sum(label == i for _, label in val_set.records) for i, n in enumerate(classes)}

    if any(n < 2 for n in train_counts.values()) or any(n < 1 for n in val_counts.values()):
        raise ValueError("각 클래스에 학습 최소 2장, 검증 최소 1장이 필요합니다.")

    print(f"클래스: {classes} / Train: {len(train_set)}, Val: {len(val_set)}")

    model = AutoModelForImageClassification.from_pretrained(
        BASE_MODEL,
        num_labels=len(classes),
        id2label={i: name for i, name in enumerate(classes)},
        label2id={name: i for i, name in enumerate(classes)},
        ignore_mismatched_sizes=True,
    )

    # 기존 1000-class 분류 헤드는 크기가 맞지 않아 새롭게 초기화됩니다.
    # 초보자 첫 학습은 Backbone을 고정하고 마지막 분류 헤드만 학습합니다.
    if args.freeze_backbone:
        for name, parameter in model.named_parameters():
            parameter.requires_grad = name.startswith("classifier.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    optim = torch.optim.AdamW(
        (p for p in model.parameters() if p.requires_grad),
        lr=args.lr
    )

    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False, num_workers=0)

    history = []
    best_acc = -1.0
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)

    for epoch in range(args.epochs):
        model.train()
        losses = []

        for batch in train_loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            optim.zero_grad()
            result = model(**batch)
            result.loss.backward()
            optim.step()
            losses.append(float(result.loss.item()))

        model.eval()
        correct = 0
        total = 0

        with torch.no_grad():
            for batch in val_loader:
                pixels = batch["pixel_values"].to(device)
                logits = model(pixel_values=pixels).logits
                correct += (logits.argmax(dim=-1).cpu() == batch["labels"]).sum().item()
                total += len(batch["labels"])

        acc = correct / total
        row = {
            "epoch": epoch + 1,
            "loss": round(sum(losses) / len(losses), 4),
            "val_accuracy": round(acc, 4)
        }
        history.append(row)
        print(row)

        if acc > best_acc:
            best_acc = acc
            model.save_pretrained(str(output), safe_serialization=True)
            processor.save_pretrained(str(output))

    (output / "training_report.json").write_text(
        json.dumps(
            {
                "base_model": BASE_MODEL,
                "classes": classes,
                "train_count": len(train_set),
                "val_count": len(val_set),
                "best_val_accuracy": best_acc,
                "history": history
            },
            ensure_ascii=False,
            indent=2
        ),
        encoding="utf-8"
    )

    print(f"저장 완료: {output} / best val accuracy={best_acc:.4f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="dataset")
    parser.add_argument("--output", default="artifacts/beans_resnet18")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--freeze-backbone", action="store_true")
    run(parser.parse_args())
