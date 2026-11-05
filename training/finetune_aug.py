"""데이터 증강 Fine-tuning (제공 finetune.py는 수정하지 않고 확장한 별도 버전).

예) python -m training.finetune_aug --augment light --seed 42 --output artifacts/m3_full_aug_light
    python -m training.finetune_aug --augment none  --freeze-backbone      # 증강 없이 Head-only

- 증강은 train 데이터에만 적용한다. val/test는 원본 그대로 평가한다.
- 에폭마다 Validation Accuracy가 최고일 때만 모델을 저장한다(제공 코드와 같은 규칙).
- training_report.json에 실행 옵션(settings)·시드·소요시간까지 기록한다.
"""
import argparse
import json
import random
import time
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from transformers import AutoImageProcessor, AutoModelForImageClassification

BASE_MODEL = "microsoft/resnet-18"
EXTS = {".jpg", ".jpeg", ".png", ".webp"}
AUGMENT_LEVELS = ("none", "light", "medium")


def build_augment(level: str):
    """증강 강도. 잎 사진은 방향이 정해져 있지 않아 뒤집기·회전을 쓰고, 병반 색이 흐려지지 않게 색 변화는 약하게 둔다."""
    if level == "none":
        return None
    if level == "light":
        return transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomRotation(15),
            transforms.ColorJitter(brightness=0.15, contrast=0.15),
        ])
    if level == "medium":
        return transforms.Compose([
            transforms.RandomResizedCrop(224, scale=(0.7, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomRotation(30),
            transforms.ColorJitter(brightness=0.25, contrast=0.25, saturation=0.1),
        ])
    raise ValueError(f"알 수 없는 증강 강도: {level}")


class FolderImages(Dataset):
    def __init__(self, root: Path, classes: list, processor, augment=None):
        self.records = []
        self.processor = processor
        self.augment = augment
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
        if self.augment is not None:
            image = self.augment(image)
        tensor = self.processor(images=image, return_tensors="pt")["pixel_values"][0]
        return {"pixel_values": tensor, "labels": torch.tensor(label, dtype=torch.long)}


def run(args):
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    data = Path(args.data)
    train_root = data / "train"
    if not train_root.is_dir():
        raise ValueError("dataset/train/ 폴더를 먼저 준비하세요.")
    classes = sorted(p.name for p in train_root.iterdir() if p.is_dir())
    if len(classes) < 2:
        raise ValueError("클래스 폴더가 2개 이상 필요합니다.")

    processor = AutoImageProcessor.from_pretrained(BASE_MODEL)
    train_set = FolderImages(train_root, classes, processor, augment=build_augment(args.augment))
    val_set = FolderImages(data / "val", classes, processor)  # 검증은 증강하지 않는다
    print(f"클래스: {classes} / Train: {len(train_set)}, Val: {len(val_set)} / 증강: {args.augment} / 시드: {args.seed}", flush=True)

    model = AutoModelForImageClassification.from_pretrained(
        BASE_MODEL, num_labels=len(classes),
        id2label={i: n for i, n in enumerate(classes)}, label2id={n: i for i, n in enumerate(classes)},
        ignore_mismatched_sizes=True,
    )
    if args.freeze_backbone:
        for name, parameter in model.named_parameters():
            parameter.requires_grad = name.startswith("classifier.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    optim = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=args.lr)

    generator = torch.Generator().manual_seed(args.seed)
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, generator=generator)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    history, best_acc, best_epoch = [], -1.0, 0
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    started = time.time()

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
        correct = total = 0
        with torch.no_grad():
            for batch in val_loader:
                logits = model(pixel_values=batch["pixel_values"].to(device)).logits
                correct += (logits.argmax(dim=-1).cpu() == batch["labels"]).sum().item()
                total += len(batch["labels"])
        acc = correct / total
        row = {"epoch": epoch + 1, "loss": round(sum(losses) / len(losses), 4), "val_accuracy": round(acc, 4)}
        history.append(row)
        print(row, flush=True)

        if acc > best_acc:
            best_acc, best_epoch = acc, epoch + 1
            model.save_pretrained(str(output), safe_serialization=True)
            processor.save_pretrained(str(output))

    (output / "training_report.json").write_text(json.dumps({
        "base_model": BASE_MODEL, "classes": classes,
        "train_count": len(train_set), "val_count": len(val_set),
        "best_val_accuracy": best_acc, "best_epoch": best_epoch, "history": history,
        "settings": {"augment": args.augment, "seed": args.seed, "epochs": args.epochs,
                     "batch_size": args.batch_size, "lr": args.lr,
                     "freeze_backbone": bool(args.freeze_backbone), "num_workers": args.num_workers},
        "elapsed_sec": round(time.time() - started, 1), "device": str(device),
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"저장 완료: {output} / best val accuracy={best_acc:.4f} (epoch {best_epoch}) / {time.time() - started:.0f}초", flush=True)


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="dataset")
    parser.add_argument("--output", default="artifacts/exp_aug")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--augment", choices=AUGMENT_LEVELS, default="light")
    parser.add_argument("--freeze-backbone", action="store_true")
    parser.add_argument("--num-workers", type=int, default=4)
    return parser


if __name__ == "__main__":
    run(get_parser().parse_args())
