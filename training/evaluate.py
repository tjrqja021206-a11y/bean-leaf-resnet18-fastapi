"""예: python -m training.evaluate --models artifacts/m1_head_only artifacts/m2_full --split test"""
import argparse
from pathlib import Path
import torch
from PIL import Image
from transformers import AutoImageProcessor, AutoModelForImageClassification

EXTS = {'.jpg', '.jpeg', '.png', '.webp'}


def evaluate(model_dir: str, data: Path, split: str):
    processor = AutoImageProcessor.from_pretrained(model_dir)
    model = AutoModelForImageClassification.from_pretrained(model_dir)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device).eval()
    names = [model.config.id2label[i] for i in range(len(model.config.id2label))]
    matrix = [[0] * len(names) for _ in names]
    for true_idx, name in enumerate(names):
        folder = data / split / name
        for path in sorted(folder.iterdir()):
            if path.suffix.lower() not in EXTS:
                continue
            with Image.open(path) as raw:
                pixels = processor(images=raw.convert('RGB'), return_tensors='pt')['pixel_values'].to(device)
            with torch.no_grad():
                pred = model(pixel_values=pixels).logits.argmax(-1).item()
            matrix[true_idx][pred] += 1
    total = sum(map(sum, matrix))
    correct = sum(matrix[i][i] for i in range(len(names)))
    return names, matrix, correct / total, total


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--models', nargs='+', required=True)
    parser.add_argument('--data', default='dataset')
    parser.add_argument('--split', default='test')
    args = parser.parse_args()
    for model_dir in args.models:
        names, matrix, acc, total = evaluate(model_dir, Path(args.data), args.split)
        print(f'\n=== {model_dir}  [{args.split}: {total}장]  accuracy = {acc:.4f}')
        print('정답\예측 ' + ' '.join(f'{n[:12]:>12}' for n in names))
        for i, row in enumerate(matrix):
            recall = row[i] / max(sum(row), 1)
            print(f'{names[i][:12]:>12} ' + ' '.join(f'{v:>12}' for v in row) + f'   재현율 {recall:.2f}')
