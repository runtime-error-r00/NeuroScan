"""Model creation, checkpointing, training and evaluation shared by notebooks and the app."""

import timm
import torch
from sklearn.metrics import f1_score


def build_model(model_name, num_classes, pretrained=True):
    """timm backbone with a new classification head of num_classes outputs."""
    return timm.create_model(model_name, pretrained=pretrained, num_classes=num_classes)


def save_checkpoint(path, model, model_name, classes, img_size, epoch, val_metrics):
    """Save weights together with everything needed to rebuild the model."""
    torch.save({
        "model_name": model_name,
        "classes": classes,
        "img_size": img_size,
        "epoch": epoch,
        "val_metrics": val_metrics,
        "state_dict": model.state_dict(),
    }, path)


def load_checkpoint(path, device):
    """Rebuild the model from a checkpoint; returns (model in eval mode, checkpoint dict)."""
    ckpt = torch.load(path, map_location=device, weights_only=False)
    model = build_model(ckpt["model_name"], len(ckpt["classes"]), pretrained=False)
    model.load_state_dict(ckpt["state_dict"])
    return model.to(device).eval(), ckpt


def train_one_epoch(model, loader, criterion, optimizer):
    device = next(model.parameters()).device
    model.train()
    total_loss = 0.0
    for xb, yb in loader:
        xb, yb = xb.to(device, non_blocking=True), yb.to(device, non_blocking=True)

        optimizer.zero_grad()
        logits = model(xb)
        loss = criterion(logits, yb)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * len(yb)
    return total_loss / len(loader.dataset)


@torch.no_grad()
def evaluate(model, loader, criterion):
    device = next(model.parameters()).device
    model.eval()
    total_loss = 0.0
    all_preds, all_labels = [], []
    for xb, yb in loader:
        xb, yb = xb.to(device, non_blocking=True), yb.to(device, non_blocking=True)
        logits = model(xb)
        total_loss += criterion(logits, yb).item() * len(yb)
        all_preds.append(logits.argmax(dim=1).cpu())
        all_labels.append(yb.cpu())

    preds, labels = torch.cat(all_preds), torch.cat(all_labels)
    return {
        "loss": total_loss / len(loader.dataset),
        "acc": (preds == labels).float().mean().item(),
        "macro_f1": f1_score(labels, preds, average="macro"),
    }
