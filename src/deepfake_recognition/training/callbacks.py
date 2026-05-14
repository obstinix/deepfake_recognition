"""Early stopping and model checkpointing."""
from __future__ import annotations
import shutil
from pathlib import Path
import torch


class EarlyStopping:
    def __init__(self, patience: int = 8, monitor: str = "auc", mode: str = "max"):
        self.patience = patience
        self.monitor = monitor
        self.mode = mode
        self.best = float("-inf") if mode == "max" else float("inf")
        self.counter = 0

    def step(self, metrics: dict) -> bool:
        val = metrics.get(self.monitor, 0)
        improved = val > self.best if self.mode == "max" else val < self.best
        if improved:
            self.best = val
            self.counter = 0
            return False   # do not stop
        self.counter += 1
        return self.counter >= self.patience


class ModelCheckpoint:
    def __init__(self, checkpoint_dir: str, save_top_k: int = 3,
                 monitor: str = "auc", mode: str = "max"):
        self.dir = Path(checkpoint_dir)
        self.dir.mkdir(parents=True, exist_ok=True)
        self.save_top_k = save_top_k
        self.monitor = monitor
        self.mode = mode
        self.saved: list[tuple[float, Path]] = []

    def save(self, model, optimizer, epoch: int, metrics: dict):
        val = metrics.get(self.monitor, 0)
        fname = self.dir / f"epoch_{epoch:03d}_{self.monitor}{val:.4f}.pth"
        torch.save({
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "metrics": metrics,
        }, fname)

        self.saved.append((val, fname))
        self.saved.sort(key=lambda x: x[0], reverse=(self.mode == "max"))

        # Symlink best
        best_link = self.dir / "best.pth"
        if best_link.exists() or best_link.is_symlink():
            best_link.unlink()
        best_link.symlink_to(self.saved[0][1].name)

        # Remove worst if over limit
        while len(self.saved) > self.save_top_k:
            _, old = self.saved.pop()
            if old.exists():
                old.unlink()

        return fname
