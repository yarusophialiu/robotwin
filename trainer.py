# train/trainer.py
import os, math, time
from datetime import datetime
from dataclasses import dataclass
from typing import Optional, Tuple
from torch.utils.tensorboard import SummaryWriter

import torch, sys
import torch.nn as nn
import torch.optim as optim
# from torch.cuda.amp import GradScaler, autocast
from torch.amp import GradScaler, autocast
from torch.optim.lr_scheduler import _LRScheduler
from datasets import LABELS, LABEL_TO_IDX, IDX_TO_LABEL
# from utils.utils import compute_RMSE, compute_RMSEP

class LoggedWriter:
    """
    A simple logger that wraps TensorBoard SummaryWriter
    and optionally prints metrics.
    """
    def __init__(self, log_dir: str = "runs", exp_name: str = None):
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        self.log_dir = os.path.join(log_dir, exp_name or timestamp)
        os.makedirs(self.log_dir, exist_ok=True)
        self.writer = SummaryWriter(self.log_dir)
        print(f"[LoggedWriter] Logging to: {self.log_dir}")

    def add_scalars(self, tag, scalars_dict, step):
        """Add multiple scalar values under one tag."""
        for key, value in scalars_dict.items():
            self.writer.add_scalar(f"{tag}/{key}", value, step)

    def add_scalar(self, tag, value, step):
        self.writer.add_scalar(tag, value, step)

    def flush(self):
        self.writer.flush()

    def close(self):
        self.writer.close()


# ----- Utilities -----
def build_ordinal_targets(target_idx: torch.Tensor, num_classes: int) -> torch.Tensor:
    """CORAL-style targets: [B] -> [B, K-1]"""
    B = target_idx.shape[0]
    device = target_idx.device
    thresholds = torch.arange(num_classes - 1, device=device).unsqueeze(0).expand(B, -1)
    return (target_idx.unsqueeze(1) > thresholds).float()

@torch.no_grad()
def top1_accuracy(logits: torch.Tensor, target_idx: torch.Tensor) -> float:
    return (logits.argmax(dim=1) == target_idx).float().mean().item()

@torch.no_grad()
def macro_accuracy(logits: torch.Tensor, target_idx: torch.Tensor) -> float:
    """
    Macro-averaged accuracy over classes:
    - compute per-class accuracy
    - then average across classes (ignoring classes not present in target)
    """
    num_classes = logits.size(1)
    acc_per_class = []

    for c in range(num_classes):
        mask = (target_idx == c)
        if mask.sum() == 0:
            # no samples of this class in this batch / split
            continue
        acc_c = top1_accuracy(logits[mask], target_idx[mask])
        acc_per_class.append(acc_c)

    if len(acc_per_class) == 0:
        return 0.0

    return float(sum(acc_per_class) / len(acc_per_class))


@dataclass
class EarlyStoppingConfig:
    patience: int = 5          # epochs to wait with no improvement
    min_delta: float = 0.0     # minimum improvement to count
    # mode: str = "max"          # "max" (e.g., accuracy) or "min" (e.g., loss)

class EarlyStopping:
    def __init__(self, cfg: EarlyStoppingConfig):
        self.cfg = cfg
        # self.best = -float("inf") if cfg.mode == "max" else float("inf")
        self.best = float("inf")
        self.num_bad = 0

    def step(self, value: float) -> bool:
        """
        value: val_loss
        Returns True if we should stop early.
        """
        # if max, value is val accuracy, higher the better
        # otherwise value is val_loss, lower the better
        # improved = (
        #     (value > self.best + self.cfg.min_delta) if self.cfg.mode == "max"
        #     else (value < self.best - self.cfg.min_delta)
        # )
        improved = value < self.best - self.cfg.min_delta
        if improved:
            self.best = value
            self.num_bad = 0
        else:
            self.num_bad += 1
        return self.num_bad > self.cfg.patience

# ----- Trainer -----
class Trainer:
    def __init__(
        self,
        model_head: nn.Module,
        spatial_backbone: nn.Module,
        num_classes: int,
        num_epochs: int,
        device: torch.device,
        ce_loss: nn.Module, # cross entropy loss
        bce_loss: Optional[nn.Module], # BCEWithLogitsLoss
        lambda_ord: float,
        out_dir: str,
        optimizer: optim.Optimizer = None,
        scheduler: Optional[_LRScheduler] = None,
        lr: float = 1e-4,
        weight_decay: float = 1e-2,
        use_amp: bool = True,
        save_every: int = 1,
        eval: bool = False
    ):
        self.head = model_head.to(device)
        self.spatial = spatial_backbone.to(device).eval()  # spatial is frozen
        for p in self.spatial.parameters():
            p.requires_grad = False

        self.num_classes = num_classes
        self.device = device
        self.ce_loss = ce_loss
        self.bce_loss = bce_loss
        self.lambda_ord = lambda_ord
        
        # optimizer_local = optim.AdamW([p for p in model_head.parameters() if p.requires_grad], lr=lr, weight_decay=weight_decay)
        # scheduler_local = CosineAnnealingLR(optimizer, T_max=max(num_epochs, 1))

        self.optimizer = optimizer 
        self.scheduler = scheduler 
        self.out_dir = out_dir
        self.use_amp = use_amp
        # self.scaler = GradScaler(enabled=use_amp)
        # self.scaler = GradScaler(enabled=use_amp)
        self.save_every = save_every
        self.global_step = 0
        self.lr = lr

        date_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        exp_name = f"exp_temporal_test_lr{self.lr}_" + date_str
        self.out_dir = os.path.join(self.out_dir, exp_name)
        if not eval:
            self.writer = LoggedWriter(log_dir=self.out_dir)
            os.makedirs(self.out_dir, exist_ok=True)
            print(f"🔹 Saving outputs to: {self.out_dir}")


    @staticmethod
    def _get_lr(optimizer):
        for g in optimizer.param_groups:
            if "lr" in g:
                return float(g["lr"])
        return None


    def _forward_batch(self, batch, train: bool = True, SHOW_WEIGHTED_LOSS: bool = False, TEST_EVAL: bool = False):
        rgb = batch["rgb"].to(self.device, non_blocking=True)        # [B,T,C,H,W]
        motion = batch["motion"].to(self.device, non_blocking=True)  # [B,Cm,T,H,W]
        target = batch["label"].to(self.device, non_blocking=True)   # [B]

        B, T, C, H, W = rgb.shape
        frames = rgb.view(B * T, C, H, W)

        with torch.no_grad():
        # per-frame spatial features, then reshape to [B,T,Ds]
            s_feats = self.spatial(frames).view(B, T, -1)

        # with autocast(enabled=self.use_amp, device_type='cuda'):
        # logits is raw unnormalized scores float
        logits, ord_logits, _ = self.head(motion, s_feats, return_att=True)
        loss = self.ce_loss(logits, target)
        ord_loss = None
        if (self.bce_loss is not None) and (ord_logits is not None):
            ord_tgts = build_ordinal_targets(target, self.num_classes)
            ord_loss = self.bce_loss(ord_logits, ord_tgts).mean()
            loss = loss + self.lambda_ord * ord_loss
            lambda_ord_loss = self.lambda_ord * ord_loss
            # print(f'ce_loss {self.ce_loss(logits, target)}, loss {loss}\n')

        acc = top1_accuracy(logits.detach(), target)
        macro_acc = macro_accuracy(logits.detach(), target)

        logs = {
            "loss": float(loss.detach().item()),
            "ce_loss": float(self.ce_loss(logits, target).detach().item()),
            "ord_loss": float(ord_loss.detach().item()) if ord_loss is not None else 0,
            "lambda_ord_loss": float(lambda_ord_loss.detach().item()) if ord_loss is not None else 0,
            "acc": acc,
            "macro_acc": macro_acc,
        }

        if SHOW_WEIGHTED_LOSS:
            with torch.no_grad():
                logits, ord_logits, _ = self.head(motion, s_feats, return_att=True)
                loss_weighted = self.ce_loss(logits, target)

                # compare with unweighted loss
                ce_unweighted = torch.nn.CrossEntropyLoss().to(self.device)
                loss_unweighted = ce_unweighted(logits, target)
                print(f'ce_unweighted {ce_unweighted.weight}')
                print(f'self.ce_loss.weight {self.ce_loss.weight}')

                print(f"Weighted val loss: {loss_weighted.item():.4f} | Unweighted val loss: {loss_unweighted.item():.4f}")
         
        if TEST_EVAL:
            preds = logits.detach().argmax(dim=1)   # [B]
            # return preds and GT to collect for confusion matrix / saving
            return loss, acc, logs, preds, target
        return loss, acc, logs, None, None



    def train_one_epoch(self, loader) -> Tuple[float, float]:
        self.head.train()
        total_loss, total_acc, n = 0.0, 0.0, 0
        for batch in loader:
            self.optimizer.zero_grad(set_to_none=True)
            loss, acc, logs, _, _ = self._forward_batch(batch, train=True)
            logs = {k: (float(v) if v is not None else None) for k, v in logs.items()}

            # self.scaler.scale(loss).backward()
            # self.scaler.step(self.optimizer)
            # self.scaler.update()
            loss.backward()
            self.optimizer.step()

            total_loss += float(loss.item())
            total_acc += float(acc)
            n += 1
            if n % 1000 == 0:
                print(f"[Epoch step {self.global_step}] Batch {n}/{len(loader)}", flush=True)
            
            self.global_step += 1
            # ---- logging (per-batch) ----
            lr = self._get_lr(self.optimizer)
            # amp_scale = float(self.scaler.get_scale()) if hasattr(self.scaler, "get_scale") else None

            # Core scalars
            self.writer.add_scalars("train/step", {
                "loss_total": logs.get("loss", float(loss.detach().item())),
                "acc": float(acc),
                "macro_acc": logs.get("macro_acc"),
                "ce_loss": logs.get("ce_loss"),
                "ord_loss": logs.get("ord_loss"),
                "lambda_ord_loss": logs.get("lambda_ord_loss"),
            }, self.global_step)

            if self.global_step % 50 == 0:
                self.writer.flush()

        # ---- epoch averages ----
        avg_loss = total_loss / max(n, 1)
        avg_acc  = total_acc  / max(n, 1)
        # print(f'total_loss {total_loss}, n {n}, avg_loss {avg_loss}')

        self.writer.add_scalars("train/epoch", {
            "loss_avg": avg_loss,
            "acc_avg": avg_acc,
            "macro_acc": logs.get("macro_acc"),
        }, self.global_step)
        self.writer.flush()
        return avg_loss, avg_acc
        # return total_loss / max(n, 1), total_acc / max(n, 1)

    @torch.no_grad()
    def evaluate(self, loader) -> Tuple[float, float]:
        self.head.eval()
        total_loss, total_acc, n = 0.0, 0.0, 0
        for batch in loader:
            loss, acc, logs, _, _ = self._forward_batch(batch, train=False, SHOW_WEIGHTED_LOSS=False)
            total_loss += float(loss.item())
            total_acc += float(acc)
            n += 1
        return total_loss / max(n, 1), total_acc / max(n, 1), logs["macro_acc"] / max(n, 1)

    
    @torch.no_grad()
    def evaluate_test(self, loader) -> Tuple[float, float, float, torch.Tensor, torch.Tensor]:
        """
        Test-time eval: returns metrics + all predictions and targets
        for confusion matrix / saving.
        """
        self.head.eval()
        total_loss, total_acc, total_macro, n = 0.0, 0.0, 0.0, 0

        all_preds = []
        all_targets = []

        jod_preds = []   # JOD at predicted resolution
        jod_tgts  = []   # JOD at target resolution
        jod_map = loader.dataset.jod_map  # (scene_name, start, res) -> jod

        for batch in loader:
            loss, acc, logs, preds, targets = self._forward_batch(
                batch,
                train=False,
                SHOW_WEIGHTED_LOSS=False,
                TEST_EVAL=True,
            )
            total_loss += float(loss.item())
            total_acc += float(acc)
            total_macro += float(logs["macro_acc"])
            n += 1

            preds_indices = preds.cpu().tolist() # Converts tensor to a Python list: [0, 1, 0]
            preds_labels = [IDX_TO_LABEL[i] for i in preds_indices]
            targets_indices = targets.cpu().tolist() # Converts tensor to a Python list: [0, 1, 0]
            targets_labels = [IDX_TO_LABEL[i] for i in targets_indices]

            all_preds.append(torch.tensor(preds_labels))
            all_targets.append(torch.tensor(targets_labels))

            meta = batch["meta"]
            # meta is a dict of lists / tensors with batch dimension
            scene_names = meta["scene_name"]             # list of str, len=B
            scene_dists = meta["scene_dist"]             # list[str], len=B
            starts      = meta["start"]                  # tensor [B]
            label_vals  = meta["label_value"]            # tensor [B], resolution values
            preds_cpu = preds.cpu()
            B = preds_cpu.shape[0]
            # print(f'jod_map {jod_map}')
            # print(f'IDX_TO_LABEL {IDX_TO_LABEL}')
            for i in range(B):
                scene_dist = scene_dists[i]
                start_i = int(starts[i].item())
                tgt_res = int(label_vals[i].item())          # target resolution (value)
                pred_idx = int(preds_cpu[i].item())
                pred_res = IDX_TO_LABEL[pred_idx]            # predicted resolution (value)

                key_tgt  = (scene_dist, start_i, tgt_res)
                key_pred = (scene_dist, start_i, pred_res)
                # print(f'key_tgt {key_tgt}')
                # print(f'key_pred {key_pred}')

                if key_tgt not in jod_map or key_pred not in jod_map:
                    continue

                jod_tgts.append(jod_map[key_tgt])
                jod_preds.append(jod_map[key_pred])
                # print(f'jod_map[key_pred] {jod_map[key_pred]}')
                # print(f'jod_tgts {jod_tgts}')
        # print(f'jod_preds {jod_preds}')
        denom = max(n, 1)
        test_loss = total_loss / denom
        test_acc = total_acc / denom
        macro_acc = total_macro / denom

        if all_preds:
            all_preds = torch.cat(all_preds, dim=0)
            all_targets = torch.cat(all_targets, dim=0)
        else:
            all_preds = torch.empty(0, dtype=torch.long)
            all_targets = torch.empty(0, dtype=torch.long)

        return test_loss, test_acc, macro_acc, all_preds, all_targets, jod_preds, jod_tgts

    def save_checkpoint(self, epoch: int, name: str, extra: dict = None):
        ckpt = {
            "epoch": epoch,
            "head_state": self.head.state_dict(),
            "spatial_state": self.spatial.state_dict(),  # frozen, but keep for reproducibility
            "optimizer": self.optimizer.state_dict(),
            # "scaler": self.scaler.state_dict(),
        }
        if self.scheduler is not None:
            ckpt["scheduler"] = self.scheduler.state_dict()
        if extra:
            ckpt.update(extra)
        torch.save(ckpt, os.path.join(self.out_dir, f"{name}.pth"))

    def fit(
        self,
        train_loader,
        val_loader=None,
        epochs: int = 20,
        early_cfg: Optional[EarlyStoppingConfig] = None,
        # monitor: str = "val_acc",  # "val_acc" | "val_loss"
    ):
        early = EarlyStopping(early_cfg) if early_cfg is not None else None
        best_metric = -float("inf")
        best_epoch = 0
        start_time = time.time()

        for epoch in range(1, epochs + 1):
            epoch_start = time.time()
            print(f'=========================== Epoch {epoch} ===========================')
            train_loss, train_acc = self.train_one_epoch(train_loader)
            if self.scheduler is not None:
                self.scheduler.step()

            if val_loader is not None:
                val_loss, val_acc, macro_acc = self.evaluate(val_loader)
                self.writer.add_scalars("val/epoch", {
                    "loss": val_loss,
                    "acc": val_acc,
                    "macro_acc": macro_acc,
                }, self.global_step)
                self.writer.flush()
            else:
                val_loss, val_acc, macro_acc = float("nan"), float("nan"), float("nan")


            epoch_time = time.time() - epoch_start
            # print(f"Epoch {epoch} took {epoch_time/60:.2f} min ({epoch_time:.1f} sec)")

            msg = (f"Epoch {epoch:03d}/{epochs} | "
                   f"train loss {train_loss:.4f} acc {train_acc:.4f} | "
                   f"val loss {val_loss:.4f} acc {val_acc:.4f} | "
                   f"macro_acc {macro_acc:.4f} | "
                   f"took {epoch_time/60:.2f} min ({epoch_time:.1f} sec)")
            print(msg)



            if (epoch % self.save_every) == 0:
                self.save_checkpoint(epoch, f"epoch_{epoch:03d}")

            # pick metric
            # metric = val_acc if monitor == "val_acc" else (-val_loss)
            # best.pth has highest val accuracy
            metric = val_acc
            if not math.isnan(metric) and metric > best_metric:
                best_metric = metric
                best_epoch = epoch
                self.save_checkpoint(epoch, "best", extra={"best_metric": best_metric})
                
            # early stopping has lowest val_loss
            if early is not None and val_loader is not None:
                # stop = early.step(val_acc if early.cfg.mode == "max" else val_loss)
                stop = early.step(val_loss)
                if stop:
                    print(f"[EarlyStopping] Stop at epoch {epoch} (best epoch {best_epoch}, metric {best_metric:.4f})")
                    break

        
        total_time = time.time() - start_time
        print(f"✅ Total training time: {total_time/3600:.2f} hours ({total_time/60:.2f} min)")
        # always save final
        self.save_checkpoint(epoch, "last", extra={"best_epoch": best_epoch, "best_metric": best_metric})
        print("Training finished.")
