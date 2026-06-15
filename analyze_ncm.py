import argparse
import json
import os

import torch
import torch.nn.functional as F

from src.data import get_dataset
from src.models.baseline import BaselineMLP
from src.models.cms_mlp import SCMS_MLP


def build_model(name):
    if name == "baseline":
        return BaselineMLP()
    if name == "scms":
        return SCMS_MLP()
    raise ValueError(f"NCM analysis currently supports baseline/scms, got {name}")


def features(model, name, x):
    if name == "baseline":
        out = x
        for layer in list(model.net.children())[:-1]:
            out = layer(out)
        return out
    if name == "scms":
        out = model.fast_memory(x)
        out = model.medium_memory(out)
        out = model.slow_memory(out)
        return out
    raise ValueError(name)


@torch.no_grad()
def build_prototypes(model, model_name, loaders, device, num_classes=10):
    sums = None
    counts = torch.zeros(num_classes, device=device)
    for loader in loaders:
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)
            z = F.normalize(features(model, model_name, x), dim=1)
            if sums is None:
                sums = torch.zeros(num_classes, z.shape[1], device=device)
            sums.index_add_(0, y, z)
            counts.index_add_(0, y, torch.ones_like(y, dtype=torch.float))

    prototypes = sums / counts.clamp_min(1).unsqueeze(1)
    return F.normalize(prototypes, dim=1)


@torch.no_grad()
def eval_ncm(model, model_name, prototypes, loaders, device):
    per_task = []
    total_correct = 0
    total_seen = 0
    for loader in loaders:
        correct = 0
        seen = 0
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)
            z = F.normalize(features(model, model_name, x), dim=1)
            logits = z @ prototypes.t()
            pred = logits.argmax(dim=1)
            correct += pred.eq(y).sum().item()
            seen += y.numel()
        per_task.append(correct / seen)
        total_correct += correct
        total_seen += seen
    return total_correct / total_seen, per_task


def main():
    parser = argparse.ArgumentParser(description="Offline nearest-class-mean readout for trained CL checkpoints")
    parser.add_argument("--dataset", default="split", choices=["split"])
    parser.add_argument("--models", nargs="+", default=["baseline", "scms"])
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 123, 7])
    parser.add_argument("--optimizer", default="SGD")
    parser.add_argument("--f", type=int, default=20)
    parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument("--beta3", type=float, default=0.9)
    parser.add_argument("--stab", type=int, default=1)
    parser.add_argument("--task", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=512)
    args = parser.parse_args()

    device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
    tasks_train, tasks_test, _, _, _ = get_dataset(args.dataset, batch_size=args.batch_size)

    rows = []
    for model_name in args.models:
        for seed in args.seeds:
            prefix = f"{args.dataset}_{model_name}_{args.optimizer}_f{args.f}_a{args.alpha}_b{args.beta3}_s{args.stab}_sd{seed}"
            ckpt_path = os.path.join("data", "results", "checkpoints", prefix, f"task{args.task}.pt")
            if not os.path.exists(ckpt_path):
                print(f"[WARN] Missing checkpoint: {ckpt_path}")
                continue

            model = build_model(model_name).to(device)
            state = torch.load(ckpt_path, map_location=device)
            model.load_state_dict(state)
            model.eval()

            prototypes = build_prototypes(model, model_name, tasks_train, device)
            acc, per_task = eval_ncm(model, model_name, prototypes, tasks_test, device)
            row = {
                "dataset": args.dataset,
                "model": model_name,
                "optimizer": args.optimizer,
                "seed": seed,
                "checkpoint": ckpt_path,
                "ncm_acc": acc,
                "per_task_acc": per_task,
            }
            rows.append(row)
            print(f"{model_name} seed={seed} NCM_ACC={acc:.4f} per_task={[round(x, 4) for x in per_task]}")

    out_path = os.path.join("data", "results", "metrics", f"{args.dataset}_ncm_summary.json")
    with open(out_path, "w") as f:
        json.dump(rows, f, indent=2)
    print(f"[INFO] Wrote {out_path}")


if __name__ == "__main__":
    main()
