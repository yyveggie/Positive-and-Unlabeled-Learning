from __future__ import annotations

import argparse
import csv
import importlib.util
import itertools
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))
from pu_lib import metrics_from_scores


DATASETS = ("cifar10", "fashionmnist")
METHODS = ("uasn_e2e", "nnpu_e2e", "holistic_e2e")
POSITIVE_CLASSES = {
    "cifar10": {0, 1, 8, 9},
    "fashionmnist": {0, 2, 4, 6},
}
DEFAULT_LR = {

    "cifar10": 0.0015,
    "fashionmnist": 0.002,
}
FIELDS = [
    "dataset", "seed", "method", "backbone", "device", "n_train",
    "n_labeled_positive", "n_unlabeled", "alpha_unlabeled", "c",
    "learning_rate", "batch_size", "epochs", "warm_epochs",
    "finetune_epochs", "bacc", "f1", "auc", "auprc",
    "f1_bestthr", "thr_best", "precision", "recall", "pred_rate", "ece",
    "brier", "selected_positive_rate", "selected_positive_f1", "secs",
    "status", "error",
]


def _load_function(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(f"survey_{path.stem}", path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return getattr(module, name)


ResNet18 = _load_function(
    ROOT / "external" / "holisticpu" / "model" / "ResNet_Zoo.py", "ResNet18"
)
lenet_fmnist = _load_function(
    ROOT / "external" / "holisticpu" / "model" / "lenet5.py", "lenet_fmnist"
)
cnn_cifar = _load_function(
    ROOT / "external" / "holisticpu" / "model" / "cnn7.py", "cnn_cifar"
)
RandAugmentMC = _load_function(
    ROOT / "external" / "holisticpu" / "dataset" / "randaugment.py",
    "RandAugmentMC",
)
CIFAR_FACTORY = ResNet18
CIFAR_BACKBONE_NAME = "HolisticPU-ResNet18"


class TwoViewSubset(Dataset):
    def __init__(self, base, indices: np.ndarray, weak, strong=None):
        self.base = base
        self.indices = np.asarray(indices, dtype=int)
        self.weak = weak
        self.strong = strong

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, position):
        image, _ = self.base[int(self.indices[position])]
        weak = self.weak(image)
        strong = weak if self.strong is None else self.strong(image)
        return weak, strong, int(position)


class EvalSubset(Dataset):
    def __init__(self, base, indices: np.ndarray, transform, binary_targets):
        self.base = base
        self.indices = np.asarray(indices, dtype=int)
        self.transform = transform
        self.binary_targets = np.asarray(binary_targets, dtype=np.int8)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, position):
        image, _ = self.base[int(self.indices[position])]
        return self.transform(image), int(self.binary_targets[position]), int(position)


def _targets(base) -> np.ndarray:
    y = base.targets
    if torch.is_tensor(y):
        y = y.numpy()
    return np.asarray(y, dtype=int)


def _stratified_cap(labels: np.ndarray, maximum: int, seed: int) -> np.ndarray:
    if maximum <= 0 or maximum >= len(labels):
        return np.arange(len(labels))
    rng = np.random.default_rng(seed)
    classes = np.unique(labels)
    per_class = maximum // len(classes)
    remainder = maximum - per_class * len(classes)
    chosen = []
    for order, cls in enumerate(classes):
        n = per_class + int(order < remainder)
        idx = np.flatnonzero(labels == cls)
        chosen.extend(rng.choice(idx, min(n, len(idx)), replace=False))
    return np.sort(np.asarray(chosen, dtype=int))


def make_data(dataset: str, root: Path, seed: int, c: float, max_train: int):
    if dataset == "cifar10":
        train_base = datasets.CIFAR10(root, train=True, download=True)
        test_base = datasets.CIFAR10(root, train=False, download=True)
        mean, std = (0.4914, 0.4822, 0.4465), (0.2471, 0.2435, 0.2616)
        weak = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(32, padding=4, padding_mode="reflect"),
            transforms.ToTensor(), transforms.Normalize(mean, std),
        ])
        strong = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(32, padding=4, padding_mode="reflect"),
            RandAugmentMC(n=2, m=10),
            transforms.ToTensor(), transforms.Normalize(mean, std),
        ])
        evaluate = transforms.Compose([
            transforms.ToTensor(), transforms.Normalize(mean, std),
        ])
        backbone = CIFAR_BACKBONE_NAME
    elif dataset == "fashionmnist":
        train_base = datasets.FashionMNIST(root, train=True, download=True)
        test_base = datasets.FashionMNIST(root, train=False, download=True)
        mean, std = (0.1307,), (0.3081,)
        weak = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(28, padding=4, padding_mode="reflect"),
            transforms.ToTensor(), transforms.Normalize(mean, std),
        ])
        strong = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(28, padding=4, padding_mode="reflect"),
            transforms.RandomAutocontrast(), transforms.RandomEqualize(),
            transforms.ColorJitter(), transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])
        evaluate = transforms.Compose([
            transforms.ToTensor(), transforms.Normalize(mean, std),
        ])
        backbone = "HolisticPU-LeNet"
    else:
        raise ValueError(dataset)

    train_labels_all = _targets(train_base)
    retained = _stratified_cap(train_labels_all, max_train, seed)
    retained_labels = train_labels_all[retained]
    positive = np.isin(retained_labels, list(POSITIVE_CLASSES[dataset]))
    rng = np.random.default_rng(seed)


    per_class = []
    for cls in sorted(POSITIVE_CLASSES[dataset]):
        candidates = np.flatnonzero(retained_labels == cls)
        n_label = max(1, int(round(c * len(candidates))))
        per_class.append(rng.choice(candidates, n_label, replace=False))
    p_positions = np.sort(np.concatenate(per_class))
    u_positions = np.setdiff1d(np.arange(len(retained)), p_positions)
    p_indices, u_indices = retained[p_positions], retained[u_positions]
    y_u = positive[u_positions].astype(np.int8)

    test_labels = _targets(test_base)
    y_test = np.isin(test_labels, list(POSITIVE_CLASSES[dataset])).astype(np.int8)
    p_data = TwoViewSubset(train_base, p_indices, weak, strong)
    u_data = TwoViewSubset(train_base, u_indices, weak, strong)
    u_eval = EvalSubset(train_base, u_indices, evaluate, y_u)
    test_data = EvalSubset(test_base, np.arange(len(test_base)), evaluate, y_test)
    return p_data, u_data, u_eval, test_data, y_u, backbone


def new_model(dataset: str):
    return CIFAR_FACTORY() if dataset == "cifar10" else lenet_fmnist()


def _positive_logit(output: torch.Tensor) -> torch.Tensor:
    return output[:, 0] - output[:, 1]


def _cycle(loader):
    while True:
        yield from loader


def _loaders(p_data, u_data, u_eval, test_data, batch: int, workers: int,
             seed: int, strong_views: bool):
    generator = torch.Generator().manual_seed(seed)
    common = dict(num_workers=workers, pin_memory=False)
    if not strong_views:
        p_data = TwoViewSubset(
            p_data.base, p_data.indices, p_data.weak, strong=None
        )
        u_data = TwoViewSubset(
            u_data.base, u_data.indices, u_data.weak, strong=None
        )
    p = DataLoader(p_data, batch_size=batch, shuffle=True, drop_last=False,
                   generator=generator, **common)
    u = DataLoader(u_data, batch_size=batch, shuffle=True, drop_last=False,
                   generator=generator, **common)
    ue = DataLoader(u_eval, batch_size=max(batch, 256), shuffle=False, **common)
    te = DataLoader(test_data, batch_size=max(batch, 256), shuffle=False, **common)
    return p, u, ue, te


def _optimizer(model, total_steps: int, lr: float):
    no_decay = ("bias", "bn")
    groups = [
        {
            "params": [p for name, p in model.named_parameters()
                       if not any(token in name for token in no_decay)],
            "weight_decay": 5e-4,
        },
        {
            "params": [p for name, p in model.named_parameters()
                       if any(token in name for token in no_decay)],
            "weight_decay": 0.0,
        },
    ]
    opt = torch.optim.SGD(
        groups, lr=lr, momentum=0.9, nesterov=True,
    )
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=max(total_steps, 1)
    )
    return opt, sched


def train_baseline(dataset: str, method: str, p_loader, u_loader, alpha_u: float,
                   device, epochs: int, lr: float):
    model = new_model(dataset).to(device)
    opt, sched = _optimizer(model, epochs * len(u_loader), lr)
    p_stream = _cycle(p_loader)
    for _ in range(epochs):
        model.train()
        for u_weak, _, _ in u_loader:
            p_weak, _, _ = next(p_stream)
            p_weak, u_weak = p_weak.to(device), u_weak.to(device)
            z_p = _positive_logit(model(p_weak))
            z_u = _positive_logit(model(u_weak))
            if method == "uasn_e2e":
                loss = F.softplus(-z_p).mean() + F.softplus(z_u).mean()
            elif method == "nnpu_e2e":
                pos = torch.sigmoid(-z_p).mean()
                neg = torch.sigmoid(z_u).mean() - alpha_u * torch.sigmoid(z_p).mean()
                loss = alpha_u * pos + torch.clamp(neg, min=0.0)
                if float(neg.detach()) < 0:
                    loss = -neg
            else:
                raise ValueError(method)
            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()
    return model.eval()


def _scores_in_order(model, loader, device) -> tuple[np.ndarray, np.ndarray]:
    scores = np.empty(len(loader.dataset), dtype=np.float64)
    labels = np.empty(len(loader.dataset), dtype=np.int8)
    model.eval()
    with torch.no_grad():
        for image, target, position in loader:
            output = model(image.to(device))
            prob = torch.softmax(output, dim=1)[:, 0].cpu().numpy()
            position = position.numpy()
            scores[position] = prob
            labels[position] = target.numpy()
    return scores, labels


def _jenks_break_2(values: np.ndarray) -> float:
    x = np.sort(np.asarray(values, dtype=np.float64))
    if len(x) < 3:
        return float(x[0])
    c1, c2 = np.cumsum(x), np.cumsum(x * x)
    split = np.arange(1, len(x))
    lower = c2[split - 1] - c1[split - 1] ** 2 / split
    upper = ((c2[-1] - c2[split - 1])
             - (c1[-1] - c1[split - 1]) ** 2 / (len(x) - split))
    return float(x[int(np.argmin(lower + upper))])


def train_holistic(dataset: str, p_loader, u_loader, u_eval_loader, device,
                   warm_epochs: int, finetune_epochs: int, lr: float):

    warm = new_model(dataset).to(device)
    opt, sched = _optimizer(warm, warm_epochs * len(u_loader), lr)
    p_stream = _cycle(p_loader)
    sequence = []
    for _ in range(warm_epochs):
        warm.train()
        for u_weak, u_strong, _ in u_loader:
            p_weak, p_strong, _ = next(p_stream)
            p_weak, p_strong = p_weak.to(device), p_strong.to(device)
            u_weak, u_strong = u_weak.to(device), u_strong.to(device)
            target_p = torch.zeros(len(p_weak), dtype=torch.long, device=device)
            target_u = torch.ones(len(u_weak), dtype=torch.long, device=device)
            loss_p = 0.5 * (
                F.cross_entropy(warm(p_weak), target_p, label_smoothing=0.1)
                + F.cross_entropy(warm(p_strong), target_p, label_smoothing=0.1)
            )
            loss_u = F.cross_entropy(warm(u_weak), target_u)
            loss = loss_p + loss_u
            opt.zero_grad(); loss.backward(); opt.step()
            sched.step()
        sequence.append(_scores_in_order(warm, u_eval_loader, device)[0])

    sequence = np.asarray(sequence)
    delta = np.diff(sequence, axis=0)
    trends = np.log1p(delta + 0.5 * delta ** 2).mean(axis=0)
    breakpoint = _jenks_break_2(trends)
    if breakpoint > 0:
        filtered = trends[trends < 0.2 / 9.0]
        if len(filtered) >= 3:
            breakpoint = _jenks_break_2(filtered)
    pseudo = np.where(trends > breakpoint, 0, 1).astype(np.int64)


    model = new_model(dataset).to(device)
    opt, sched = _optimizer(model, finetune_epochs * len(u_loader), lr)
    p_stream = _cycle(p_loader)
    for epoch in range(finetune_epochs):
        model.train()
        blend = (epoch / max(finetune_epochs, 1)) ** 0.8
        for u_weak, u_strong, u_position in u_loader:
            p_weak, _, _ = next(p_stream)
            p_weak = p_weak.to(device)
            u_weak, u_strong = u_weak.to(device), u_strong.to(device)
            target_p = torch.zeros(len(p_weak), dtype=torch.long, device=device)
            provisional = torch.ones(len(u_weak), dtype=torch.long, device=device)
            selected = torch.as_tensor(
                pseudo[u_position.numpy()], dtype=torch.long, device=device
            )
            soft_target = (
                (1.0 - blend) * F.one_hot(provisional, 2).float()
                + blend * F.one_hot(selected, 2).float()
            )
            logits_weak = model(u_weak)
            logits_strong = model(u_strong)
            loss_p = F.cross_entropy(model(p_weak), target_p)
            loss_u = F.cross_entropy(logits_weak, soft_target)
            with torch.no_grad():
                confidence, self_target = torch.softmax(logits_weak, 1).max(1)
                mask = (confidence >= 0.9).float()
            consistency = (F.cross_entropy(
                logits_strong, self_target, reduction="none"
            ) * mask).mean()
            loss = loss_p + loss_u + consistency
            opt.zero_grad(); loss.backward(); opt.step()
            sched.step()
    return model.eval(), pseudo


def _device(requested: str):
    if requested != "auto":
        return torch.device(requested)
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def _completed(path: Path):
    done = set()
    if path.exists():
        with path.open(newline="") as fh:
            for row in csv.DictReader(fh):
                if row.get("status") == "ok":
                    done.add((row["dataset"], int(row["seed"]), row["method"]))
    return done


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="*", default=list(DATASETS))
    ap.add_argument("--methods", nargs="*", choices=METHODS, default=list(METHODS))
    ap.add_argument("--seeds", nargs="*", type=int, default=[0, 1, 2])
    ap.add_argument("--c", type=float, default=0.2)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--warm-epochs", type=int, default=8)
    ap.add_argument("--finetune-epochs", type=int, default=12)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument(
        "--lr", type=float, default=None,
        help="override the released dataset-specific learning rate",
    )
    ap.add_argument("--workers", type=int, default=0)
    ap.add_argument("--max-train", type=int, default=0)
    ap.add_argument("--device", default="auto")
    ap.add_argument(
        "--cifar-backbone", choices=["cnn7", "resnet18"], default="cnn7",
        help="both factories are imported from the HolisticPU release",
    )
    ap.add_argument(
        "--out", default=str(ROOT / "results" / "end_to_end_vision_track.csv")
    )
    args = ap.parse_args()
    if args.warm_epochs < 2 and "holistic_e2e" in args.methods:
        raise ValueError("HolisticPU trend estimation needs at least two warm epochs")

    global CIFAR_FACTORY, CIFAR_BACKBONE_NAME
    if args.cifar_backbone == "cnn7":
        CIFAR_FACTORY = cnn_cifar
        CIFAR_BACKBONE_NAME = "HolisticPU-CNN7"
    else:
        CIFAR_FACTORY = ResNet18
        CIFAR_BACKBONE_NAME = "HolisticPU-ResNet18"
    device = _device(args.device)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    done = _completed(out)


    new = (not out.exists()) or out.stat().st_size == 0
    with out.open("a", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=FIELDS)
        if new:
            writer.writeheader()
            fh.flush()
        for dataset, seed in itertools.product(args.datasets, args.seeds):
            random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
            learning_rate = args.lr if args.lr is not None else DEFAULT_LR[dataset]
            p_data, u_data, u_eval, test_data, y_u, backbone = make_data(
                dataset, ROOT / "features" / "end_to_end", seed, args.c,
                args.max_train,
            )
            alpha_u = float(y_u.mean())
            for method in args.methods:
                key = (dataset, seed, method)
                if key in done:
                    continue


                torch.manual_seed(seed)
                p_loader, u_loader, u_eval_loader, test_loader = _loaders(
                    p_data, u_data, u_eval, test_data, args.batch_size,
                    args.workers, seed, strong_views=(method == "holistic_e2e"),
                )
                print(f"[vision-e2e] {dataset} seed={seed} {method} on {device}",
                      flush=True)
                start = time.time()
                row = dict(
                    dataset=dataset, seed=seed, method=method,
                    backbone=backbone, device=str(device),
                    n_train=len(p_data) + len(u_data),
                    n_labeled_positive=len(p_data), n_unlabeled=len(u_data),
                    alpha_unlabeled=alpha_u, c=args.c,
                    learning_rate=learning_rate,
                    batch_size=args.batch_size,
                    epochs=(args.warm_epochs + args.finetune_epochs
                            if method == "holistic_e2e" else args.epochs),
                    warm_epochs=args.warm_epochs if method == "holistic_e2e" else 0,
                    finetune_epochs=(args.finetune_epochs
                                     if method == "holistic_e2e" else 0),
                    selected_positive_rate="", selected_positive_f1="",
                    status="ok", error="",
                )
                model = None
                try:
                    if method == "holistic_e2e":
                        model, pseudo = train_holistic(
                            dataset, p_loader, u_loader, u_eval_loader, device,
                            args.warm_epochs, args.finetune_epochs,
                            learning_rate,
                        )
                        selected_positive = (pseudo == 0).astype(int)
                        denom = selected_positive.sum() + y_u.sum()
                        pseudo_f1 = (2 * np.sum(selected_positive * y_u) / denom
                                     if denom else 0.0)
                        row.update(
                            selected_positive_rate=float(selected_positive.mean()),
                            selected_positive_f1=float(pseudo_f1),
                        )
                    else:
                        model = train_baseline(
                            dataset, method, p_loader, u_loader, alpha_u,
                            device, args.epochs, learning_rate,
                        )
                    score, y_test = _scores_in_order(model, test_loader, device)
                    row.update(**metrics_from_scores(
                        score, y_test, threshold=0.5, extended=True
                    ), secs=time.time() - start)
                except Exception as exc:
                    row.update(status="failed", error=f"{type(exc).__name__}: {exc}",
                               secs=time.time() - start)
                    print(f"  FAILED: {row['error']}", flush=True)
                writer.writerow(row); fh.flush()
                if model is not None:
                    del model
                if device.type == "mps":
                    torch.mps.empty_cache()
    print(f"done -> {out}")


if __name__ == "__main__":
    main()
