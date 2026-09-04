from __future__ import annotations

import math
import os
import sys

import numpy as np
import torch
import torch.nn as nn

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(HERE, "external", "vpu"))
sys.path.insert(0, os.path.join(HERE, "external", "distpu"))

from model.model_vec import NetworkPhi
from losses.distributionLoss import LabelDistributionLoss
from losses.entropyMinimization import loss_entropy
from customized.mixup import mixup_two_targets, mixup_bce

from pu_lib import MLP, PUData


class _Predictor:

    def __init__(self, fn):
        self.fn = fn

    def proba(self, X):
        with torch.no_grad():
            return self.fn(torch.tensor(X, dtype=torch.float32)).numpy()


def train_vpu_official(pu_fit: PUData, pu_va: PUData, seed: int,
                       epochs=50, val_iterations=30, batch=500,
                       lr=3e-5, mix_alpha=0.3, lam=0.03) -> _Predictor:
    torch.manual_seed(seed)
    Xp = torch.tensor(pu_fit.X_p, dtype=torch.float32)
    Xu = torch.tensor(pu_fit.X_u, dtype=torch.float32)
    Xp_va = torch.tensor(pu_va.X_p, dtype=torch.float32)
    Xu_va = torch.tensor(pu_va.X_u, dtype=torch.float32)

    model = NetworkPhi(input_size=Xp.shape[1])
    opt = torch.optim.Adam(model.parameters(), lr=lr, betas=(0.5, 0.99))
    m = torch.distributions.beta.Beta(mix_alpha, mix_alpha)

    lowest_val_var, best_state = math.inf, None
    for epoch in range(epochs):
        if epoch % 20 == 19:
            lr /= 2
            opt = torch.optim.Adam(model.parameters(), lr=lr, betas=(0.5, 0.99))
        model.train()
        for _ in range(val_iterations):
            ix = torch.randint(0, len(Xu), (min(batch, len(Xu)),))


            ip = torch.randint(0, len(Xp), (len(ix),))
            data_p, data_x = Xp[ip], Xu[ix]
            out_all = model(torch.cat((data_p, data_x)))
            log_phi_p = out_all[:len(data_p), 1]
            log_phi_x = out_all[len(data_p):, 1]
            var_loss = (torch.logsumexp(log_phi_x, dim=0)
                        - math.log(len(log_phi_x)) - torch.mean(log_phi_p))

            target_x = log_phi_x.exp()
            target_p = torch.ones(len(data_p))
            perm = torch.randperm(data_p.size(0))
            lam_mix = m.sample()
            data = lam_mix * data_x + (1 - lam_mix) * data_p[perm]
            target = lam_mix * target_x + (1 - lam_mix) * target_p[perm]
            reg = ((torch.log(target) - model(data)[:, 1]) ** 2).mean()
            loss = var_loss + lam * reg
            opt.zero_grad(); loss.backward(); opt.step()

        model.eval()
        with torch.no_grad():
            lp = model(Xp_va)[:, 1]
            lx = model(Xu_va)[:, 1]
            val_var = (torch.logsumexp(lx, dim=0) - math.log(len(lx))
                       - torch.mean(lp)).item()
        if val_var < lowest_val_var:
            lowest_val_var = val_var
            best_state = {k: v.clone() for k, v in model.state_dict().items()}

    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        log_max_phi = model(Xu)[:, 1].max()

    def proba(X):
        return (model(X)[:, 1] - log_max_phi).exp().clamp(0.0, 1.0)

    return _Predictor(proba)


def train_distpu_official(pu_fit: PUData, alpha: float, seed: int,
                          warm_epochs=60, pu_epochs=60, batch=256,
                          co_mu=2e-3, co_entropy_max=0.004,
                          co_mix_entropy=0.04, co_mixup=5.0,
                          mix_beta=6.0, hidden=64, warm_lr=5e-4) -> _Predictor:
    torch.manual_seed(seed)
    np.random.seed(seed)
    device = "cpu"
    X = torch.tensor(np.vstack([pu_fit.X_p, pu_fit.X_u]), dtype=torch.float32)
    Y = torch.cat([torch.ones(len(pu_fit.X_p)), torch.zeros(len(pu_fit.X_u))])
    n = len(Y)

    model = MLP(X.shape[1], hidden=hidden)
    base_loss = LabelDistributionLoss(prior=alpha, device=device)

    def loss_warm(outputs, labels):
        scores = torch.sigmoid(torch.clamp(outputs, min=-10, max=10))
        return base_loss(outputs, labels) + co_mu * loss_entropy(scores[labels != 1])


    opt = torch.optim.Adam(model.parameters(), lr=warm_lr, weight_decay=5e-3)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, pu_epochs)
    for _ in range(warm_epochs):
        model.train()
        for idx in torch.randperm(n).split(batch):
            out = model(X[idx]).reshape(-1)
            loss = loss_warm(out, Y[idx])
            opt.zero_grad(); loss.backward(); opt.step()
        sched.step()


    model.eval()
    with torch.no_grad():
        psudo = torch.sigmoid(model(X).reshape(-1)).detach()


    opt = torch.optim.Adam(model.parameters(), lr=5e-5, weight_decay=1e-3)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, pu_epochs, 0.7 * 5e-5)
    for epoch in range(pu_epochs):
        co_ent = (1 - math.cos((float(epoch) / pu_epochs) * (math.pi / 2))) \
            * co_entropy_max
        model.train()
        for idx in torch.randperm(n).split(batch):
            Xs, Ys = X[idx], Y[idx]
            ps = psudo[idx].clone()
            ps[Ys == 1] = 1
            mixed_x, y_a, y_b, lam_mix = mixup_two_targets(Xs, ps, mix_beta, device)
            scores_mix = torch.sigmoid(
                torch.clamp(model(mixed_x).reshape(-1), min=-10, max=10))
            out_clean = torch.clamp(model(Xs).reshape(-1), min=-10, max=10)
            scores_clean = torch.sigmoid(out_clean)
            loss = (base_loss(out_clean, Ys.float())
                    + co_ent * loss_entropy(scores_clean[Ys != 1])
                    + co_mix_entropy * loss_entropy(scores_mix)
                    + co_mixup * mixup_bce(scores_mix, y_a, y_b, lam_mix))
            opt.zero_grad(); loss.backward(); opt.step()
            with torch.no_grad():
                psudo[idx] = scores_clean.detach()
        sched.step()

    model.eval()

    def proba(Xq):
        return torch.sigmoid(torch.clamp(model(Xq).reshape(-1), min=-10, max=10))

    return _Predictor(proba)


class _HolisticMLP(nn.Module):

    def __init__(self, d_in):
        super().__init__()
        self.fc = nn.ModuleList([nn.Linear(d_in, 16), nn.Linear(16, 18),
                                 nn.Linear(18, 20), nn.Linear(20, 24),
                                 nn.Linear(24, 2)])

    def forward(self, x):
        x = torch.relu(self.fc[0](x))
        x = torch.relu(self.fc[1](x))
        x = nn.functional.dropout(x, p=0.25, training=self.training)
        x = torch.relu(self.fc[2](x))
        x = torch.relu(self.fc[3](x))
        return self.fc[4](x)


def _jenks_break_2(v):
    x = np.sort(np.asarray(v, dtype=np.float64))
    n = len(x)
    if n < 3:
        return float(x[0])
    c1, c2 = np.cumsum(x), np.cumsum(x * x)
    i = np.arange(1, n)
    lo = c2[i - 1] - c1[i - 1] ** 2 / i
    hi = (c2[-1] - c2[i - 1]) - (c1[-1] - c1[i - 1]) ** 2 / (n - i)
    return float(x[np.argmin(lo + hi)])


def _holistic_phase(Xp, Xu, targets_u, epochs, lr, batch, seed, record=False):
    torch.manual_seed(seed)
    model = _HolisticMLP(Xp.shape[1])
    opt = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, nesterov=True)
    n_u = len(Xu)
    seq = []
    for _ in range(epochs):
        model.train()
        for start in range(0, n_u, batch):
            iu = torch.arange(start, min(start + batch, n_u))
            ip = torch.randint(0, len(Xp), (len(iu),))
            zp, zu = model(Xp[ip]), model(Xu[iu])
            loss = (nn.functional.cross_entropy(zp, torch.zeros(len(ip), dtype=torch.long))
                    + nn.functional.cross_entropy(zu, targets_u[iu]))
            opt.zero_grad(); loss.backward(); opt.step()
        if record:
            model.eval()
            with torch.no_grad():
                seq.append(torch.softmax(model(Xu), 1)[:, 0].numpy())
    return model, np.asarray(seq)


def train_holistic_official(pu_fit: PUData, seed: int, warm_epochs=10,
                            ft_epochs=30, lr=1e-3, batch=50) -> _Predictor:
    Xp = torch.tensor(pu_fit.X_p, dtype=torch.float32)
    Xu = torch.tensor(pu_fit.X_u, dtype=torch.float32)
    all_neg = torch.ones(len(Xu), dtype=torch.long)

    _, seq = _holistic_phase(Xp, Xu, all_neg, warm_epochs, lr, batch, seed,
                             record=True)
    d = np.diff(seq, axis=0)
    trends = np.log1p(d + 0.5 * d ** 2).mean(axis=0)
    pseudo = torch.tensor(np.where(trends > _jenks_break_2(trends), 0, 1),
                          dtype=torch.long)

    model, _ = _holistic_phase(Xp, Xu, pseudo, ft_epochs, lr, batch, seed + 1)
    model.eval()

    def proba(Xq):
        return torch.softmax(model(Xq), 1)[:, 0]

    return _Predictor(proba)


OFFICIAL_GRID = {
    "vpu_off": [{"lr": 3e-5, "lam": 0.03},
                {"lr": 3e-5, "lam": 0.10},
                {"lr": 1e-4, "lam": 0.03},
                {"lr": 1e-4, "lam": 0.10}],
    "distpu_off": [{"warm_lr": 5e-4, "co_mixup": 5.0},
                   {"warm_lr": 5e-4, "co_mixup": 2.0},
                   {"warm_lr": 1e-3, "co_mixup": 5.0},
                   {"warm_lr": 1e-3, "co_mixup": 2.0}],
    "holistic_off": [{"lr": 1e-3, "warm_epochs": 10},
                     {"lr": 1e-3, "warm_epochs": 20},
                     {"lr": 2e-3, "warm_epochs": 10},
                     {"lr": 2e-3, "warm_epochs": 20}],
}


def train_official(method: str, pu_fit: PUData, pu_va: PUData, alpha: float,
                   seed: int, cfg=None, **kw) -> _Predictor:
    cfg = dict(cfg or {})
    if method == "vpu_off":
        return train_vpu_official(pu_fit, pu_va, seed, **cfg,
                                  **{k: v for k, v in kw.items()
                                     if k in ("epochs", "val_iterations", "batch")})
    if method == "distpu_off":
        return train_distpu_official(pu_fit, alpha, seed, **cfg,
                                     **{k: v for k, v in kw.items()
                                        if k in ("warm_epochs", "pu_epochs",
                                                 "hidden", "batch")})
    if method == "holistic_off":
        return train_holistic_official(pu_fit, seed, **cfg,
                                       **{k: v for k, v in kw.items()
                                          if k in ("ft_epochs", "batch")})
    raise ValueError(method)
