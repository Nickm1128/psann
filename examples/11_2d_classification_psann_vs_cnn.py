import math

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from psann.conv import PSANNConv2dNet


def make_synthetic_images(N=2000, H=16, W=16, seed=0):
    rs = np.random.RandomState(seed)
    X = rs.randn(N, 1, H, W).astype(np.float32)
    # Label rule: class 1 if weighted center mass > threshold
    yy, xx = np.meshgrid(np.linspace(-1, 1, H), np.linspace(-1, 1, W), indexing="ij")
    weight = (xx**2 + yy**2).astype(np.float32)
    cm = (X[:, 0] * weight).mean(axis=(1, 2))
    y = (cm > 0.05).astype(np.int64)
    return X, y


class SimpleCNN(nn.Module):
    def __init__(self, num_classes=2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, 3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
        )
        self.fc = nn.Linear(32, num_classes)

    def forward(self, x):
        z = self.net(x).flatten(1)
        return self.fc(z)


def train_eval_cls(model, train, val, test, epochs=40, bs=128, lr=1e-3, device=None):
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()
    dl_tr = DataLoader(
        TensorDataset(*[torch.from_numpy(a) for a in train]), batch_size=bs, shuffle=True
    )
    dl_va = DataLoader(TensorDataset(*[torch.from_numpy(a) for a in val]), batch_size=bs)
    Xte, yte = [torch.from_numpy(a) for a in test]
    best = math.inf
    best_state = None
    patience = 10
    for epoch in range(epochs):
        model.train()
        tot = cnt = 0
        for xb, yb in dl_tr:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            logits = model(xb)
            loss = loss_fn(logits, yb)
            loss.backward()
            opt.step()
            tot += float(loss.item()) * xb.size(0)
            cnt += xb.size(0)
        tr = tot / max(cnt, 1)
        # val
        model.eval()
        with torch.no_grad():
            tot = cnt = 0
            for xb, yb in dl_va:
                xb, yb = xb.to(device), yb.to(device)
                logits = model(xb)
                loss = loss_fn(logits, yb)
                tot += float(loss.item()) * xb.size(0)
                cnt += xb.size(0)
            va = tot / max(cnt, 1)
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1:3d} - train {tr:.4f} - val {va:.4f}")
        if va + 1e-8 < best:
            best = va
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            patience = 10
        else:
            patience -= 1
            if patience <= 0:
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    # test
    model.eval()
    with torch.no_grad():
        logits = model(Xte.to(device))
        pred = logits.argmax(dim=1).cpu().numpy()
        acc = float((pred == yte.numpy()).mean())
    return acc


if __name__ == "__main__":
    X, y = make_synthetic_images(N=3000, H=16, W=16, seed=2)
    n = len(X)
    n_tr = int(0.7 * n)
    n_va = int(0.15 * n)
    train = (X[:n_tr], y[:n_tr])
    val = (X[n_tr : n_tr + n_va], y[n_tr : n_tr + n_va])
    test = (X[n_tr + n_va :], y[n_tr + n_va :])

    # PSANN Conv classifier: set out_dim=num_classes
    class PSANNCNN(nn.Module):
        def __init__(self, num_classes=2):
            super().__init__()
            self.net = PSANNConv2dNet(
                1, num_classes, hidden_layers=2, hidden_channels=32, kernel_size=3
            )

        def forward(self, x):
            return self.net(x)

    print("Training PSANN Conv classifier...")
    acc_psann = train_eval_cls(PSANNCNN(), train, val, test)
    print("Training Simple CNN classifier...")
    acc_cnn = train_eval_cls(SimpleCNN(), train, val, test)
    print(f"PSANN Acc: {acc_psann*100:.2f}% | CNN Acc: {acc_cnn*100:.2f}%")
