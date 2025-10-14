import math
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from psann.conv import PSANNConv2dNet


def make_seq_images(N=1200, T=10, H=8, W=8, seed=0):
    rs = np.random.RandomState(seed)
    # Generate sequences where amplitude drifts over time
    X = rs.randn(N, T, 1, H, W).astype(np.float32)
    t = np.linspace(0, 1, T, dtype=np.float32)
    amp = (1.0 + 0.5 * np.sin(2 * np.pi * (t * 0.8)))  # (T,)
    X *= amp[None, :, None, None, None]
    # Target: predict mean of next frame (scalar)
    y = X.mean(axis=(2, 3, 4))[:, -1]  # (N,)
    return X, y.astype(np.float32)[:, None]


class PSANNWithAttention(nn.Module):
    def __init__(self, in_channels=1, embed=32, depth=2, heads=2):
        super().__init__()
        # per-frame encoder using PSANN
        self.enc = PSANNConv2dNet(in_channels, embed, hidden_layers=2, hidden_channels=32, kernel_size=3)
        enc_layer = nn.TransformerEncoderLayer(d_model=embed, nhead=heads, dim_feedforward=embed * 2, batch_first=True)
        self.tx = nn.TransformerEncoder(enc_layer, num_layers=depth)
        self.head = nn.Linear(embed, 1)

    def forward(self, x):
        # x: (N, T, C, H, W)
        N, T, C, H, W = x.shape
        x = x.reshape(N * T, C, H, W)
        z = self.enc(x)  # (N*T, embed)
        z = z.reshape(N, T, -1)
        z = self.tx(z)  # (N, T, embed)
        return self.head(z[:, -1, :])


class LSTMSeq(nn.Module):
    def __init__(self, in_channels=1, hidden=64):
        super().__init__()
        self.enc = nn.Sequential(
            nn.Conv2d(in_channels, 16, 3, padding=1), nn.ReLU(), nn.AdaptiveAvgPool2d(1)
        )
        self.lstm = nn.LSTM(16, hidden, batch_first=True)
        self.fc = nn.Linear(hidden, 1)

    def forward(self, x):
        # x: (N, T, C, H, W)
        N, T, C, H, W = x.shape
        x = x.reshape(N * T, C, H, W)
        z = self.enc(x).flatten(1)  # (N*T, 16)
        z = z.reshape(N, T, -1)
        out, _ = self.lstm(z)
        return self.fc(out[:, -1, :])


def train_eval(model, train, val, test, epochs=60, bs=128, lr=1e-3, device=None):
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()
    dl_tr = DataLoader(TensorDataset(*[torch.from_numpy(a) for a in train]), batch_size=bs, shuffle=True)
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
            pred = model(xb)
            loss = loss_fn(pred, yb)
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
                pred = model(xb)
                loss = loss_fn(pred, yb)
                tot += float(loss.item()) * xb.size(0)
                cnt += xb.size(0)
            va = tot / max(cnt, 1)
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1:3d} - train {tr:.6f} - val {va:.6f}")
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
    model.eval()
    with torch.no_grad():
        pred = model(Xte.to(device)).cpu().numpy()
    mse = float(np.mean((pred - yte.numpy()) ** 2))
    return mse


if __name__ == "__main__":
    X, y = make_seq_images(N=1600, T=12, H=8, W=8, seed=1)
    n = len(X)
    n_tr = int(0.7 * n)
    n_va = int(0.15 * n)
    train = (X[:n_tr], y[:n_tr])
    val = (X[n_tr : n_tr + n_va], y[n_tr : n_tr + n_va])
    test = (X[n_tr + n_va :], y[n_tr + n_va :])

    print("Training PSANN+Attention...")
    mse_psann_attn = train_eval(PSANNWithAttention(), train, val, test)
    print("Training LSTM baseline...")
    mse_lstm = train_eval(LSTMSeq(), train, val, test)
    print(f"PSANN+Attn MSE: {mse_psann_attn:.6f} | LSTM MSE: {mse_lstm:.6f}")
