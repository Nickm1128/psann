import math
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from psann import PSANNRegressor


def make_amp_drift_series(T=6000, f=0.02, drift=0.0006, noise=0.05, seed=0):
    rs = np.random.RandomState(seed)
    t = np.arange(T, dtype=np.float32)
    amp = 1.0 + 0.5 * np.sin(2 * np.pi * drift * t)
    x = amp * np.sin(2 * np.pi * f * t) + noise * rs.randn(T).astype(np.float32)
    return x.astype(np.float32)


def r2_score_np(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = y_true.reshape(-1)
    y_pred = y_pred.reshape(-1)
    u = np.sum((y_true - y_pred) ** 2)
    v = np.sum((y_true - y_true.mean()) ** 2)
    return float(1.0 - (u / v if v != 0 else np.nan))


class LSTM1(nn.Module):
    def __init__(self, hidden=64):
        super().__init__()
        self.lstm = nn.LSTM(1, hidden, batch_first=True)
        self.fc = nn.Linear(hidden, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = out[:, -1, :]
        return self.fc(out)


if __name__ == "__main__":
    series = make_amp_drift_series(T=7000, f=0.02, drift=0.001, noise=0.05, seed=42)
    n = len(series)
    n_train = int(0.7 * n)
    n_val = int(0.15 * n)
    train = series[:n_train]
    val = series[n_train : n_train + n_val]
    test = series[n_train + n_val :]

    X_tr = train[:-1].reshape(-1, 1)
    y_tr = train[1:].reshape(-1, 1)
    X_va = val[:-1].reshape(-1, 1)
    y_va = val[1:].reshape(-1, 1)
    X_te = test[:-1].reshape(-1, 1)
    y_te = test[1:].reshape(-1, 1)

    # PSANN: stateful, attached (detach=False for the state update rule)
    ps = PSANNRegressor(
        hidden_layers=2,
        hidden_width=32,
        epochs=200,
        batch_size=256,
        lr=1e-3,
        early_stopping=True,
        patience=25,
        stateful=True,
        state={"rho": 0.985, "beta": 1.0, "max_abs": 3.0, "init": 1.0, "detach": False},
        state_reset="none",
    )
    print("Training PSANN (stateful attached)...")
    ps.fit(X_tr, y_tr, validation_data=(X_va, y_va), verbose=1)
    seq_preds = ps.predict_sequence(X_te, reset_state=True, return_sequence=True).reshape(-1, 1)
    print("PSANN MSE:", float(np.mean((seq_preds - y_te) ** 2)), "R^2:", r2_score_np(y_te, seq_preds))

    # LSTM baseline on windows
    def windows(x, win=50):
        Xs, ys = [], []
        for i in range(len(x) - win):
            Xs.append(x[i : i + win])
            ys.append(x[i + win])
        Xs = np.asarray(Xs, dtype=np.float32)[..., None]
        ys = np.asarray(ys, dtype=np.float32)[:, None]
        return Xs, ys

    win = 50
    Xtr, ytr = windows(train, win)
    Xva, yva = windows(val, win)
    Xte, yte = windows(test, win)

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    lstm = LSTM1(64).to(dev)
    opt = torch.optim.Adam(lstm.parameters(), lr=1e-3)
    loss_fn = nn.MSELoss()

    dl_tr = DataLoader(TensorDataset(torch.from_numpy(Xtr), torch.from_numpy(ytr)), batch_size=256, shuffle=True)
    dl_va = DataLoader(TensorDataset(torch.from_numpy(Xva), torch.from_numpy(yva)), batch_size=512)

    best = math.inf
    patience = 15
    best_state = None
    print("Training LSTM...")
    for epoch in range(150):
        lstm.train()
        tot = 0.0
        cnt = 0
        for xb, yb in dl_tr:
            xb, yb = xb.to(dev), yb.to(dev)
            opt.zero_grad()
            pred = lstm(xb)
            loss = loss_fn(pred, yb)
            loss.backward()
            opt.step()
            tot += float(loss.item()) * xb.size(0)
            cnt += xb.size(0)
        tr = tot / max(cnt, 1)
        # val
        lstm.eval()
        with torch.no_grad():
            tot = 0.0
            cnt = 0
            for xb, yb in dl_va:
                xb, yb = xb.to(dev), yb.to(dev)
                pred = lstm(xb)
                loss = loss_fn(pred, yb)
                tot += float(loss.item()) * xb.size(0)
                cnt += xb.size(0)
            va = tot / max(cnt, 1)
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1:3d} - train {tr:.6f} - val {va:.6f}")
        if va + 1e-8 < best:
            best = va
            patience = 15
            best_state = {k: v.detach().cpu().clone() for k, v in lstm.state_dict().items()}
        else:
            patience -= 1
            if patience <= 0:
                break
    if best_state is not None:
        lstm.load_state_dict(best_state)
    # test
    with torch.no_grad():
        lstm_pred = lstm(torch.from_numpy(Xte).to(dev)).cpu().numpy()
    print("LSTM  MSE:", float(np.mean((lstm_pred - yte) ** 2)), "R^2:", r2_score_np(yte, lstm_pred))
