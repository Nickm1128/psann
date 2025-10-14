import math
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from psann import PSANNRegressor


def make_amplitude_modulated_sine(T=4000, f=0.02, amp_f=0.002, noise=0.05, seed=0):
    rs = np.random.RandomState(seed)
    t = np.arange(T, dtype=np.float32)
    amp = 1.0 + 0.5 * np.sin(2 * np.pi * amp_f * t)
    x = amp * np.sin(2 * np.pi * f * t + 0.25) + noise * rs.randn(T).astype(np.float32)
    return x.astype(np.float32)


def make_windows(series: np.ndarray, win: int):
    Xs = []
    ys = []
    for i in range(len(series) - win):
        Xs.append(series[i : i + win])
        ys.append(series[i + win])
    Xs = np.asarray(Xs, dtype=np.float32)
    ys = np.asarray(ys, dtype=np.float32)
    Xs = Xs[..., None]  # (N, win, 1)
    ys = ys[:, None]    # (N, 1)
    return Xs, ys


class LSTMForecaster(nn.Module):
    def __init__(self, input_size=1, hidden_size=64, num_layers=1, dropout=0.0):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers=num_layers, batch_first=True, dropout=dropout)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = out[:, -1, :]
        return self.fc(out)


def r2_score_np(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = y_true.reshape(-1)
    y_pred = y_pred.reshape(-1)
    u = np.sum((y_true - y_pred) ** 2)
    v = np.sum((y_true - y_true.mean()) ** 2)
    return float(1.0 - (u / v if v != 0 else np.nan))


if __name__ == "__main__":
    # Generate data
    series = make_amplitude_modulated_sine(T=5000, f=0.02, amp_f=0.002, noise=0.05, seed=123)

    # Split into train/val/test by time
    n = len(series)
    n_train = int(0.7 * n)
    n_val = int(0.15 * n)
    train = series[:n_train]
    val = series[n_train : n_train + n_val]
    test = series[n_train + n_val :]

    # PSANN: one-step pairs (x_t -> y_{t+1})
    X_train_ps = train[:-1].reshape(-1, 1)
    y_train_ps = train[1:].reshape(-1, 1)
    X_val_ps = val[:-1].reshape(-1, 1)
    y_val_ps = val[1:].reshape(-1, 1)
    X_test_ps = test[:-1].reshape(-1, 1)
    y_test_ps = test[1:].reshape(-1, 1)

    psann = PSANNRegressor(
        hidden_layers=2,
        hidden_width=32,
        epochs=250,
        batch_size=256,
        lr=1e-3,
        early_stopping=True,
        patience=25,
        stateful=True,
        state={"rho": 0.985, "beta": 1.0, "max_abs": 3.0, "init": 1.0, "detach": True},
        state_reset="none",
    )
    print("Training PSANN (stateful)...")
    psann.fit(X_train_ps, y_train_ps, validation_data=(X_val_ps, y_val_ps), verbose=1)

    # Evaluate PSANN on test via iterative sequence prediction using true x_t
    psann.reset_state()
    psann_preds = psann.predict_sequence(X_test_ps, reset_state=True, return_sequence=True)
    psann_preds = psann_preds.reshape(-1, 1)
    psann_mse = float(np.mean((psann_preds - y_test_ps) ** 2))
    psann_r2 = r2_score_np(y_test_ps, psann_preds)
    print(f"PSANN Test MSE: {psann_mse:.6f} | R^2: {psann_r2:.4f}")

    # LSTM: sliding windows
    win = 50
    X_train_lstm, y_train_lstm = make_windows(train, win)
    X_val_lstm, y_val_lstm = make_windows(val, win)
    X_test_lstm, y_test_lstm = make_windows(test, win)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = LSTMForecaster(input_size=1, hidden_size=64, num_layers=1, dropout=0.0).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.MSELoss()

    ds_tr = TensorDataset(torch.from_numpy(X_train_lstm), torch.from_numpy(y_train_lstm))
    ds_va = TensorDataset(torch.from_numpy(X_val_lstm), torch.from_numpy(y_val_lstm))
    dl_tr = DataLoader(ds_tr, batch_size=256, shuffle=True)
    dl_va = DataLoader(ds_va, batch_size=512)

    best = math.inf
    patience = 15
    best_state = None
    print("Training LSTM...")
    for epoch in range(150):
        model.train()
        total = 0.0
        count = 0
        for xb, yb in dl_tr:
            xb = xb.to(device)
            yb = yb.to(device)
            opt.zero_grad()
            pred = model(xb)
            loss = loss_fn(pred, yb)
            loss.backward()
            opt.step()
            total += float(loss.item()) * xb.size(0)
            count += xb.size(0)
        tr_loss = total / max(count, 1)

        # Validation
        model.eval()
        with torch.no_grad():
            tot = 0.0
            cnt = 0
            for xb, yb in dl_va:
                xb = xb.to(device)
                yb = yb.to(device)
                pred = model(xb)
                loss = loss_fn(pred, yb)
                tot += float(loss.item()) * xb.size(0)
                cnt += xb.size(0)
            va_loss = tot / max(cnt, 1)

        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1:3d} | train: {tr_loss:.6f} | val: {va_loss:.6f}")

        if va_loss + 1e-8 < best:
            best = va_loss
            patience = 15
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            patience -= 1
            if patience <= 0:
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    # Test LSTM
    model.eval()
    with torch.no_grad():
        Xb = torch.from_numpy(X_test_lstm).to(device)
        lstm_preds = model(Xb).cpu().numpy()
    lstm_mse = float(np.mean((lstm_preds - y_test_lstm) ** 2))
    lstm_r2 = r2_score_np(y_test_lstm, lstm_preds)
    print(f"LSTM  Test MSE: {lstm_mse:.6f} | R^2: {lstm_r2:.4f}")

    print("\nSummary:")
    print(f"  PSANN (stateful): MSE={psann_mse:.6f}, R2={psann_r2:.4f}")
    print(f"  LSTM           : MSE={lstm_mse:.6f}, R2={lstm_r2:.4f}")
