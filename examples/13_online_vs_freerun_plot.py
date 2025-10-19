import matplotlib.pyplot as plt
import numpy as np

from psann import PSANNRegressor


def make_series(T=2500, f=0.02, amp_f=0.002, noise=0.05, seed=0):
    rs = np.random.RandomState(seed)
    t = np.arange(T, dtype=np.float32)
    amp = 1.0 + 0.6 * np.sin(2 * np.pi * amp_f * t)
    x = amp * np.sin(2 * np.pi * f * t + 0.25) + noise * rs.randn(T).astype(np.float32)
    return x.astype(np.float32)


if __name__ == "__main__":
    # Generate and split
    series = make_series(T=3500, seed=11)
    X = series[:-1].reshape(-1, 1)
    y = series[1:].reshape(-1, 1)
    n = len(X)
    n_tr = int(0.7 * n)
    n_va = int(0.15 * n)
    X_tr, y_tr = X[:n_tr], y[:n_tr]
    X_va, y_va = X[n_tr : n_tr + n_va], y[n_tr : n_tr + n_va]
    X_te, y_te = X[n_tr + n_va :], y[n_tr + n_va :]

    model = PSANNRegressor(
        hidden_layers=2,
        hidden_width=32,
        epochs=500,
        batch_size=256,
        lr=1e-3,
        stream_lr=3e-4,
        early_stopping=False,
        # patience=20,
        stateful=True,
        state={"rho": 0.985, "beta": 1.0, "max_abs": 3.0, "init": 1.0, "detach": True},
        state_reset="epoch",
    )
    model.fit(X_tr, y_tr, validation_data=(X_va, y_va), verbose=1)

    # Predict on a holdout slice
    free = model.predict_sequence(X_te, reset_state=True, return_sequence=True).reshape(-1)
    online = model.predict_sequence_online(X_te, y_te, reset_state=True).reshape(-1)

    t = np.arange(len(y_te))
    fig, axs = plt.subplots(2, 1, figsize=(10, 7), sharex=True)

    axs[0].plot(t, y_te.reshape(-1), label="target", color="black", linewidth=1)
    axs[0].plot(t, free, label="free-run", alpha=0.8)
    axs[0].plot(t, online, label="online-updated", alpha=0.8)
    axs[0].set_title("PSANN: Free-run vs Online Updated Predictions")
    axs[0].legend(loc="upper right")

    # Error curves
    e_free = (free - y_te.reshape(-1)) ** 2
    e_online = (online - y_te.reshape(-1)) ** 2
    axs[1].plot(t, e_free, label="MSE free", alpha=0.8)
    axs[1].plot(t, e_online, label="MSE online", alpha=0.8)
    axs[1].set_ylabel("Squared error")
    axs[1].legend(loc="upper right")

    axs[1].set_xlabel("Time index")
    plt.tight_layout()
    plt.show()
