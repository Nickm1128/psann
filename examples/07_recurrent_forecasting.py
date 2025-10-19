import numpy as np

from psann import PSANNRegressor


def make_sine_series(T=500, freq=0.05, noise=0.05, seed=0):
    rs = np.random.RandomState(seed)
    t = np.arange(T, dtype=np.float32)
    x = np.sin(2 * np.pi * freq * t) + noise * rs.randn(T).astype(np.float32)
    return x.astype(np.float32)


if __name__ == "__main__":
    # Build supervised dataset for one-step ahead prediction: x_t -> y_t+1
    series = make_sine_series(T=2000, freq=0.01, noise=0.05)
    D = 1
    X = series[:-1].reshape(-1, D)
    y = series[1:].reshape(-1, 1)

    # Train a small stateful PSANN on shuffled points (toy setup)
    model = PSANNRegressor(
        hidden_layers=2,
        hidden_width=32,
        epochs=200,
        lr=1e-3,
        batch_size=256,
        early_stopping=True,
        patience=20,
        stateful=True,
        state={
            "init": 1.0,
            "rho": 0.98,  # persistence
            "beta": 1.0,  # update scale from |activation|
            "max_abs": 3.0,  # intelligent clipping bound
            "detach": True,  # no BPTT across steps (recommended for long sequences)
        },
        state_reset="none",  # carry state across batches for sequence-wise behavior
    )
    model.fit(X, y, verbose=1)

    # Forecast iteratively on a fresh sequence
    test = make_sine_series(T=600, freq=0.01, noise=0.05, seed=123)
    X_seq = test[:-1].reshape(-1, 1)
    # Reset state and run sequential prediction
    pred_last = model.predict_sequence(X_seq, reset_state=True, return_sequence=False)
    seq_preds = model.predict_sequence(X_seq, reset_state=True, return_sequence=True)
    print("Last-step prediction:", float(pred_last))
    print("Sequence preds shape:", seq_preds.shape)
