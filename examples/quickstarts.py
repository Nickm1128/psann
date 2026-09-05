"""Small, executable public workflows with two checkpoint round trips.

Run from a source checkout: python examples/quickstarts.py --workflow core
Use --workflow preprocessing, episodic, or lm for the other task APIs.
"""

import argparse
from pathlib import Path

import numpy as np
import torch

from psann import PSANNRegressor
from psann.architectures import ArchitectureConfig
from psann.episodic import EpisodeScheduleConfig, EpisodicTrainer, HISSOConfig
from psann.preprocessing import LSMConfig, LSMPretrainingConfig, PreprocessorConfig


def regression_data():
    x = np.linspace(-1, 1, 64, dtype=np.float32)[:, None]
    return x, np.sin(3 * x).astype(np.float32)


def core(output, *, device="cpu", preprocessing=False):
    x, y = regression_data()
    preprocessor = (
        PreprocessorConfig(
            LSMConfig.dense(
                output_dim=8,
                hidden_layers=1,
                hidden_units=8,
                random_state=7,
                pretraining=LSMPretrainingConfig(epochs=2),
            )
        )
        if preprocessing
        else None
    )
    estimator = PSANNRegressor(
        architecture=ArchitectureConfig.dense(),
        preprocessor=preprocessor,
        hidden_layers=1,
        hidden_units=16,
        epochs=8,
        batch_size=16,
        lr=0.01,
        random_state=7,
        device=device,
    )
    estimator.fit(x, y)
    predictions = estimator.predict(x)
    for generation in (1, 2):
        path = Path(output) / f"core-{generation}.pt"
        estimator.save(path)
        estimator = PSANNRegressor.load(path, map_location=device)
        np.testing.assert_allclose(estimator.predict(x), predictions, rtol=0, atol=0)
    return estimator, x, y, predictions


def episodic(output, *, device="cpu"):
    t = np.linspace(0, 2, 64, dtype=np.float32)
    prices = np.stack([np.exp(0.2 * t), np.exp(-0.1 * t)], axis=1)
    estimator = PSANNRegressor(
        architecture=ArchitectureConfig.dense(),
        hidden_layers=1,
        hidden_units=8,
        epochs=8,
        lr=0.01,
        output_shape=(2,),
        random_state=7,
        device=device,
    )
    trainer = EpisodicTrainer(
        estimator=estimator,
        strategy=HISSOConfig(
            schedule=EpisodeScheduleConfig(episode_length=8, batch_episodes=2),
            reward="finance",
            primary_transform="softmax",
            transition_penalty=0.001,
        ),
    )
    trainer.fit(prices)
    actions = trainer.predict(prices)
    reward = trainer.evaluate(prices)
    for generation in (1, 2):
        path = Path(output) / f"episodic-{generation}.pt"
        trainer.save(path)
        trainer = EpisodicTrainer.load(path, map_location=device)
        np.testing.assert_allclose(trainer.predict(prices), actions, rtol=0, atol=0)
        assert trainer.evaluate(prices) == reward
    return trainer, prices, actions, reward


def lm(output, *, device="cpu"):
    from psannlm import LMArchitectureConfig, LMConfig, PSANNLM, PSANNLMDataPrep, TrainConfig

    torch.manual_seed(7)
    data = PSANNLMDataPrep(
        ["waves learn useful patterns in small sequences " * 16],
        tokenizer="simple",
        max_length=8,
    )
    model = PSANNLM(
        config=LMConfig(
            architecture=LMArchitectureConfig.wave(),
            d_model=16,
            n_layers=1,
            n_heads=2,
            d_mlp=32,
            vocab_size=data.vocab_size,
        ),
        device=device,
    )
    model.fit(
        data,
        train=TrainConfig(
            epochs=1,
            steps_per_epoch=3,
            batch_tokens=16,
            lr=0.003,
            warmup_steps=0,
            amp="fp32",
            ddp="off",
            checkpoint_dir=str(Path(output) / "trainer"),
        ),
    )
    generated = model.generate("waves learn", max_new_tokens=4, temperature=0)
    for generation in (1, 2):
        path = Path(output) / f"lm-{generation}.pt"
        model.save(str(path))
        model = PSANNLM.load(str(path), map_location=device)
        assert model.generate("waves learn", max_new_tokens=4, temperature=0) == generated
    return model, data, generated


def main():
    parser = argparse.ArgumentParser(description="Train and persist canonical PSANN workflows")
    parser.add_argument(
        "--workflow", choices=("core", "preprocessing", "episodic", "lm"), default="core"
    )
    parser.add_argument("--output", type=Path, default=Path("outputs/quickstart"))
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    if args.workflow in ("core", "preprocessing"):
        result = core(
            args.output, device=args.device, preprocessing=args.workflow == "preprocessing"
        )
        print("Training MSE:", float(np.mean((result[3] - result[2]) ** 2)))
    elif args.workflow == "episodic":
        print("Evaluation reward:", episodic(args.output, device=args.device)[3])
    else:
        print(lm(args.output, device=args.device)[2])


if __name__ == "__main__":
    main()
