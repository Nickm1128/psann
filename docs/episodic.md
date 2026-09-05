# Episodic training

`EpisodicTrainer` composes a `PSANNRegressor` with an immutable `HISSOConfig`. The estimator continues to own architecture and preprocessing. The trainer owns reward dispatch, episode scheduling, warm start, prediction transformation, evaluation, and persistence.

```python
from psann import PSANNRegressor
from psann.architectures import ArchitectureConfig
from psann.episodic import EpisodeScheduleConfig, EpisodicTrainer, HISSOConfig
trainer = EpisodicTrainer(
    estimator=PSANNRegressor(architecture=ArchitectureConfig.dense(), epochs=3),
    strategy=HISSOConfig(
        schedule=EpisodeScheduleConfig(episode_length=8, batch_episodes=2),
        reward="finance", primary_transform="softmax", transition_penalty=0.001,
    ),
)
```

Call `trainer.fit(series)`, `trainer.predict(series)`, and `trainer.evaluate(series)`. Finance examples use positive price series and softmax allocation weights; their reported reward is defined by the selected strategy and transaction penalty. Custom rewards receive the documented reward tensors/context; registered reward names permit portable reconstruction. `SupervisedWarmStartConfig` optionally trains supplied targets before episodic optimization.

`trainer.save(path)` and `EpisodicTrainer.load(path, map_location="cpu")` preserve estimator, preprocessing, strategy, and supported fitted state. Callable rewards and context descriptors must satisfy the persistence rules in [migration](migration.md).

The [quickstart](../examples/quickstarts.py) trains, predicts, evaluates, and checks two checkpoint generations. [Allocation](../examples/26_hisso_unsupervised_allocation.py), [LSM allocation](../examples/27_hisso_lsm_allocation.py), and [HISSO configurations](../configs/hisso/) are maintained consumers. Run packaged logging with `python -m psann.scripts.hisso_log_run --help`.
