from __future__ import annotations

import numpy as np
import pytest

from psann.architectures import (
    ArchitectureConfig,
    ConvolutionConfig,
    GeometryConfig,
    ResidualConfig,
    SequenceConfig,
)
from psann.estimators import PSANNRegressor
from sklearn.base import clone


def test_canonical_dense_fit_predict_clone_and_nested_params():
    X = np.arange(16, dtype=np.float32).reshape(8, 2)
    y = X.sum(axis=1)
    estimator = PSANNRegressor(epochs=1, batch_size=4, random_state=0)
    assert clone(estimator).architecture == estimator.architecture
    estimator.set_params(architecture__activation__decay_init=0.2)
    assert estimator.architecture.activation.decay_init == 0.2
    estimator.fit(X, y)
    assert estimator.predict(X[:2]).shape == (2,)


def test_explicit_architecture_rejects_legacy_architecture_keyword():
    with pytest.raises(ValueError, match="architecture conflicts"):
        PSANNRegressor(architecture=ArchitectureConfig.dense(), preserve_shape=True)


def test_canonical_convolutional_fit_uses_registry():
    X = np.ones((6, 1, 4), dtype=np.float32)
    y = np.ones(6, dtype=np.float32)
    estimator = PSANNRegressor(
        architecture=ArchitectureConfig.convolutional(convolution=ConvolutionConfig(channels=4)),
        epochs=1,
        batch_size=3,
        random_state=0,
    ).fit(X, y)
    assert estimator._architecture_capabilities_.kind == "convolutional"
    assert estimator.predict(X[:2]).shape == (2,)


@pytest.mark.parametrize(
    ("architecture", "X"),
    [
        (ArchitectureConfig.dense(residual=ResidualConfig()), np.ones((6, 2), dtype=np.float32)),
        (ArchitectureConfig.for_wave(), np.ones((6, 2), dtype=np.float32)),
        (ArchitectureConfig.for_sequence(), np.ones((6, 2), dtype=np.float32)),
        (
            ArchitectureConfig.geometric_sparse(geometry=GeometryConfig(shape=(1, 2))),
            np.ones((6, 2), dtype=np.float32),
        ),
    ],
)
def test_canonical_non_dense_architectures_fit_and_predict(architecture, X):
    estimator = PSANNRegressor(
        architecture=architecture, hidden_layers=2, epochs=1, batch_size=3, random_state=0
    ).fit(X, np.ones(len(X), dtype=np.float32))
    assert estimator._architecture_capabilities_.kind == architecture.kind
    assert estimator.predict(X[:2]).shape == (2,)
