from __future__ import annotations

import numpy as np
import pytest
from sklearn.base import clone

from psann.architectures import ArchitectureConfig, ConvolutionConfig
from psann.estimators import PSANNRegressor


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
