"""Hugging Face configuration for the canonical PSANN-LM architectures."""

from __future__ import annotations

from transformers import PretrainedConfig

from psannlm.architectures.config import normalize_lm_config, to_mapping


class PSANNConfig(PretrainedConfig):
    model_type = "psann"
    is_composition = True

    def __init__(self, lm_config=None, model_max_length=2048, **kwargs):
        if lm_config is None:
            # Transformers constructs a default configuration when computing diffs.
            lm_config = {"architecture": "residual", "vocab_size": 4}
        resolved = normalize_lm_config(lm_config, for_build=True)
        if isinstance(model_max_length, bool) or not isinstance(model_max_length, int):
            raise TypeError("model_max_length must be a positive integer.")
        if model_max_length <= 0:
            raise ValueError("model_max_length must be a positive integer.")
        if kwargs.pop("use_cache", False):
            raise ValueError("PSANN export does not support use_cache=True.")
        kwargs.pop("is_decoder", None)
        kwargs.pop("is_encoder_decoder", None)
        super().__init__(is_decoder=True, is_encoder_decoder=False, **kwargs)
        self.lm_config = to_mapping(resolved)
        self.model_max_length = model_max_length
        self.max_position_embeddings = model_max_length
        self.vocab_size = resolved.vocab_size
        self.hidden_size = resolved.d_model
        self.num_hidden_layers = resolved.n_layers
        self.num_attention_heads = resolved.n_heads
        self.intermediate_size = resolved.d_mlp
        self.tie_word_embeddings = False
        self.use_cache = False


PSANNConfig.register_for_auto_class()
