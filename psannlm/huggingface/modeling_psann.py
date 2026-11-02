"""Transformers adapter using the unchanged canonical PSANN inference runtime."""

from __future__ import annotations

import torch
from torch.nn import functional as F
from transformers import GenerationConfig, GenerationMixin, PreTrainedModel
from transformers.modeling_outputs import CausalLMOutput

from psannlm.architectures.registry import build_lm_model

from .configuration_psann import PSANNConfig


class PSANNForCausalLM(PreTrainedModel, GenerationMixin):
    # Transformers 4.x infers its overridable config_class default as None.
    config_class = PSANNConfig  # type: ignore[assignment]
    generation_config: GenerationConfig
    base_model_prefix = "model"
    main_input_name = "input_ids"
    _supports_cache_class = False

    def __init__(self, config, native_model=None):
        super().__init__(config)
        # Canonical initializers inspect scalar tensors and sparse connectivity.
        # Build on CPU even inside Transformers' default meta-device load context.
        with torch.device("cpu"):
            self.model = (
                native_model if native_model is not None else build_lm_model(config.lm_config).model
            )
        # Sparse layers can share connectivity buffers. SafeTensors stores named
        # tensors independently; separate the buffers without changing their values.
        seen = set()
        for child in self.model.modules():
            # HF 5 recreates non-persistent buffers as uninitialized tensors. Store
            # derived indices/ALiBi slopes explicitly in the HF artifact instead.
            child._non_persistent_buffers_set.clear()
            for name, buffer in child._buffers.items():
                if buffer is not None and buffer.numel():
                    pointer = buffer.untyped_storage().data_ptr()
                    if pointer in seen:
                        child._buffers[name] = buffer.clone()
                    else:
                        seen.add(pointer)
        # The canonical builder owns initialization; do not initialize a second time.
        self.post_init()

    def _init_weights(self, module):
        pass

    def get_input_embeddings(self):
        return self.model.embed

    def set_input_embeddings(self, value):
        self.model.embed = value

    def get_output_embeddings(self):
        return self.model.lm_head

    def set_output_embeddings(self, value):
        self.model.lm_head = value

    def prepare_inputs_for_generation(self, input_ids, attention_mask=None, **kwargs):
        if kwargs.get("use_cache") or kwargs.get("past_key_values") is not None:
            raise ValueError("PSANN export supports full-context generation only; use_cache=False.")
        return {"input_ids": input_ids, "attention_mask": attention_mask, "use_cache": False}

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        labels=None,
        use_cache=False,
        past_key_values=None,
        return_dict=None,
        position_ids=None,
        **kwargs,
    ):
        if use_cache or past_key_values is not None:
            raise ValueError("PSANN export supports full-context generation only; use_cache=False.")
        if position_ids is not None:
            raise ValueError(
                "Explicit position_ids are unsupported; positions follow unpadded tokens."
            )
        for name, value in kwargs.items():
            if name not in {"cache_position", "logits_to_keep", "num_items_in_batch"}:
                if value is not None:
                    raise ValueError(f"Unsupported PSANN model argument: {name}.")
        if input_ids is None or input_ids.ndim != 2 or input_ids.shape[1] == 0:
            raise ValueError("input_ids must be a nonempty (batch, sequence) tensor.")
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)
        if attention_mask.shape != input_ids.shape:
            raise ValueError("attention_mask must have the same shape as input_ids.")
        if not bool(((attention_mask == 0) | (attention_mask == 1)).all()):
            raise ValueError("attention_mask must contain only zero and one.")
        mask = attention_mask.to(device=input_ids.device, dtype=torch.bool)
        rows, losses, counts = [], [], []
        if labels is not None and labels.shape != input_ids.shape:
            raise ValueError("labels must have the same shape as input_ids.")
        # Unpadding also preserves temporal-convolution and spectral-gate semantics.
        for index in range(input_ids.shape[0]):
            positions = mask[index].nonzero(as_tuple=True)[0]
            if positions.numel() == 0:
                raise ValueError("attention_mask contains an entirely empty row.")
            if int(positions[-1] - positions[0] + 1) != positions.numel():
                raise ValueError(
                    "attention_mask must describe contiguous tokens (left/right padding)."
                )
            if positions.numel() > self.config.model_max_length:
                raise ValueError("Input exceeds the exported model_max_length.")
            ids = input_ids[index : index + 1, positions]
            logits = self.model(ids)
            restored = logits.new_zeros((input_ids.shape[1], logits.shape[-1]))
            rows.append(restored.index_copy(0, positions, logits[0]))
            if labels is not None:
                targets = labels[index, positions][1:].to(logits.device)
                predictions = logits[0, :-1].float()
                losses.append(
                    F.cross_entropy(predictions, targets, ignore_index=-100, reduction="sum")
                )
                counts.append((targets != -100).sum())
        logits = torch.stack(rows)
        loss = None
        if losses:
            loss = torch.stack(losses).sum() / torch.stack(counts).sum().clamp_min(1)
        if return_dict is False or (return_dict is None and not self.config.use_return_dict):
            return (loss, logits) if loss is not None else (logits,)
        return CausalLMOutput(loss=loss, logits=logits)


PSANNForCausalLM.register_for_auto_class("AutoModelForCausalLM")
