"""Transformers tokenizer preserving PSANN's fitted vocabulary and ID semantics."""

from __future__ import annotations

import base64
import json
from pathlib import Path

from transformers import PreTrainedTokenizer

from psannlm.lm.data.tokenizer import Tokenizer


class PSANNTokenizer(PreTrainedTokenizer):
    vocab_files_names = {"state_file": "psann_tokenizer.json"}
    model_input_names = ["input_ids", "attention_mask"]

    def __init__(self, state_file=None, tokenizer_state=None, **kwargs):
        if tokenizer_state is None:
            if state_file is None:
                raise ValueError("A fitted PSANN tokenizer state is required.")
            tokenizer_state = json.loads(Path(state_file).read_text(encoding="utf-8"))
            if tokenizer_state.get("backend") == "sentencepiece":
                tokenizer_state["model"] = base64.b64decode(tokenizer_state["model"], validate=True)
        self.native = Tokenizer.from_state(tokenizer_state)
        self._size = self.native.vocab_size
        # Stable opaque token labels avoid collisions and preserve unused/remapped IDs.
        self._vocabulary = {self._label(i): i for i in range(self._size)}
        special = dict(pad=0, bos=1, eos=2, unk=3)
        if tokenizer_state["backend"] == "tokenizers" and tokenizer_state["passthrough_ids"]:
            special = tokenizer_state["special_ids"]
        for name, token_id in special.items():
            if not 0 <= token_id < self._size:
                raise ValueError(f"tokenizer.{name}_id does not index the fitted vocabulary.")
            kwargs[name + "_token"] = self._label(token_id)
        kwargs.setdefault("padding_side", "left")
        kwargs.setdefault("clean_up_tokenization_spaces", False)
        super().__init__(**kwargs)

    @staticmethod
    def _label(index):
        return f"<psann_id_{index}>"

    @property
    def vocab_size(self):
        return self._size

    def get_vocab(self):
        return dict(self._vocabulary, **self.added_tokens_encoder)

    def _convert_token_to_id(self, token):
        return self._vocabulary.get(token, self._vocabulary[self.unk_token])

    def _convert_id_to_token(self, index):
        return self._label(index)

    def _tokenize(self, text, **kwargs):
        return [self._label(i) for i in self.native.encode(text, add_specials=False)]

    def tokenize(self, text, **kwargs):
        # Do not split or normalize text around the opaque ID labels.
        return self._tokenize(text, **kwargs)

    def convert_tokens_to_string(self, tokens):
        return self.native.decode(
            [self._convert_token_to_id(t) for t in tokens], skip_specials=False
        )

    def _decode(self, token_ids, skip_special_tokens=False, **kwargs):
        return self.native.decode(token_ids, skip_specials=skip_special_tokens)

    def build_inputs_with_special_tokens(self, token_ids_0, token_ids_1=None):
        ids = [self.bos_token_id] + token_ids_0 + [self.eos_token_id]
        if token_ids_1 is not None:
            ids += [self.bos_token_id] + token_ids_1 + [self.eos_token_id]
        return ids

    def get_special_tokens_mask(
        self, token_ids_0, token_ids_1=None, already_has_special_tokens=False
    ):
        if already_has_special_tokens:
            return [int(i in self.all_special_ids) for i in token_ids_0]
        result = [1] + [0] * len(token_ids_0) + [1]
        if token_ids_1 is not None:
            result += [1] + [0] * len(token_ids_1) + [1]
        return result

    def save_vocabulary(self, save_directory, filename_prefix=None):
        directory = Path(save_directory)
        directory.mkdir(parents=True, exist_ok=True)
        prefix = (filename_prefix + "-") if filename_prefix else ""
        state = self.native.to_state()
        if state["backend"] == "sentencepiece":
            (directory / (prefix + "tokenizer.model")).write_bytes(state["model"])
            state["model"] = base64.b64encode(state["model"]).decode("ascii")
        elif state["backend"] == "tokenizers":
            (directory / (prefix + "tokenizer.json")).write_text(state["model"], encoding="utf-8")
        path = directory / (prefix + "psann_tokenizer.json")
        path.write_text(json.dumps(state, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        return (str(path),)


PSANNTokenizer.register_for_auto_class("AutoTokenizer")
