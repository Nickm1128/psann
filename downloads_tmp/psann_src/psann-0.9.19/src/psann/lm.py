from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import warnings

from .nn import PSANNNet, ResidualPSANNNet, WithPreprocessor
from .sklearn import ResPSANNRegressor
from .tokenizer import BaseTokenizer, SimpleWordTokenizer
from .embeddings import SineTokenEmbedder
from .utils import choose_device, seed_all


@dataclass
class LMConfig:
    embedding_dim: int = 64
    extras_dim: int = 0
    episode_length: int = 64
    batch_episodes: int = 32
    random_state: Optional[int] = None
    # Perplexity monitoring
    ppx_every: int = 0           # 0 disables periodic perplexity
    ppx_temperature: float = 1.0
    # Curriculum learning over token stream (progressively unlock prefix)
    curriculum_type: Optional[str] = None  # 'progressive_span' or None
    curriculum_warmup_epochs: int = 10     # epochs to reach full coverage
    curriculum_min_frac: float = 0.1       # starting fraction of stream
    curriculum_max_frac: float = 1.0       # final fraction


class PSANNLanguageModel:
    """PSANN-LM built on top of :class:`ResPSANNRegressor`.

    The core regression network is the residual PSANN architecture while extras
    act as a lightweight latent state. During training we roll a sequential
    buffer of extras alongside the embeddings; the buffer is randomly
    initialised on the first pass and refreshed from the model's own predictions
    after fitting, mimicking the original LMExtras behaviour where extras were
    fed forward step by step.
    """

    def __init__(
        self,
        *,
        tokenizer: Optional[BaseTokenizer] = None,
        embedder: Optional[SineTokenEmbedder] = None,
        lm_cfg: Optional[LMConfig] = None,
        hidden_layers: int = 8,
        hidden_units: Optional[int] = None,
        hidden_width: Optional[int] = 128,
        activation_type: str = "psann",
        w0: float = 30.0,
        device: torch.device | str = "auto",
        random_state: Optional[int] = None,
        w0_first: float = 12.0,
        w0_hidden: float = 1.0,
        norm: str = "rms",
        drop_path_max: float = 0.0,
        residual_alpha_init: float = 0.0,
    ) -> None:
        self.tokenizer = tokenizer or SimpleWordTokenizer()
        self.embedder = embedder
        self.cfg = lm_cfg or LMConfig()
        self.hidden_layers = int(hidden_layers)
        user_set_units = hidden_units is not None
        user_set_width = hidden_width is not None
        if user_set_width and not user_set_units:
            warnings.warn('`hidden_width` is deprecated; use `hidden_units` instead.', DeprecationWarning, stacklevel=2)
        if user_set_units and user_set_width and int(hidden_units) != int(hidden_width):
            warnings.warn('`hidden_units` overrides `hidden_width` because the values differ.', UserWarning, stacklevel=2)
        units_val = hidden_units if user_set_units else hidden_width
        if units_val is None:
            units_val = 128
        units = int(units_val)
        self.hidden_units = units
        self.hidden_width = units
        self.activation_type = activation_type
        self.w0 = float(w0)
        self.device = choose_device(device)
        self.random_state = random_state
        self.w0_first = float(w0_first)
        self.w0_hidden = float(w0_hidden)
        self.norm = str(norm)
        self.drop_path_max = float(drop_path_max)
        self.residual_alpha_init = float(residual_alpha_init)
        self.model_: Optional[nn.Module] = None
        self.regressor_: Optional[ResPSANNRegressor] = None
        self.regressor_params_: Optional[Dict[str, object]] = None
        self.history: List[dict] = []
        self._extras_cache: Optional[np.ndarray] = None

    # ------------------------------------------------------------------ helpers
    def _ensure_embedder(self, vocab_size: int) -> None:
        seed_all(self.random_state)
        if self.embedder is None:
            self.embedder = SineTokenEmbedder(self.cfg.embedding_dim)
        self.embedder.set_vocab_size(vocab_size)
        try:
            self.embedder.to(self.device)
        except Exception:
            pass

    def _create_model_module(self) -> nn.Module:
        D = int(self.cfg.embedding_dim)
        K = int(self.cfg.extras_dim)
        core = ResidualPSANNNet(
            D + K,
            D + K,
            hidden_layers=self.hidden_layers,
            hidden_width=self.hidden_width,
            act_kw={},
            activation_type=self.activation_type,
            w0_first=self.w0_first,
            w0_hidden=self.w0_hidden,
            norm=self.norm,
            drop_path_max=self.drop_path_max,
            residual_alpha_init=self.residual_alpha_init,
        )
        return WithPreprocessor(None, core).to(self.device)

    def _concat_corpus(self, corpus: Iterable[str]) -> np.ndarray:
        ids: List[int] = []
        for text in corpus:
            ids.extend(self.tokenizer.encode(text, add_bos=True, add_eos=True))
        return np.asarray(ids, dtype=np.int64)

    def _prepare_training_pairs(self, token_ids: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        if token_ids.shape[0] < 2:
            raise ValueError("Need at least two tokens to build training pairs")
        assert self.embedder is not None
        ids_prev = torch.from_numpy(token_ids[:-1]).to(self.device)
        ids_next = torch.from_numpy(token_ids[1:]).to(self.device)
        D = int(self.cfg.embedding_dim)
        K = int(self.cfg.extras_dim)
        with torch.no_grad():
            emb_prev = self.embedder(ids_prev).reshape(-1, D)
            emb_next = self.embedder(ids_next).reshape(-1, D)
        X = emb_prev.detach().cpu().numpy().astype(np.float32, copy=False)
        y = emb_next.detach().cpu().numpy().astype(np.float32, copy=False)
        if K > 0:
            if self._extras_cache is None or self._extras_cache.shape[0] != token_ids.shape[0]:
                seed = self.random_state if self.random_state is not None else None
                rng = np.random.default_rng(seed)
                extras_seq = rng.normal(loc=0.0, scale=0.1, size=(token_ids.shape[0], K)).astype(np.float32)
                extras_seq[0] = 0.0
            else:
                extras_seq = self._extras_cache.astype(np.float32, copy=True)
            extras_in = extras_seq[:-1]
            extras_next = extras_seq[1:]
            X = np.concatenate([X, extras_in], axis=1)
            y = np.concatenate([y, extras_next], axis=1)
            self._extras_cache = extras_seq
        else:
            self._extras_cache = None
        return X, y, token_ids[1:]

    def _refresh_extras_cache(self, token_ids: np.ndarray) -> None:
        K = int(self.cfg.extras_dim)
        if K <= 0 or self.model_ is None or self.embedder is None:
            self._extras_cache = None
            return
        D = int(self.cfg.embedding_dim)
        N = int(token_ids.shape[0])
        extras_seq = np.zeros((N, K), dtype=np.float32)
        extras_t = torch.zeros((1, K), device=self.device)
        for idx in range(N - 1):
            tok = torch.tensor([token_ids[idx]], dtype=torch.long, device=self.device)
            emb = self.embedder(tok)
            inp = torch.cat([emb, extras_t], dim=-1)
            with torch.no_grad():
                y = self.model_(inp)
            next_extras = y[:, D : D + K].detach()
            extras_seq[idx + 1] = next_extras.cpu().numpy()
            extras_t = next_extras
        self._extras_cache = extras_seq

    # --------------------------------------------------------------------- train
    def fit(
        self,
        corpus: List[str],
        *,
        epochs: int = 50,
        lr: float = 1e-3,
        noisy: Optional[float] = None,
        verbose: int = 1,
        ppx_every: Optional[int] = None,
        ppx_temperature: Optional[float] = None,
        curriculum_type: Optional[str] = None,
        curriculum_warmup_epochs: Optional[int] = None,
        curriculum_min_frac: Optional[float] = None,
        curriculum_max_frac: Optional[float] = None,
        extras_loss_weight: Optional[float] = None,
        extras_loss_schedule: Optional[str] = None,
        extras_loss_cycle: Optional[int] = None,
    ) -> "PSANNLanguageModel":
        self.tokenizer.fit(corpus)
        vocab_size = self.tokenizer.vocab_size
        self._ensure_embedder(vocab_size)
        assert self.embedder is not None

        if ppx_every is not None:
            self.cfg.ppx_every = int(ppx_every)
        if ppx_temperature is not None:
            self.cfg.ppx_temperature = float(ppx_temperature)
        if curriculum_type is not None:
            self.cfg.curriculum_type = str(curriculum_type)
        if curriculum_warmup_epochs is not None:
            self.cfg.curriculum_warmup_epochs = int(curriculum_warmup_epochs)
        if curriculum_min_frac is not None:
            self.cfg.curriculum_min_frac = float(curriculum_min_frac)
        if curriculum_max_frac is not None:
            self.cfg.curriculum_max_frac = float(curriculum_max_frac)

        token_ids = self._concat_corpus(corpus)
        X_train, y_train, next_ids = self._prepare_training_pairs(token_ids)

        batch_size = int(self.cfg.batch_episodes) * max(1, int(self.cfg.episode_length))
        if batch_size <= 0:
            batch_size = 128

        D = int(self.cfg.embedding_dim)
        K = int(self.cfg.extras_dim)

        weight_val = extras_loss_weight
        if weight_val is None:
            weight_val = 1.0 if K > 0 else 0.0
        extras_weight = float(weight_val or 0.0)
        extras_mode = (extras_loss_schedule or ("alternate" if (extras_weight > 0.0 and K > 0) else "joint")).lower()
        if extras_mode not in {"joint", "alternate"}:
            raise ValueError("extras_loss_schedule must be 'joint' or 'alternate'")
        extras_cycle = max(1, int(extras_loss_cycle) if extras_loss_cycle is not None else 2)
        loss_state = {"step": 0}

        def _language_model_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
            loss_state["step"] += 1
            step = loss_state["step"]
            primary_loss = F.mse_loss(pred[:, :D], target[:, :D])
            if K <= 0 or extras_weight <= 0.0:
                return primary_loss
            extras_loss = F.mse_loss(pred[:, D:D + K], target[:, D:D + K])
            if extras_mode == "joint":
                return primary_loss + extras_weight * extras_loss
            if extras_mode == "alternate":
                if extras_cycle <= 1:
                    return extras_weight * extras_loss
                if (step - 1) % extras_cycle == 0:
                    return primary_loss
                return extras_weight * extras_loss
            raise ValueError("extras_loss_schedule must be 'joint' or 'alternate'")

        self.regressor_ = ResPSANNRegressor(
            hidden_layers=self.hidden_layers,
            hidden_width=self.hidden_width,
            hidden_units=self.hidden_units,
            epochs=int(epochs),
            batch_size=batch_size,
            lr=float(lr),
            activation_type=self.activation_type,
            device=self.device,
            random_state=self.random_state,
            w0=self.w0,
            w0_first=self.w0_first,
            w0_hidden=self.w0_hidden,
            norm=self.norm,
            drop_path_max=self.drop_path_max,
            residual_alpha_init=self.residual_alpha_init,
            extras=int(self.cfg.extras_dim),
            loss=_language_model_loss,
        )
        self.regressor_.fit(X_train, y_train, verbose=int(verbose), noisy=noisy)
        self.model_ = self.regressor_.model_
        if self.model_ is None:
            raise RuntimeError("ResPSANNRegressor failed to produce a trained model")
        self.model_.to(self.device)
        try:
            self.model_.eval()
        except Exception:
            pass
        self.regressor_params_ = self.regressor_.get_params(deep=False)

        self.history = []
        with torch.no_grad():
            X_tensor = torch.from_numpy(X_train).to(self.device)
            pred = self.model_(X_tensor)
            tgt = torch.from_numpy(y_train[:, :D]).to(self.device)
            mse = torch.mean((pred[:, :D] - tgt) ** 2).item()
            record = {"epoch": int(epochs), "train_mse": float(mse)}
            if K > 0:
                extras_target = torch.from_numpy(y_train[:, D:D + K]).to(self.device)
                extras_mse = torch.mean((pred[:, D:D + K] - extras_target) ** 2).item()
                record["extras_mse"] = float(extras_mse)
                record["extras_loss_weight"] = float(extras_weight)
                record["extras_loss_mode"] = extras_mode
                record["extras_loss_cycle"] = int(extras_cycle)
            if int(self.cfg.ppx_every) > 0:
                pred_bt = pred[:, :D].reshape(-1, 1, D)
                next_ids_tensor = torch.from_numpy(next_ids.astype(np.int64)).reshape(-1, 1).to(self.device)
                record["perplexity"] = self._batch_perplexity(pred_bt, next_ids_tensor)
            self.history.append(record)

        self._refresh_extras_cache(token_ids)
        return self

    # ---------------------------------------------------------------- inference
    @torch.no_grad()
    def _batch_perplexity(self, pred: torch.Tensor, next_ids: torch.Tensor) -> float:
        """Compute perplexity using cosine-sim softmax over vocab embeddings."""
        D = pred.shape[-1]
        try:
            if any(p.requires_grad for p in self.embedder.parameters()):
                self.embedder._rebuild_table()
        except Exception:
            pass
        table = self.embedder.embedding_matrix()  # (V,D)
        if table.numel() == 0:
            return float("nan")
        tn = table / (table.norm(p=2, dim=1, keepdim=True) + 1e-8)
        y = pred.reshape(-1, D)
        yn = y / (y.norm(p=2, dim=1, keepdim=True) + 1e-8)
        logits = torch.matmul(yn, tn.T) / max(1e-6, float(self.cfg.ppx_temperature))
        tgt = next_ids.reshape(-1).to(device=logits.device, dtype=torch.long)
        log_probs = logits - torch.logsumexp(logits, dim=1, keepdim=True)
        nll = -log_probs[torch.arange(logits.shape[0]), tgt].mean()
        return float(torch.exp(nll).item())

    @torch.no_grad()
    def predict(self, text: str, *, return_embedding: bool = False) -> str | np.ndarray:
        if self.model_ is None or self.embedder is None:
            raise RuntimeError("Model not fitted")
        ids = self.tokenizer.encode(text, add_bos=True, add_eos=False)
        if len(ids) == 0:
            return ""
        last = torch.tensor([ids[-1]], dtype=torch.long, device=self.device)
        emb = self.embedder(last)
        K = int(self.cfg.extras_dim)
        if K > 0:
            inp = torch.cat([emb, torch.zeros((1, K), device=self.device)], dim=-1)
        else:
            inp = emb
        y = self.model_(inp)
        D = int(self.cfg.embedding_dim)
        y_emb = y[:, :D]
        if return_embedding:
            return y_emb[0].detach().cpu().numpy()
        try:
            if any(p.requires_grad for p in self.embedder.parameters()):
                self.embedder._rebuild_table()
        except Exception:
            pass
        table = self.embedder.embedding_matrix()
        v = y_emb[0]
        vn = v / (v.norm(p=2) + 1e-8)
        tn = table / (table.norm(p=2, dim=1, keepdim=True) + 1e-8)
        sim = torch.matmul(tn, vn)
        idx = int(torch.argmax(sim).item())
        return self.tokenizer.decode([idx])

    @torch.no_grad()
    def generate(self, prompt: str, *, max_tokens: int = 20) -> str:
        if self.model_ is None or self.embedder is None:
            raise RuntimeError("Model not fitted")
        ids = self.tokenizer.encode(prompt, add_bos=True, add_eos=False)
        if len(ids) == 0:
            return prompt
        K = int(self.cfg.extras_dim)
        extras_t = torch.zeros((1, K), device=self.device) if K > 0 else None
        for _ in range(int(max_tokens)):
            last_id = torch.tensor([ids[-1]], dtype=torch.long, device=self.device)
            emb = self.embedder(last_id)
            if K > 0 and extras_t is not None:
                inp = torch.cat([emb, extras_t], dim=-1)
            else:
                inp = emb
            y = self.model_(inp)
            D = int(self.cfg.embedding_dim)
            y_emb = y[:, :D]
            if K > 0:
                extras_t = y[:, D : D + K]
            try:
                if any(p.requires_grad for p in self.embedder.parameters()):
                    self.embedder._rebuild_table()
            except Exception:
                pass
            table = self.embedder.embedding_matrix()
            vn = y_emb[0] / (y_emb[0].norm(p=2) + 1e-8)
            tn = table / (table.norm(p=2, dim=1, keepdim=True) + 1e-8)
            sim = torch.matmul(tn, vn)
            idx = int(torch.argmax(sim).item())
            ids.append(idx)
            try:
                eos_id = self.tokenizer._tok2id.get(SimpleWordTokenizer.EOS, -1)  # type: ignore[attr-defined]
            except Exception:
                eos_id = -1
            if idx == eos_id:
                break
        return self.tokenizer.decode(ids)

    def gen(self, prompt: str, *, max_tokens: int = 20) -> str:
        return self.generate(prompt, max_tokens=max_tokens)

    # ---------------------------------------------------------------- persistence
    def save(self, path: str) -> None:
        if self.model_ is None or self.embedder is None:
            raise RuntimeError("Model not fitted")
        tok_meta = {
            "type": type(self.tokenizer).__name__,
            "config": getattr(self.tokenizer, "get_config", lambda: None)(),
            "id2tok": getattr(self.tokenizer, "_id2tok", None),
            "lowercase": getattr(self.tokenizer, "lowercase", True),
            "max_vocab": getattr(self.tokenizer, "max_vocab", None),
        }
        emb_meta = {
            "type": type(self.embedder).__name__,
            "embedding_dim": int(self.embedder.embedding_dim),
            "base": float(getattr(self.embedder, "base", 10000.0)),
            "scale": float(getattr(self.embedder, "scale", 1.0)),
            "trainable": bool(getattr(self.embedder, "trainable", False)),
        }
        emb_state = self.embedder.state_dict()
        reg_params = self.regressor_params_ or (
            self.regressor_.get_params(deep=False) if self.regressor_ is not None else None
        )
        payload = {
            "class": "PSANNLanguageModel",
            "params": {
                "hidden_layers": self.hidden_layers,
                "hidden_units": self.hidden_units,
                "hidden_width": self.hidden_width,
                "activation_type": self.activation_type,
                "w0": self.w0,
                "w0_first": self.w0_first,
                "w0_hidden": self.w0_hidden,
                "norm": self.norm,
                "drop_path_max": self.drop_path_max,
                "residual_alpha_init": self.residual_alpha_init,
                "random_state": self.random_state,
            },
            "cfg": {
                "embedding_dim": int(self.cfg.embedding_dim),
                "extras_dim": int(self.cfg.extras_dim),
                "episode_length": int(self.cfg.episode_length),
                "batch_episodes": int(self.cfg.batch_episodes),
                "random_state": self.cfg.random_state,
                "ppx_every": int(getattr(self.cfg, "ppx_every", 0)),
                "ppx_temperature": float(getattr(self.cfg, "ppx_temperature", 1.0)),
                "curriculum_type": getattr(self.cfg, "curriculum_type", None),
                "curriculum_warmup_epochs": int(getattr(self.cfg, "curriculum_warmup_epochs", 10)),
                "curriculum_min_frac": float(getattr(self.cfg, "curriculum_min_frac", 0.1)),
                "curriculum_max_frac": float(getattr(self.cfg, "curriculum_max_frac", 1.0)),
            },
            "tokenizer": tok_meta,
            "embedder": emb_meta,
            "embedder_state": emb_state,
            "model_state": self.model_.state_dict(),
            "regressor": {"params": reg_params},
            "extras_cache": self._extras_cache,
            "meta": {"version": 2},
        }
        torch.save(payload, path)

    @classmethod
    def load(cls, path: str, map_location: Optional[str | torch.device] = None) -> "PSANNLanguageModel":
        payload = torch.load(path, map_location=map_location or "cpu")
        params = payload.get("params", {})
        cfg_d = payload.get("cfg", {})
        tok_meta = payload.get("tokenizer", {})
        emb_meta = payload.get("embedder", {})
        emb_state = payload.get("embedder_state", {})
        model_state = payload.get("model_state", {})
        reg_meta = payload.get("regressor", {})
        extras_cache = payload.get("extras_cache", None)
        version = int(payload.get("meta", {}).get("version", 1))

        tok_type = tok_meta.get("type", "")
        if tok_type == "SimpleWordTokenizer":
            tok = SimpleWordTokenizer(
                lowercase=bool(tok_meta.get("lowercase", True)),
                max_vocab=tok_meta.get("max_vocab", None),
            )
            id2tok = list(tok_meta.get("id2tok", []))
            if id2tok:
                tok._id2tok = id2tok  # type: ignore[attr-defined]
                tok._tok2id = {w: i for i, w in enumerate(id2tok)}  # type: ignore[attr-defined]
        else:
            tok = SimpleWordTokenizer()

        emb_type = emb_meta.get("type", "")
        if emb_type == "SineTokenEmbedder":
            emb = SineTokenEmbedder(
                int(emb_meta.get("embedding_dim", cfg_d.get("embedding_dim", 64))),
                base=float(emb_meta.get("base", 10000.0)),
                scale=float(emb_meta.get("scale", 1.0)),
                trainable=bool(emb_meta.get("trainable", False)),
            )
            emb.load_state_dict(emb_state)
        else:
            emb = SineTokenEmbedder(int(cfg_d.get("embedding_dim", 64)))

        cfg = LMConfig(
            embedding_dim=int(cfg_d.get("embedding_dim", 64)),
            extras_dim=int(cfg_d.get("extras_dim", 0)),
            episode_length=int(cfg_d.get("episode_length", 64)),
            batch_episodes=int(cfg_d.get("batch_episodes", 32)),
            random_state=cfg_d.get("random_state", None),
            ppx_every=int(cfg_d.get("ppx_every", 0)),
            ppx_temperature=float(cfg_d.get("ppx_temperature", 1.0)),
            curriculum_type=cfg_d.get("curriculum_type", None),
            curriculum_warmup_epochs=int(cfg_d.get("curriculum_warmup_epochs", 10)),
            curriculum_min_frac=float(cfg_d.get("curriculum_min_frac", 0.1)),
            curriculum_max_frac=float(cfg_d.get("curriculum_max_frac", 1.0)),
        )

        hidden_units = params.get("hidden_units")
        if hidden_units is None:
            hidden_units = params.get("hidden_width")
        hidden_units = int(hidden_units if hidden_units is not None else 128)
        hidden_width = int(params.get("hidden_width", hidden_units))
        obj = cls(
            tokenizer=tok,
            embedder=emb,
            lm_cfg=cfg,
            hidden_layers=int(params.get("hidden_layers", 8)),
            hidden_units=hidden_units,
            hidden_width=hidden_width,
            activation_type=str(params.get("activation_type", "psann")),
            w0=float(params.get("w0", 30.0)),
            device=map_location or "cpu",
            random_state=params.get("random_state", None),
            w0_first=float(params.get("w0_first", 12.0)),
            w0_hidden=float(params.get("w0_hidden", 1.0)),
            norm=str(params.get("norm", "rms")),
            drop_path_max=float(params.get("drop_path_max", 0.0)),
            residual_alpha_init=float(params.get("residual_alpha_init", 0.0)),
        )

        vocab_size = tok.vocab_size
        obj._ensure_embedder(vocab_size)
        if version >= 2:
            obj.model_ = obj._create_model_module()
        else:
            D = int(obj.cfg.embedding_dim)
            K = int(obj.cfg.extras_dim)
            core = PSANNNet(
                D + K,
                D + K,
                hidden_layers=obj.hidden_layers,
                hidden_width=obj.hidden_width,
                act_kw={},
                state_cfg=None,
                activation_type=obj.activation_type,
                w0=obj.w0,
            )
            obj.model_ = WithPreprocessor(None, core).to(obj.device)
        assert obj.model_ is not None
        obj.model_.load_state_dict(model_state)
        obj.model_.to(obj.device)
        obj.regressor_params_ = reg_meta.get("params", None)
        if extras_cache is not None:
            obj._extras_cache = np.asarray(extras_cache, dtype=np.float32)
        else:
            obj._extras_cache = None
        return obj
