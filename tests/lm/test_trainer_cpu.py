from pathlib import Path

from psannlm.lm.config import TrainConfig
from psannlm.lm.data.dataset import LMDataset, PackingConfig
from psannlm.lm.data.tokenizer import Tokenizer, TokenizerConfig
from psannlm.lm.models.transformer_respsann import ResPSANNTransformer, ResPSANNTransformerConfig
from psannlm.lm.train.trainer import Trainer


def _tiny_dataset(max_length: int = 8):
    texts = [
        "the quick brown fox jumps over the lazy dog",
        "psann trainers run fine on cpu data chunks",
    ]
    tok = Tokenizer(TokenizerConfig(backend="simple"))
    tok.fit(texts)
    pack_cfg = PackingConfig(max_length=max_length, pack_sequences=True)
    dataset = LMDataset(texts, tok, pack_cfg)
    assert len(dataset) > 0, "dataset should yield at least one chunk"
    return dataset, pack_cfg, tok


def test_trainer_cpu_smoke(tmp_path):
    dataset, pack_cfg, tok = _tiny_dataset()
    model_cfg = ResPSANNTransformerConfig(
        vocab_size=tok.vocab_size,
        d_model=32,
        n_layers=1,
        n_heads=4,
        d_mlp=64,
        dropout=0.0,
        rope=False,
    )
    model = ResPSANNTransformer(model_cfg)
    ckpt_dir = Path(tmp_path) / "lm_ckpts"
    cfg = TrainConfig(
        epochs=1,
        batch_tokens=pack_cfg.max_length * 4,
        lr=5e-4,
        warmup_steps=1,
        weight_decay=0.0,
        amp="fp32",
        grad_clip=0.0,
        grad_accum_steps=1,
        checkpoint_dir=str(ckpt_dir),
        log_interval_steps=1,
        save_interval_steps=128,
        ddp="off",
    )
    trainer = Trainer(cfg)
    trainer.train(model, dataset, max_length=pack_cfg.max_length, val_dataset=dataset)
    final_ckpt = ckpt_dir / "final.pt"
    assert final_ckpt.exists()
    val_loss = trainer.validate(model, dataset)
    assert val_loss > 0.0
