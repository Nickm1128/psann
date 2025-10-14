"""Demo: Fit SineTokenEmbedder and PSANN-LM, then predict and generate.

This is a minimal demonstration using the new PSANNLanguageModel pipeline.
"""

from pathlib import Path
import sys as _sys
try:  # noqa: F401
    import psann  # type: ignore
except Exception:  # pragma: no cover
    _sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
    import psann  # type: ignore

from psann import PSANNLanguageModel, LMConfig, SimpleWordTokenizer, SineTokenEmbedder


if __name__ == "__main__":
    # Tiny corpus
    corpus = [
        "the quick brown fox jumps over the lazy dog",
        "the quick blue hare sprints under the bright sun",
        "curious cats nap over warm rugs",
        "dogs bark and foxes dash swiftly",
    ]

    # Build tokenizer and embedder
    tok = SimpleWordTokenizer(lowercase=True)
    emb = SineTokenEmbedder(embedding_dim=32, trainable=False)

    # Configure LM (no extras to keep it simple)
    cfg = LMConfig(embedding_dim=32, extras_dim=4, episode_length=16, batch_episodes=16, random_state=0)
    lm = PSANNLanguageModel(tokenizer=tok, embedder=emb, lm_cfg=cfg, hidden_layers=8, hidden_width=64, activation_type="psann")

    # Fit on corpus
    lm.fit(
        corpus,
        epochs=50,
        lr=1e-3,
        verbose=1,
        ppx_every=5,
        curriculum_type="progressive_span",
        curriculum_warmup_epochs=10,
        curriculum_min_frac=0.2,
        curriculum_max_frac=1.0,
    )

    # Predict next token
    prompt = "the quick"
    next_tok = lm.predict(prompt)
    print("Prompt:", prompt)
    print("Predicted next token:", next_tok)

    # Generate continuation
    gen = lm.generate("the", max_tokens=10)
    print("Generated:", gen)
