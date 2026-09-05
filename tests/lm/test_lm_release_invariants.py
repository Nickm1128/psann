"""Real CLI output and both LM artifact writers stay release-aligned."""

import importlib.metadata

import pytest
import torch

from psannlm.__main__ import main
from psannlm import persistence


@pytest.mark.parametrize("streaming", [False, True], ids=["local", "exhausted-stream"])
def test_cli_budget_and_exhaustion_text_after_real_training_and_versioned_resaves(
    streaming,
    tmp_path,
    monkeypatch,
    capsys,
):
    from psannlm._train import cli

    text = tmp_path / "text.txt"
    text.write_text("hello world networks learn words " * 16, encoding="utf-8")
    manifest = tmp_path / "manifest.txt"
    manifest.write_text(str(text), encoding="utf-8")
    before = {}
    factory = cli.build_lm_model

    def build(config):
        result = factory(config)
        before.update({key: value.clone() for key, value in result.model.state_dict().items()})
        return result

    monkeypatch.setattr(cli, "build_lm_model", build)
    if streaming:

        def stream(**kwargs):
            assert kwargs["seq_len"] == 2
            yield {"input_ids": torch.tensor([1, 2]), "labels": torch.tensor([2, 3])}

        monkeypatch.setattr(cli, "streamed_token_iterator", stream)

    args = [
        "train",
        "--data-manifest",
        str(manifest),
        "--tokenizer-backend",
        "simple",
        "--architecture",
        "transformer",
        "--d-model",
        "8",
        "--n-layers",
        "1",
        "--n-heads",
        "2",
        "--max-length",
        "2",
        "--batch-tokens",
        "2",
        "--max-steps",
        "3",
        "--warmup-steps",
        "0",
        "--amp",
        "fp32",
        "--ddp",
        "off",
        "--device",
        "cpu",
        "--num-workers",
        "0",
        "--checkpoint-dir",
        str(tmp_path / "checkpoint"),
    ]
    if streaming:
        args[1:3] = ["--hf-dataset", "local-test-stream"]
        args.extend(["--dataset-streaming", "true"])
    assert main(args) == (2 if streaming else 0)
    output = capsys.readouterr().out
    assert "tokens_per_step_global~2 target_tokens~n/a max_steps=3" in output
    assert output.isascii(), output
    if streaming:
        assert "step=1 < max_steps=3; trained_tokens~2 vs target_tokens~n/a" in output
    else:
        assert "Streaming dataset exhausted early" not in output
    path = tmp_path / "checkpoint/final.pt"
    payload = torch.load(path, weights_only=True)
    assert payload["package_version"] == "0.13.0"
    assert (payload["schema"], payload["schema_version"]) == ("psannlm.trainer", 1)
    steps = 1 if streaming else 3
    assert payload["state"]["step"] == steps
    assert all(int(state["step"]) == steps for state in payload["optim"]["state"].values())
    assert not torch.equal(payload["model"]["lm_head.weight"], before["lm_head.weight"])
    loaded = persistence.load_lm_checkpoint(path)
    loaded.model.eval()
    ids = torch.tensor([[1, 2]])
    expected = loaded.model(ids).detach()

    # Exercise the source fallback while emitting real model checkpoints too.
    def missing(_):
        raise importlib.metadata.PackageNotFoundError("psannlm")

    monkeypatch.setattr(persistence, "version", missing)
    for generation in (1, 2):
        path = tmp_path / f"model-{generation}.pt"
        torch.save(persistence.model_payload(loaded.model, loaded.tokenizer), path)
        payload = torch.load(path, weights_only=True)
        assert payload["package_version"] == "0.13.0"
        assert (payload["schema"], payload["schema_version"]) == ("psannlm.model", 1)
        loaded = persistence.load_lm_checkpoint(path)
        loaded.model.eval()
        torch.testing.assert_close(loaded.model(ids), expected, rtol=0, atol=0)
        for key, value in loaded.model.state_dict().items():
            torch.testing.assert_close(value, payload["state_dict"][key], rtol=0, atol=0)
