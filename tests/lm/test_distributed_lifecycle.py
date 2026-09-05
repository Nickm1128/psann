"""Distributed wrappers participate in actual optimizer, gather and resume paths."""

from dataclasses import replace
from datetime import timedelta
from pathlib import Path

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from psannlm import LMArchitectureConfig, LMConfig, LMTrainer, TrainConfig
from psannlm.architectures import build_lm_model, to_mapping
from psannlm.persistence import load_lm_checkpoint
from psannlm.lm.train.trainer import collate_batch


def _dataset():
    return [
        {"input_ids": (torch.arange(7) * 3 + i) % 29, "labels": (torch.arange(7) * 3 + i + 1) % 29}
        for i in range(32)
    ]


def _ddp_worker(rank, init, output, kind):
    torch.set_num_threads(1)
    dist.init_process_group(
        "gloo", init_method=init, rank=rank, world_size=2, timeout=timedelta(seconds=45)
    )
    try:
        torch.manual_seed(137)
        architecture = (
            LMArchitectureConfig.wave(temporal={"mode": "attention-only"})
            if kind == "attention-only"
            else kind
        )
        config = LMConfig(
            architecture=architecture, d_model=24, n_layers=2, n_heads=3, d_mlp=36, vocab_size=29
        )
        model = build_lm_model(config).model
        before = model.lm_head.weight.detach().clone()
        cfg = TrainConfig(
            steps_per_epoch=3,
            batch_tokens=14,
            lr=0.003,
            warmup_steps=0,
            amp="fp32",
            ddp="on",
            grad_checkpoint=True,
            checkpoint_dir=output,
            dataloader_num_workers=0,
            eval_interval_steps=1,
            eval_max_batches=1,
            save_interval_steps=2,
        )
        trainer = LMTrainer(cfg)
        trainer.train(model, _dataset(), max_length=7, device="cpu", val_dataset=_dataset()[:2])
        assert not torch.equal(before, model.lm_head.weight)
        gathered = [torch.empty_like(model.lm_head.weight) for _ in range(2)]
        dist.all_gather(gathered, model.lm_head.weight)
        torch.testing.assert_close(gathered[0], gathered[1], rtol=0, atol=0)
        dist.barrier()
        loaded = load_lm_checkpoint(Path(output) / "final.pt")
        assert loaded.payload["state"]["step"] == 3
        torch.testing.assert_close(
            loaded.model.lm_head.weight, model.lm_head.weight, rtol=0, atol=0
        )
        before_resume = loaded.model.lm_head.weight.detach().clone()
        resumed = LMTrainer(replace(cfg, steps_per_epoch=6))
        resumed.train(
            loaded.model,
            _dataset(),
            max_length=7,
            device="cpu",
            resume_checkpoint=str(Path(output) / "final.pt"),
        )
        assert not torch.equal(before_resume, loaded.model.lm_head.weight)
        dist.all_gather(gathered, loaded.model.lm_head.weight)
        torch.testing.assert_close(gathered[0], gathered[1], rtol=0, atol=0)
        dist.barrier()
        best = load_lm_checkpoint(Path(output) / "best.pt")
        assert best.payload["state"]["step"] == 2
        batch = collate_batch(_dataset()[:2])
        loss = torch.nn.functional.cross_entropy(
            best.model(batch["input_ids"]).flatten(0, 1), batch["labels"].flatten()
        ).item()
        if rank == 0:
            assert abs(trainer.best_val_loss - loss) < 1e-6
        final = torch.load(Path(output) / "final.pt", weights_only=True)
        assert final["state"]["step"] == 6 and final["config"] == to_mapping(config)
        assert all(v["step"].item() == 6 for v in final["optim"]["state"].values())
        assert not any(k.startswith("module.") for k in final["model"])
    finally:
        dist.destroy_process_group()


@pytest.mark.slow
@pytest.mark.parametrize("kind", ["residual", "attention-only", "geometric-sparse"])
def test_two_rank_cpu_ddp_optimizer_sync_checkpoint_and_resume(tmp_path, kind):
    rendezvous = (tmp_path / "rendezvous").as_uri()
    mp.spawn(
        _ddp_worker, args=(rendezvous, str(tmp_path / "checkpoint"), kind), nprocs=2, join=True
    )


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("use_orig_params", [True, False])
def test_cuda_fsdp_full_state_optimizer_and_resume(tmp_path, use_orig_params):
    # A single physical GPU exercises FSDP's documented world-size-one NO_SHARD path.
    dist.init_process_group(
        "gloo",
        init_method=(tmp_path / "rendezvous").as_uri(),
        rank=0,
        world_size=1,
        timeout=timedelta(seconds=45),
    )
    try:
        config = LMConfig(
            architecture="residual", d_model=24, n_layers=2, n_heads=3, d_mlp=36, vocab_size=29
        )
        torch.manual_seed(173)
        model = build_lm_model(config).model
        original = model.lm_head.weight.detach().clone()
        cfg = TrainConfig(
            steps_per_epoch=3,
            batch_tokens=14,
            lr=0.003,
            warmup_steps=0,
            amp="fp16",
            ddp="off",
            fsdp="full_shard",
            fsdp_auto_wrap_policy="none",
            fsdp_use_orig_params=use_orig_params,
            grad_checkpoint=True,
            checkpoint_dir=str(tmp_path),
            dataloader_num_workers=0,
        )
        trainer = LMTrainer(cfg)
        torch.manual_seed(919)
        trainer.train(
            model,
            _dataset(),
            max_length=7,
            device="cuda",
            data_loader=torch.utils.data.DataLoader(
                _dataset(), batch_size=2, shuffle=False, collate_fn=collate_batch
            ),
        )
        loaded = load_lm_checkpoint(tmp_path / "final.pt")
        first = loaded.payload
        assert first["state"]["step"] == 3 and first["scaler"]["_growth_tracker"] == 3
        assert not torch.equal(original, loaded.model.lm_head.weight)
        assert first["optim"]["state"] and all(isinstance(k, str) for k in first["optim"]["state"])
        resumed = LMTrainer(replace(cfg, steps_per_epoch=6))
        resumed.train(
            loaded.model,
            _dataset(),
            max_length=7,
            device="cuda",
            resume_checkpoint=str(tmp_path / "final.pt"),
        )
        final = load_lm_checkpoint(tmp_path / "final.pt")
        assert final.payload["state"]["step"] == 6
        assert final.payload["scaler"]["_growth_tracker"] == 6
        assert not torch.equal(
            first["model"]["lm_head.weight"], final.payload["model"]["lm_head.weight"]
        )
        assert all(v["step"].item() == 6 for v in final.payload["optim"]["state"].values())
        tokens = _dataset()[0]["input_ids"][None]
        torch.manual_seed(173)
        plain = build_lm_model(config).model
        torch.manual_seed(919)
        LMTrainer(replace(cfg, fsdp="off", checkpoint_dir=str(tmp_path / "plain"))).train(
            plain,
            _dataset(),
            max_length=7,
            device="cuda",
            data_loader=torch.utils.data.DataLoader(
                _dataset(), batch_size=2, shuffle=False, collate_fn=collate_batch
            ),
        )
        plain.cpu().eval()
        loaded_first = build_lm_model(config).model
        loaded_first.load_state_dict(first["model"])
        loaded_first.eval()
        torch.testing.assert_close(plain(tokens), loaded_first(tokens), rtol=0.001, atol=0.001)
        for key, value in plain.state_dict().items():
            torch.testing.assert_close(value, first["model"][key], rtol=0.001, atol=0.001)
    finally:
        dist.destroy_process_group()
