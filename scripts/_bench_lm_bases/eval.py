# ruff: noqa: F403,F405
from __future__ import annotations

from .shared import *


def _eval_model(
    model: torch.nn.Module,
    dataset: HFTextStreamingLMDataset,
    *,
    max_tokens: int,
    max_batches: int,
    batch_tokens: int,
    seq_len: int,
    amp_mode: str,
) -> Dict[str, Any]:
    from torch.nn import functional as F
    from torch.utils.data import DataLoader

    device = next(model.parameters()).device
    batch_size = max(1, int(batch_tokens) // int(seq_len))

    use_amp = device.type == "cuda" and amp_mode in {"bf16", "fp16"}
    amp_dtype = torch.bfloat16 if amp_mode == "bf16" else torch.float16
    autocast_ctx = torch.amp.autocast("cuda", dtype=amp_dtype) if use_amp else torch.no_grad()

    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_tokens = 0
    batches = 0

    dl = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    with torch.no_grad():
        for batch in dl:
            input_ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)
            with autocast_ctx:
                logits = model(input_ids)
                B, T, V = logits.shape
                loss = F.cross_entropy(logits.view(B * T, V), labels.view(B * T), reduction="sum")
            preds = logits.argmax(dim=-1)
            total_correct += int((preds == labels).sum().item())
            total_loss += float(loss.detach().float().item())
            total_tokens += int(B * T)
            batches += 1
            if max_batches > 0 and batches >= max_batches:
                break
            if max_tokens > 0 and total_tokens >= max_tokens:
                break

    if total_tokens <= 0:
        return {
            "val_loss": float("nan"),
            "val_ppl": float("nan"),
            "val_top1_acc": float("nan"),
            "val_tokens": 0,
            "val_batches": batches,
        }

    val_loss = total_loss / float(total_tokens)
    val_ppl = math.exp(val_loss)
    val_acc = float(total_correct) / float(total_tokens)
    return {
        "val_loss": round(val_loss, 6),
        "val_ppl": round(val_ppl, 6),
        "val_top1_acc": round(val_acc, 6),
        "val_tokens": int(total_tokens),
        "val_batches": int(batches),
    }


def _run_lm_eval(
    run_dir: Path,
    *,
    ckpt_path: str,
    tokenizer_meta: Dict[str, Any],
    tasks: List[str],
    limit: int,
    num_fewshot: int,
    device: str,
) -> Dict[str, Any]:
    import subprocess

    tok_backend = str(tokenizer_meta.get("backend", "auto"))
    tok_model = tokenizer_meta.get("model_path")
    tok_special = tokenizer_meta.get("special_tokens_map_path")

    cmd = [
        sys.executable,
        "scripts/run_lm_eval_psann.py",
        "--ckpt",
        str(ckpt_path),
        "--tokenizer-backend",
        tok_backend,
        "--tasks",
        ",".join(tasks),
        "--limit",
        str(int(limit)),
        "--num-fewshot",
        str(int(num_fewshot)),
        "--device",
        device,
        "--output",
        str(run_dir / "lm_eval.json"),
    ]
    if tok_model:
        cmd.extend(["--tokenizer-model-path", str(tok_model)])
    if tok_special:
        cmd.extend(["--tokenizer-special-map-path", str(tok_special)])

    try:
        subprocess.run(cmd, check=True)
    except Exception as exc:
        return {"status": "error", "error": str(exc), "cmd": " ".join(cmd)}
    return {"status": "ok", "output": str(run_dir / "lm_eval.json"), "cmd": " ".join(cmd)}
