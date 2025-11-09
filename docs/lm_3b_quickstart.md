PSANN‑LM 3B Quickstart

- One-command streaming training (FSDP-ready):

  `python scripts/train_psann_lm.py --base waveresnet --d-model 3072 --n-layers 30 --n-heads 24 --tokenizer-backend tokenizers --train-tokenizer --tokenizer-save-dir runs/tokenizer_3b --hf-dataset allenai/c4 --hf-name en --hf-split train --hf-text-key text --hf-keep-ascii-only --hf-lang en --batch-tokens 65536 --grad-accum-steps 8 --amp bf16 --grad-checkpoint --fsdp full_shard --epochs 1 --save-interval-steps 2000 --checkpoint-dir runs/lm/3b_en --export-dir artifacts/psannlm_3b_bundle`

- Deduplicate corpus (exact or MinHash):

  `python tools/dedupe.py --input shards.txt --output shards_unique.txt`

  `python tools/dedupe.py --inputs shard1.txt shard2.txt --minhash --threshold 0.9 > shards_dedup.txt`

- Evaluate with chat template on MC tasks:

  `python scripts/run_lm_eval_psann.py --hf-repo <user>/<repo> --hf-filename psannlm_chat_final.pt --tokenizer-backend tokenizers --hf-tokenizer-repo <user>/<repo> --hf-tokenizer-filename tokenizer_final/tokenizer.json --tasks hellaswag,piqa,winogrande --device cuda --num-fewshot 5 --apply-chat-template --fewshot-as-multiturn --output eval_out/mc_chat.json`

Notes

- For large-scale training, pass `--steps-per-epoch` if using an IterableDataset with no fixed length to improve LR scheduling.
- When using HF tokenizers backend, pass the `special_tokens_map.json` for exact special id parity.
- Add `--hf-keep-ascii-only` and repeatable `--hf-lang en` to enforce English-only rows. Language filtering requires `langdetect` (install via `pip install langdetect`).
- `--train-tokenizer` uses the same data source (HF or manifest) to train a BPE tokenizer before model training. Control samples with `--tokenizer-sample-limit` (default 200k docs) and persist assets via `--tokenizer-save-dir`.
- Use `--export-dir` to collect `model.pt`, `tokenizer.json`, `special_tokens_map.json`, and `psann_artifacts.json` in a single folder ready for `huggingface-cli upload`.

Manifest Creation (optional if using HF datasets)

- Build a manifest from directories (Python helper):
  - `python tools/build_manifest.py --roots /data/en_text --pattern "*.txt" --recurse --absolute --output /data/en_manifest.txt`

- Windows PowerShell (all `.txt` recursively):
  - `Get-ChildItem -Recurse -File -Filter *.txt C:\data\en | ForEach-Object { $_.FullName } | Set-Content C:\data\en_manifest.txt`

- Bash (Linux/macOS):
  - `find /data/en -type f -name '*.txt' | sort > /data/en_manifest.txt`

Tip: After dedup/decontam, you may consolidate to a single text file (one line per document) and use a 1-line manifest that points to that file.
