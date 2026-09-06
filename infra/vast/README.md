# vast.ai runbook

Rent a GPU box on vast.ai, push the code, run a training config, sync
checkpoints to object storage, tear down.

Everything runs from the devenv shell (`vastai` + `rclone` on PATH, `.env`
sourced). `vastai` reads `VAST_API_KEY` from the environment directly.

## 1. Create the template (once, or after editing `onstart.sh`)

```
infra/vast/create-template.sh                      # gls-model-1x80gb
```

Idempotent - looks the template up by name and updates in place. Bakes in the
disk size, the non-secret env, the onstart script, and a `--search_params`
filter for the shape. No secret is stored in a template (remote credentials reach the box only via
`.env` in step 3); never pass `--public`.

It prints the matching `vastai search offers` line and a verify command.

## 2. Rent an instance

```
vastai search offers '<the line create-template.sh printed>' -o 'dph_total+'
vastai create instance <offer-id> --template_hash <hash> --disk 120
vastai show instances                              # wait for actual_status = running
```

`--template_hash` pulls image/env/onstart; `--disk` must still be repeated here.

## 3. Push code + data

```
infra/vast/push.sh <instance-id>
```

Waits for the onstart sentinel (bounded by `WAIT_SECS`; an ssh auth failure is
fatal, not "still provisioning"), rsyncs the git-tracked tree and `.env`
(chmod 600) over SSH, then on the box: `rclone copy` the data pack from
`$GLS_REMOTE`, `uv sync`, and `gls env | tee runs/rented-env.txt` (paste into
`hardware.md`).

`--no-data` skips the pack; `--no-sync` skips `uv sync`.

## 4. Launch under tmux

```
ssh -i ~/.ssh/vast_ed25519 -o IdentitiesOnly=yes $(vastai ssh-url <instance-id> | sed -E 's#ssh://([^@]+)@([^:]+):([0-9]+)#-p \3 \1@\2#')
tmux new -s train
uv run gls train --config configs/medium-fineweb.toml --resume auto --wandb
#   Ctrl-b d to detach; tmux attach -t train to return
```

`--resume auto` is a no-op on a fresh run and picks up `latest.json` on
re-launch, so the recovery command is identical to the launch command. On
preemption the box gets SIGTERM: the loop checkpoints, fires `sync_cmd`, waits
for it, exits clean. Bring up a new instance, repeat steps 2–4, re-run the same
line - pull `latest.json` back from object storage first if the disk is empty.

`patience = 0` in `medium-fineweb.toml` - the run does not self-limit. Watch
`eval/val_loss`, `train/eta_s`, `train/peak_mem_gib`, and `train/tokens_per_sec`
(phase-1.5's single-GPU baseline) in W&B. `vastai stop instance` (SIGTERM) or
Ctrl-C is the clean manual stop.

## 5. Checkpoints -> object storage

`sync_cmd` in the config runs after every save:

```
rclone copy --transfers 8 --exclude '*.tmp/**' {run} $GLS_REMOTE:$GLS_BUCKET/gls-runs/$(basename {run})
```

`$GLS_REMOTE`/`$GLS_BUCKET` expand on the box from `/etc/profile.d/gls.sh`
(written by `onstart.sh`; default `b2` / `GLS_B2_BUCKET`). `{run}` is
`runs/<run_name>`, so the destination is keyed by `run_name`
(`gls-runs/medium-fineweb/…`) and the same line drops into any config
unchanged. The `RCLONE_CONFIG_*` remote credentials and the bucket
come from `.env` - no `rclone.conf` - and the same bucket holds the 
data pack under `gls-data/`.

It syncs the whole run tree, so the destination is a mirror of the local run
dir - `gls-runs/<name>/checkpoints/step-NNNNNN/`, one level deeper than the old
`{ckpt}`-only line put it, plus the pointer files, `train_config.json`,
`log.jsonl` and `env.json`. `rclone copy` transfers only what is missing or
changed at the destination and never deletes, so each save still moves ~one
checkpoint and the remote keeps rotated step dirs. A run synced under the old
flat layout is untouched but will not match.

`medium` checkpoint ≈ 3.6 GB; `ckpt_interval = 1000` over 20000 steps ≈ 20 saves
≈ ~70 GB through the hook. The link must move 3.6 GB inside one
`ckpt_interval` or `_sync` skips that save during the loop - stderr warning only
(`sync_cmd from the previous checkpoint still running`). The sync at exit is
different: it waits out any in-flight upload, then runs once more and blocks
until it lands, so `done`/`sigterm` and the final pointers always reach the
bucket. Set the remote credentials + bucket before step 0.

## 6. Verify the run landed before teardown

The per-save sync now carries everything needed to resume or score the run off
the box - these all ride along in the run tree and used to need a manual copy:

| | why it matters |
| --- | --- |
| `checkpoints/latest.json`, `checkpoints/best.json` | without them `--ckpt runs/<name>` fails: `resolve_init` finds no `model.safetensors` and no pointer. |
| `train_config.json` | `gls eval ppl` reads seed / `val_fraction` / `block_size` from it. Missing, it silently falls back to seed 1337 and `DEFAULT_VAL_FRACTION` and carves a different holdout. |
| `log.jsonl` | the metric history, including the `train/tokens_per_sec` series phase-1.5 wants. |

Confirm the bucket matches the box before destroying anything:

```
rclone check runs/ $GLS_REMOTE:$GLS_BUCKET/gls-runs/ --one-way
```

W&B holds the config and history too, so a run tracked with `--wandb` can be
reconstructed after the fact (`api.run(...).config` / `.scan_history()`). The
pointer files cannot - they exist only on the box until the sync lands.

Two B2 hazards worth knowing before you reorganise anything (B2-specific;
another backend has its own delete semantics):

- `rclone purge` removes all versions - it is a hard delete. `rclone delete`
  and `rclone move` only write a hide marker, and the bytes stay recoverable via
  `rclone lsf --b2-versions` / `rclone copyto --b2-versions`. Prefer copy, verify
  with `rclone md5sum` on both sides, and only then delete.
- Only for a hand-run copy: point it at a *prefix* rather than a step dir and the
  checkpoint's files land loose at the prefix root, where nothing will find them.
  The `sync_cmd` mirrors the tree, so the automatic path is not exposed to this.

## 7. Evaluate locally

```
rclone copy $GLS_REMOTE:$GLS_BUCKET/gls-runs/medium-fineweb/ runs/medium-fineweb/
gls eval ppl --ckpt runs/medium-fineweb --corpus dolly --val-fraction 0.05 --batch-size 8

gls eval ppl --ckpt runs/medium-dolly-sft --corpus dolly --val-fraction 0.05 --batch-size 8
gls eval harness --ckpt runs/medium-dolly-sft
```

`--val-fraction 0.05` on both sides: the pretrain config records `0.005` and
the SFT config `0.05`, so without the flag each checkpoint is scored on a
different holdout and the before/after is not a comparison. `gls eval ppl` prints
the resolved `seed / val_fraction / block_size` it used - check the two lines
match.

`--batch-size 8` because `gls.evaluate.perplexity` promotes the full logits
tensor with `.float()`: ~2 GB at batch 8, ~4 GB at 16, against an 11.6 GiB card.

Eval and inference run fine on the local card (`medium` is 0.57 GiB bf16), so
only the pretrain needs the rental. The SFT does not run locally at the
pretrain's batch shape - see the SFT sizing note in
[hardware.md](../../privatedocs/reference/hardware.md); it needs `batch_size = 4`
and `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`.

## Multi-GPU (DDP) - the seam, not yet the road

`src/gls` has no `torch.distributed` yet. What is already in place:

- `create-template.sh` - `GPU_COUNT=4 GPU_RAM=79 DISK=200 infra/vast/create-template.sh`
  makes a `gls-model-4x79gb` template whose offer filter is `num_gpus=4`.
- `src/gls/data.py` - `PackedData` takes `(rank, world_size)`, default `(0, 1)`.
- `src/gls/cli.py` - has the `__main__` guard, so `python -m gls.cli` works.

When distributed init lands, the only launch change is:

```
torchrun --standalone --nproc_per_node=$GPU_COUNT -m gls.cli train --config <cfg> --resume auto --wandb
```

Nothing in `infra/vast/` changes. `push.sh` and the `sync_cmd` are
rank-agnostic (rank 0 writes checkpoints).
