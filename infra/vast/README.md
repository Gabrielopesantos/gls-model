# vast.ai runbook

Rent a GPU box on vast.ai, push the code, run a training config, sync
checkpoints to B2, tear down.

The immediate use is the blocked deliverable - a pretrained `medium` base,
`configs/medium-fineweb.toml`, 20000 steps × 262144 tokens = 5.24B tokens on
fineweb-edu `sample-10BT`, batch 8 at ctx 2048 ≈ **~22 GiB** - but the scripts
are parameterised (see **Knobs**), so a different config, corpus, or GPU shape is
an env var, not an edit.

Everything runs from the devenv shell (`vastai` + `rclone` on PATH, `.env`
sourced). `vastai` reads `VAST_API_KEY` from the environment directly.

## Prerequisites

- **A dedicated SSH key**, not the GPG smartcard key. vast injects account keys
  into every box; `vastai ssh-url` hands back a **direct IP**, which no
  `~/.ssh/config` `Host *.vast.ai` block matches, so `ssh` would offer the
  smartcard key first and fail. `push.sh` pins `-i ~/.ssh/vast_ed25519
  -o IdentitiesOnly=yes`.

  ```
  ssh-keygen -t ed25519 -f ~/.ssh/vast_ed25519
  vastai create ssh-key "$(cat ~/.ssh/vast_ed25519.pub)"
  ```

  `~/.ssh/config` also has a `Host *.vast.ai` block pointing at the same key, for
  manual proxy connections.

- **The data pack on B2**. Prepared locally once and uploaded - no
  `gls data prepare`, no HF download on the box:

  ```
  rclone copy data/packed/fineweb-edu b2:$GLS_B2_BUCKET/gls-data/packed/fineweb-edu --transfers 8
  ```

## Knobs

`create-template.sh` (machine shape):

| var | default | |
| --- | --- | --- |
| `GPU_COUNT` | `1` | `4` searches a 4-GPU box (DDP - see the last section) |
| `GPU_RAM` | `79` | GB of VRAM per GPU |
| `DISK` | `120` | GB ephemeral disk |
| `VAST_IMAGE` | `vastai/base-image:cuda-12.9.2-auto` | slim base; uv brings torch+cu129 |
| `TEMPLATE_NAME` | `gls-model-${GPU_COUNT}x${GPU_RAM}gb` | |
| `EXTRA_SEARCH` | - | appended to the offer filter, e.g. `geolocation notin [CN]` |

`push.sh` (defaults reproduce the `medium-fineweb` run):

| var | default | |
| --- | --- | --- |
| `VAST_SSH_KEY` | `~/.ssh/vast_ed25519` | identity offered to the box |
| `CORPUS` | `fineweb-edu` | pulls `b2:$GLS_B2_BUCKET/gls-data/packed/<corpus>` |
| `UV_GROUPS` | `dev track` | dependency groups synced on the box |
| `CONFIG` | `configs/medium-fineweb.toml` | closing hint only |
| `WAIT_SECS` | `600` | cap on the provisioning wait |

## 1. Create the template (once, or after editing `onstart.sh`)

```
infra/vast/create-template.sh                      # gls-model-1x80gb
```

Idempotent - looks the template up by name and updates in place. Bakes in the
disk size, the non-secret env, the onstart script, and a `--search_params`
filter for the shape. **No secret is stored in a template** (`HF_TOKEN` /
`WANDB_API_KEY` / `RCLONE_CONFIG_B2_*` reach the box only via `.env` in step 3);
never pass `--public`.

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
(chmod 600) over SSH, then **on the box**: `rclone copy` the data pack from B2,
`uv sync`, and `gls env | tee runs/rented-env.txt` (paste into `hardware.md`).

`--no-data` skips the pack; `--no-sync` skips `uv sync`.

## 4. Launch under tmux

```
ssh -i ~/.ssh/vast_ed25519 -o IdentitiesOnly=yes $(vastai ssh-url <instance-id> | sed -E 's#ssh://([^@]+)@([^:]+):([0-9]+)#-p \3 \1@\2#')
tmux new -s train
uv run gls train --config configs/medium-fineweb.toml --resume auto --wandb
#   Ctrl-b d to detach; tmux attach -t train to return
```

`--resume auto` is a no-op on a fresh run and picks up `latest.json` on
re-launch, so **the recovery command is identical to the launch command**. On
preemption the box gets SIGTERM: the loop checkpoints, fires `sync_cmd`, waits
for it, exits clean. Bring up a new instance, repeat steps 2–4, re-run the same
line - pull `latest.json` back from B2 first if the disk is empty.

`patience = 0` in `medium-fineweb.toml` - the run does not self-limit. Watch
`eval/val_loss`, `train/eta_s`, `train/peak_mem_gib`, and `train/tokens_per_sec`
(phase-1.5's single-GPU baseline) in W&B. `vastai stop instance` (SIGTERM) or
Ctrl-C is the clean manual stop.

## 5. Checkpoints → Backblaze B2

`sync_cmd` in the config runs after every save:

```
rclone copy --transfers 8 {ckpt} b2:$GLS_B2_BUCKET/gls-runs/$(basename {run})/$(basename {ckpt})
```

`{run}` is `runs/<run_name>`, so the destination is keyed by `run_name`
(`gls-runs/medium-fineweb/…`) and the same line drops into any config
unchanged. `RCLONE_CONFIG_B2_*` and `GLS_B2_BUCKET`
(`santoslabs-training-checkpoints`) come from `.env` - no `rclone.conf` - and the
same bucket holds the data pack under `gls-data/`.

`medium` checkpoint ≈ 3.6 GB; `ckpt_interval = 1000` over 20000 steps ≈ 20 saves
≈ **~70 GB** through the hook. The link must move 3.6 GB inside one
`ckpt_interval` or `_sync` skips one - stderr warning only
(`sync_cmd from the previous checkpoint still running`). Set the B2 key + bucket
**before step 0**.

## 6. Pull back, evaluate locally, teardown

```
rclone copy b2:$GLS_B2_BUCKET/gls-runs/medium-fineweb/ runs/medium-fineweb/checkpoints/
gls eval ppl --ckpt runs/medium-fineweb --corpus dolly

gls train --config configs/medium-dolly-sft.toml --wandb        # phase-1b after
gls eval ppl --ckpt runs/medium-dolly-sft --corpus dolly
gls eval harness --ckpt runs/medium-dolly-sft

vastai destroy instance <instance-id>                           # billing is per-minute
```

Inference and eval run fine on the local card (`medium` is 0.57 GiB bf16) - only
the pretrain needs the rental.

## Multi-GPU (DDP) - the seam, not yet the road

`src/gls` has no `torch.distributed` yet (phase-1.5). What is already in place:

- `create-template.sh` - `GPU_COUNT=4 GPU_RAM=79 DISK=200 infra/vast/create-template.sh`
  makes a `gls-model-4x79gb` template whose offer filter is `num_gpus=4`.
- `src/gls/data.py` - `PackedData` takes `(rank, world_size)`, default `(0, 1)`.
- `src/gls/cli.py` - has the `__main__` guard, so `python -m gls.cli` works.

When distributed init lands, the only launch change (step 4) is:

```
torchrun --standalone --nproc_per_node=$GPU_COUNT -m gls.cli train --config <cfg> --resume auto --wandb
```

Nothing in `infra/vast/` changes. `push.sh` and the B2 `sync_cmd` are
rank-agnostic (rank 0 writes checkpoints).
