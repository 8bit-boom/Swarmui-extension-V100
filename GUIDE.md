# Complete Setup Guide: SwarmUI on a V100 + Krea 2 (including GGUF)

This guide takes you from a bare machine with a Tesla V100 / Titan V to generating images with Krea 2 in SwarmUI, with all the V100-specific fixes applied. Follow the parts in order — each part assumes the previous one works.

---

## Part 0 — What you need

- A **Volta GPU**: Tesla V100 (16 or 32 GB), Titan V, or Quadro GV100. (Compute capability 7.0.)
- **OS**: Windows 10/11 or Linux. All commands below are given for both where they differ.
- **NVIDIA driver**: recent production branch (550+ on Linux, 552+ on Windows is fine).
- **Disk**: ~40 GB free for SwarmUI + ComfyUI + one Krea 2 variant (more if you keep several models or GGUF quants).
- **RAM**: 32 GB recommended (16 GB minimum; model files are memory-mapped during load).
- **CUDA Toolkit**: not required up front — SwarmUI's installer brings its own Python/venv. You only need the full CUDA toolkit if you later compile FlashAttention or GGUF-related native code.

> **Reality check on speed**: a V100 runs Krea 2 Turbo (12B DiT) at usable but not blazing speed. Expect roughly ~1–3 s/it at 1024px in fp8/fp16 depending on quant and attention backend. Turbo's 8 steps keep total times tolerable.

---

## Part 1 — Install SwarmUI

1. Install **git** and **.NET SDK 8** (Windows: winget install Microsoft.DotNet.SDK.8; Linux: your distro package manager or Microsoft's apt feed).
2. Clone SwarmUI and run its install script:
   ```bash
   git clone https://github.com/mcmonkeyprojects/SwarmUI
   cd SwarmUI
   # Windows: double-click install-windows.bat
   # Linux:
   chmod +x install-linux.sh && ./install-linux.sh
   ```
3. The installer sets up the SwarmUI server and offers to install a **ComfyUI backend** — accept that. It downloads Python via embedded uv and sets up a venv at `dlbackend/comfy/venv` (Linux) / `dlbackend\comfy\venv` (Windows).

> If you already have SwarmUI installed, skip to Part 2 — the V100 torch fix comes *before* first launch if possible, or right after.

---

## Part 2 — Fix PyTorch for the V100 (critical, do this early)

Modern PyTorch **cu130 builds dropped sm_70**. With a cu130 torch, SwarmUI starts fine, the backend reports "ready", and then every generation dies with `RuntimeError: no kernel image is available for execution on the device`. Fix:

> **nd-world on TrueNAS?** Your repo's `fix-swarmui-volta-torch.sh` already automates this part — run it and skip to Part 3. (Appendix A.2.)

1. Find the ComfyUI backend venv:
   - Linux: `SwarmUI/dlbackend/comfy/venv/bin/python`
   - Windows: `SwarmUI\dlbackend\comfy\venv\Scripts\python.exe`
2. Check the current torch and its supported architectures:
   ```bash
   venv/bin/python -c "import torch; print(torch.__version__); print(torch.cuda.get_arch_list())"
   ```
   If the arch list contains `sm_70`, you're fine — skip to Part 3. If it jumps from `sm_75`/`sm_80` straight past 7.0 (typical cu130: `sm_80 sm_90 sm_100 ...` only), continue.
3. Install the cu128 build (still ships sm_70 kernels):
   ```bash
   venv/bin/python -m pip install --upgrade torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
   ```
4. Re-check arch list — `sm_70` must now appear:
   ```bash
   venv/bin/python -c "import torch; print(torch.cuda.get_arch_list()); print(torch.cuda.get_device_name(0))"
   ```

> Do **not** let SwarmUI's backend updater "repair" this back to cu130 — if an update re-pins torch, re-run step 3.

---

## Part 3 — Install the V100Compat extension

1. Clone the extension into SwarmUI's extension folder:
   ```bash
   cd SwarmUI/src/Extensions
   git clone https://github.com/8bit-boom/Swarmui-extension-V100.git V100Compat
   ```
2. Rebuild and launch. Two options:
   - Run the **`update`** script in the SwarmUI root (rebuilds once, then launches), or
   - Use **`launch-dev`** (rebuilds on every launch — use this while developing/tweaking).
3. Verify in the log at startup — you should see:
   ```
   [Init] V100Compat: pre-Ampere GPU (compute capability 7.x, no bf16) detected...
   [Init] V100Compat extension loaded.
   ```
   If you only see the second line, GPU detection failed (nvidia-smi not on PATH) — the extension still works, but the patch defaults to OFF; enable it manually per Part 4.
4. In the UI: open the **Text2Image** tab → enable **Display Advanced Options** (user settings, top of page) → find the **V100 / Volta Compatibility** group.

---

## Part 4 — Configure the V100 dtype patch

1. Under **V100 / Volta Compatibility**, make sure **V100 Compatibility Patch** is on (it's on by default when a 7.x GPU was detected).
2. Keep **V100 Precision = fp16** (default) — this routes the loaded model through ComfyUI's `ModelComputeDtype` node before sampling, so no bf16 ops reach the GPU.
3. Generate a test image with any standard model (e.g. an SDXL checkpoint) before touching Krea 2. If you get black images or NaN warnings for a specific model, set **V100 Precision = fp32** for that model only (per-model presets work well for this).

**Why this exists**: without it, models that internally select bf16 (including Krea 2's bf16 variants) either crash with dtype errors or silently run in emulated fp32 at ~1/4 speed.

---

## Part 5 — Optional: FlashAttention on the V100

Skip this part if you just want things working — PyTorch cross-attention is correct, just memory-hungry at 2K resolutions. Come back here if you hit OOMs at high resolution or want more speed.

### Easy path (recommended): the ComfyUI node pack

1. In SwarmUI, find **ComfyUI Flash-Attention V100** in the installable features list (registered by this extension; Server area / alongside your ComfyUI backend) and install it.
2. Restart the backend when prompted.
3. In ComfyUI, wire the **FlashAttnV100 Controller** node between your model loader and sampler, **or** set the environment variable `FLASHATTN_V100_AUTO_PATCH=1` before launching SwarmUI so it patches automatically.
4. Keep the Part 4 dtype patch enabled — this pack fixes attention memory/speed, not bf16 compatibility.

### Power-user path: a real `flash_attn` library

If you'd rather back ComfyUI's `--use-flash-attention` flag with the best-tested unofficial Volta kernel:

1. Build [sirCamp/flash-attention-legacy](https://github.com/sirCamp/flash-attention-legacy) into the backend venv (needs CUDA toolkit):
   ```bash
   venv/bin/python -m pip install git+https://github.com/sirCamp/flash-attention-legacy.git
   ```
2. Create the shim `flash_attn.py` in the venv's site-packages:
   ```python
   from flash_attn_legacy import *  # noqa: F401,F403
   ```
3. In SwarmUI: **Server → Backends → Configure** → your ComfyUI backend → add `--use-flash-attention` to the extra args (remove `--use-pytorch-cross-attention` if present).
4. Compare a few outputs against `--use-pytorch-cross-attention` before trusting it — see the comparison table in the README for the other kernel ports (ai-bond, ZRayZzz/Coloured-glaze) and their caveats.

---

## Part 6 — Krea 2 in SwarmUI

Krea 2 is a 12B dense DiT with a Qwen3-VL 4B text encoder and Qwen Image VAE. **SwarmUI supports it natively** — no custom model class needed. What follows is just file placement and parameters.

### 6a. Standard (safetensors) setup — recommended

1. Download from [Comfy-Org/Krea-2](https://huggingface.co/Comfy-Org/Krea-2) (Hugging Face):
   | File | Save to (inside the ComfyUI backend models dir) | Notes |
   |---|---|---|
   | `krea2_turbo_fp8_scaled.safetensors` | `diffusion_models/` | Recommended for most users |
   | `qwen3vl_4b_fp8_scaled.safetensors` | `text_encoders/` | Auto-detected as the Krea 2 CLIP |
   | `qwen_image_vae.safetensors` | `vae/` | |
   Optional: `krea2_raw_*` (base model for LoRA training / experiments) and style LoRAs (`krea2_softwatercolor` etc.) → `loras/`.
2. In SwarmUI: **Server → Backends → Restart** the ComfyUI backend, then click **Refresh** on the model list.
3. Select `krea2_turbo_fp8_scaled` as the model.
4. Parameters:
   - **Steps: 8** (Turbo; 4 is the minimum, quality drops). Use 20+ only with the Raw model.
   - **CFG: 1** for Turbo (distilled). Raw uses normal CFG (4+).
   - **Sigma Shift: 1.15** (the default is correct).
   - **Resolution**: 1024 default; Krea 2 handles 128–4096, and 2K one-pass works if VRAM allows. On a 16 GB V100, stay at 1024–1536 with fp8 + fp16 patch; 32 GB can go 2K.
   - **Prompting**: long, detailed natural-language prompts work best; NSFW terms get stripped by the model's built-in refiner (use the documented bypass LoRAs if that's a problem for you).
5. Generate. With the Part 4 patch on, the fp8 model runs with fp16 compute — the correct V100 path.

### 6b. GGUF setup (optional, memory-tight or experimental)

GGUF trades speed for lower VRAM. Krea 2's GGUF files need a loader that isn't in ComfyUI core — and **not the upstream city96/ComfyUI-GGUF either**: it doesn't parse Krea 2's ops yet (city96#464), so generation fails against it. The working options:

- **ComfyUI-GGUF_KREA-2** ([RealRebelAI](https://github.com/RealRebelAI/ComfyUI-GGUF_KREA-2)) — a drop-in fork of city96's pack with Krea 2 support, including the Qwen3-VL GGUF text encoder. **Registered as an installable feature by this extension.** It shares node/class names with the upstream pack and **cannot coexist with it** — if SwarmUI previously auto-installed city96/ComfyUI-GGUF, remove that folder from `custom_nodes` before installing the fork.
- **nd-world users**: your repo already has this covered — `install-comfyui-gguf-krea2.sh` installs the same fork (removing a conflicting upstream install automatically), and the Krea 2 GGUF Quick Setup panel on the Image Gen tab downloads the model files. Use that path; don't install the city96 pack or TJ_NODE on top of it.

Steps for everyone else:

1. In SwarmUI's installable features list, install **ComfyUI-GGUF KREA-2 (fork — replaces built-in 'gguf')** (registered by this extension). Note SwarmUI's built-in `gguf` (city96) entry will also appear in that list — an extension can't hide someone else's entry — so pick the fork entry, never the built-in one. If city96's pack was previously installed, delete its folder from the ComfyUI `custom_nodes` directory first.
2. Restart the backend.
3. Download the GGUF files (community quants for Krea 2 are discussed in [city96/ComfyUI-GGUF issue #464](https://github.com/city96/ComfyUI-GGUF/issues/464)):
   - A Krea 2 (Raw or Turbo) GGUF diffusion model → `diffusion_models/`
   - Qwen3-VL 4B GGUF + its matching `.mmproj` file → `text_encoders/` (both files, same folder)
   - The VAE stays the normal `qwen_image_vae.safetensors` in `vae/`
   For a 16 GB V100, start with Q6_K/Q8_0; Q4_K_M or lower if you OOM. Community reports run 2-bit quants on 4 GB cards — quality suffers accordingly.
4. Workflow: in SwarmUI select the GGUF model file from the model list (GGUF files in `diffusion_models` are listed like any model). The KREA-2 fork registers the loader nodes that understand Krea 2's ops, so the standard generation path works. If the text encoder isn't picked up automatically, open the ComfyUI workflow editor from SwarmUI and wire the fork's GGUF CLIP loader node for the text encoder manually.
5. Expect slower sampling than fp8 safetensors — that's inherent to GGUF dequant. Keep the dtype patch on; combine with the Part 5 attention pack if VRAM is tight.

> Note: SwarmUI also supports other Krea 2 quants natively (nvfp4 for tight memory, int8, bf16 for research). GGUF is only worth it if you specifically need its compression level.

---

## Part 7 — Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| `no kernel image is available for execution on the device` | cu130 PyTorch (no sm_70 kernels) | Part 2 — install cu128 torch into the backend venv |
| `RuntimeError: expected scalar type Half but found BFloat16` | bf16 tensor reached the V100 | Enable **V100 Compatibility Patch** (Part 4) |
| Generation ~4x slower than expected, no error | Silent bf16→fp32 emulation | Same as above |
| Black images / NaN warnings with a specific model | fp16 overflow on that model | Set **V100 Precision = fp32** for that model |
| `CUDA out of memory` at 2K | Attention matrix too large | Use 1024–1536, or install the Part 5 attention pack |
| `Could not detect model type` / model missing from list | Files in wrong folder or backend not restarted | Check Part 6 folder table; restart backend; Refresh |
| Krea 2 GGUF fails to generate / TE fails with unrecognized architecture | Upstream city96 pack doesn't parse Krea 2 ops | Use the ComfyUI-GGUF_KREA-2 fork (Part 6b); remove any city96 install first |
| FlashAttn nodes missing in ComfyUI | Pack not installed / backend not restarted | Reinstall via installable features; restart backend |
| Extension not visible in UI | Build didn't run | Run `update` or `launch-dev` (Part 3, step 2) and check startup log |

Logs to check when stuck:
- SwarmUI: `SwarmUI/Logs/` (latest `launch.log`)
- ComfyUI backend: `SwarmUI/dlbackend/comfy/comfyui.log`

---

## Part 8 — Updating & publishing

**Updating the extension**: `cd SwarmUI/src/Extensions/V100Compat && git pull`, then run the `update` script (rebuild required — C# is compiled).

**Updating SwarmUI/ComfyUI**: run the `update` script normally, then re-verify Part 2's torch arch list — backend updates can re-pin cu130 torch.

**Sharing with others**: after local testing, PR your repo into SwarmUI's [`extension_list.fds`](https://github.com/mcmonkeyprojects/SwarmUI/blob/master/launchtools/extension_list.fds) so it appears in the in-app extension list. Keep the MIT license, and make sure the README clearly documents the external connections this extension triggers (installable features download from GitHub only when the user clicks install).

---

## Appendix A — nd-world on TrueNAS SCALE (your setup)

This appendix adapts the guide to SwarmUI deployed through **nd-world's** `truenas-compose.yml` (the `swarmui` Compose profile) on TrueNAS SCALE 24.10+ (Electric Eel, Docker-based apps). Ground truth from that compose file:

| Item | Actual value |
|---|---|
| Image | `ghcr.io/mcmonkeyprojects/swarmui:latest` (upstream SwarmUI image) |
| Container name | `ix-<your-app-name>-swarmui-1` on TrueNAS; `swarmui-1` on plain Compose |
| SwarmUI code | `/SwarmUI` (image layer — changes here are lost on image update) |
| Data volume | `/SwarmUI/Data` → `/mnt/DeadPool/apps/swarmui/data` |
| Models | `/SwarmUI/Models` → `/mnt/DeadPool/apps/swarmui/models` |
| ComfyUI backend | `/SwarmUI/dlbackend/ComfyUI` → `/mnt/DeadPool/apps/swarmui/dlbackend` (note the capital C — the backend *and its venv* persist on your dataset) |
| Backend venv python | `/SwarmUI/dlbackend/ComfyUI/venv/bin/python` |
| Watchtower | auto-update is deliberately disabled for the SwarmUI container (`com.centurylinklabs.watchtower.enable: "false"`) — an unpinned update can silently pull a CUDA-13 torch |
| Port | 7801 |

You already have two helper scripts in the nd-world repo that cover Parts 2 and 6b — use them instead of the generic commands below where applicable.

### A.1 GPU passthrough

Your compose file has the wiring commented in-place: assign the V100 to the **swarmui** service from the TrueNAS app's Resources/GPU screen (per-service picker when built through the wizard), or uncomment the `deploy.resources.reservations.devices` block for a pasted-YAML Custom App. If SwarmUI already generates on your V100, this is done. `docs/GPU_SETUP.md` in nd-world has the full walkthrough, including VRAM-sharing sizing if ollama and swarmui run on the same 16 GB card simultaneously.

If the container image lacks `nvidia-smi`, the extension's auto-detection defaults the patch to OFF — enable **V100 Compatibility Patch** manually (Part 4).

### A.2 Part 2 (torch fix): use your existing script

Your repo's `fix-swarmui-volta-torch.sh` (run from the TrueNAS shell) already does Part 2 end-to-end against the running container: it finds the `swarmui` container, checks `torch.cuda.get_arch_list()` for `sm_70`, and reinstalls torch/torchvision from the **cu126** index if missing (the script's comment notes cu126 as the safe CUDA-12.x choice for Volta; cu128 works too — both ship sm_70 kernels — but the script is what your setup is tested with, so prefer it):

```bash
./fix-swarmui-volta-torch.sh          # auto-finds the container
# or explicitly:
./fix-swarmui-volta-torch.sh ix-<your-app>-swarmui-1
```

It's idempotent — safe to re-run after any backend reinstall or custom-node install whose `requirements.txt` re-pulls a CUDA-13 torch (PyPI defaults to cu13 wheels now, which is exactly how this breaks). Because `dlbackend` is a bind-mounted dataset, the fixed venv survives container restarts and image updates.

### A.3 Part 3 (extension install) in your compose

The extension must survive image updates, so bind-mount it rather than cloning into the ephemeral image layer:

1. On the TrueNAS shell:
   ```bash
   mkdir -p /mnt/DeadPool/apps/swarmui/extensions
   git clone https://github.com/8bit-boom/Swarmui-extension-V100.git /mnt/DeadPool/apps/swarmui/extensions/V100Compat
   ```
2. Add a Host Path mount to the `swarmui` service in `truenas-compose.yml` (sibling of the existing three):
   ```yaml
   volumes:
     - /mnt/DeadPool/apps/swarmui/data:/SwarmUI/Data
     - /mnt/DeadPool/apps/swarmui/models:/SwarmUI/Models
     - /mnt/DeadPool/apps/swarmui/dlbackend:/SwarmUI/dlbackend
     - /mnt/DeadPool/apps/swarmui/extensions:/SwarmUI/src/Extensions   # <-- add this
   ```
   then redeploy the app (`docker compose -f truenas-compose.yml --profile swarmui up -d` on the shell, or your usual update path).
3. Watch the app logs at startup for:
   ```
   [Init] V100Compat: pre-Ampere GPU (compute capability 7.x, no bf16) detected...
   [Init] V100Compat extension loaded.
   ```
   The upstream SwarmUI image builds on start, so a restart is normally enough. If the extension doesn't appear in the UI, exec in and check:
   ```bash
   docker exec -it ix-<your-app>-swarmui-1 bash -c "cd /SwarmUI && bash update-linux.sh"
   ```

> Updating the extension later: `git -C /mnt/DeadPool/apps/swarmui/extensions/V100Compat pull` + restart the app.

### A.4 Parts 4–5

UI-driven — identical to the main guide. The FlashAttention node pack (Part 5) is registered by this extension as an installable feature and installs into the ComfyUI `custom_nodes` under `/SwarmUI/dlbackend/ComfyUI/custom_nodes` — which is on your persistent dataset, so it survives updates.

### A.5 Krea 2 GGUF: your setup already has a tested path

Per Part 6b, don't install the upstream city96 pack or TJ_NODE — your repo solved this with the **RealRebelAI/ComfyUI-GGUF_KREA-2** fork, and this extension now registers that same fork (not city96) as its installable feature. On nd-world:

1. Run `./install-comfyui-gguf-krea2.sh` (it clones the fork into the backend's `custom_nodes`, removes any conflicting city96 install, and installs its requirements via the container venv), **or** install **ComfyUI-GGUF KREA-2 (fork — replaces built-in 'gguf')** from SwarmUI's installable features list — same result, but the script is what your compose setup documents. If you use the UI list, ignore SwarmUI's separate built-in `gguf` (city96) entry: an extension can't hide it, and installing both breaks ComfyUI (duplicate node class names).
2. Use the **Krea 2 GGUF Quick Setup panel** on nd-world's Image Gen tab to download a BASE/TURBO diffusion GGUF + the Qwen3-VL-4B-Instruct text encoder into the shared models dataset (`/mnt/DeadPool/apps/swarmui/models`, mounted at `/SwarmUI/Models` for SwarmUI).
3. Restart SwarmUI, pick the 'Krea 2 Turbo' Image Gen template in nd-world, generate — and keep the Part 4 dtype patch on.

### A.6 TrueNAS-specific gotchas

- **Watchtower is off for SwarmUI by design** — updates are manual, which protects the torch fix. When you do update the image, re-run `fix-swarmui-volta-torch.sh` afterward (idempotent) and re-verify the arch list.
- **Everything lives on `/mnt/DeadPool/apps/swarmui/`** — back up that dataset and you back up models, backend venv (including the fixed torch), and the extension in one shot.
- **Log locations (host-readable, no `docker exec` needed)**: `/mnt/DeadPool/apps/swarmui/data/Logs/launch.log` and `/mnt/DeadPool/apps/swarmui/dlbackend/ComfyUI/comfyui.log`.
- **VRAM sharing**: with ollama also on the V100, mind `OLLAMA_MAX_LOADED_MODELS`/`OLLAMA_KEEP_ALIVE` and SwarmUI's concurrent-backend count — see `docs/GPU_SETUP.md` §3a for sizing guidance on a 16 GB card.
