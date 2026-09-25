# SwarmUI V100 Compatibility Extension

Patches SwarmUI's ComfyUI workflows so they run correctly on **pre-Ampere NVIDIA cards (compute capability 7.x)** — Volta (Tesla V100, Titan V) and Turing (RTX 20xx, T4), which all lack bf16 compute.

## Why the V100 needs special handling

The V100 is still a capable card (16/32 GB HBM2, fast FP16 tensor cores), but it predates a few things modern pipelines assume:

1. **No bf16 compute.** Volta only supports FP16/FP32. bf16 tensors get *silently emulated via fp32*, which is ~4x slower, spikes VRAM, and some nodes just crash with errors like `RuntimeError: expected scalar type Half but found BFloat16`. Some newer models (FLUX-family, video models) whitelist only bf16/fp32, so on a V100 ComfyUI silently falls back to fp32 and crawls. (Turing/7.5 has the same limitation — bf16 only arrived with Ampere/sm_80.)
2. **cu130 PyTorch wheels dropped sm_70.** If your torch is a cu130 build, every CUDA op fails with `no kernel image is available for execution on the device`, even though `torch.cuda.is_available()` returns True. Use a **cu128 (or older)** build:
   ```bash
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
   ```
3. **Triton 3.3+ and FlashAttention-3 dropped Volta.** SageAttention, `torch.compile`, and FA3-dependent nodes will fail on a V100.

## What this extension does

- At startup, runs `nvidia-smi` to detect a pre-Ampere GPU (compute capability 7.x — Volta 7.0 or Turing 7.5; both lack bf16, which only arrived with Ampere/sm_80).
- Adds an **Advanced Options** group "V100 / Volta Compatibility" to the generation page:
  - **V100 Compatibility Patch** — on by default when a 7.x GPU is detected. Inserts ComfyUI's core `ModelComputeDtype` node immediately after base model loading (model-gen step priority `-50`, after the loader at `-100` but before sampling), so every downstream node (sampler, refiner, etc.) gets a bf16-free dtype.
  - **V100 Precision** — `fp16` (default, ~4x faster on Volta tensor cores) or `fp32` (safe fallback for the few models that NaN/overflow in fp16).
- Registers **ComfyUI Flash-Attention V100** (the experimental node pack) as an installable feature, so it can be installed from within SwarmUI after you install this extension — no manual `custom_nodes` cloning.
- Registers **ComfyUI-GGUF KREA-2** ([RealRebelAI fork](https://github.com/RealRebelAI/ComfyUI-GGUF_KREA-2)) as an installable feature for **Krea 2 GGUF** support — see the [detailed guide](GUIDE.md). Krea 2 itself (Raw/Turbo) is already natively supported by SwarmUI; GGUF just needs this loader fork. **Conflict warning**: the fork and SwarmUI's built-in `gguf` (city96) entry register identical node class names and must never be installed together. The built-in entry still appears in the installable-features list — an extension cannot hide or unregister someone else's entry, so this is enforced by naming/documentation, not by removing the option (the feature is named "… (fork — replaces built-in 'gguf')"). If both are ever installed, delete the city96 folder from ComfyUI's `custom_nodes` and restart.
- On non-7.x machines it stays completely inert unless you manually enable the toggle.

## Installation

1. Clone this repo into `src/Extensions/` of your SwarmUI install:
   ```bash
   cd SwarmUI/src/Extensions
   git clone https://github.com/8bit-boom/Swarmui-extension-V100.git V100Compat
   ```
2. Rebuild & relaunch: run the `update` script in the Swarm root, or use `launch-dev` (rebuilds every launch) while developing.
3. Open the UI → enable **Display Advanced Options** → find the **V100 / Volta Compatibility** group → confirm the patch is on.
4. If fp16 gives you black/NaN images on a specific model, set **V100 Precision** to `fp32` for that model.

## Experimental: FlashAttention-2 on the V100

The official FlashAttention wheels don't support Volta. Several unofficial projects fill that gap — **all are experimental, none have the audit coverage of the official package**, so verify outputs against `--use-pytorch-cross-attention` before trusting any of them. Silently wrong attention output is worse than slow attention.

| Option | What it is | Install difficulty |
|---|---|---|
| **ComfyUI_Flash-Attention_v100** ([NetVoobrazhenia](https://github.com/NetVoobrazhenia/ComfyUI_Flash-Attention_v100)) | ComfyUI custom-node pack wrapping ai-bond's kernel. Controller node between loader and sampler, Status/Config nodes, `force_fp16` + NaN sanitization, `FLASHATTN_V100_AUTO_PATCH=1` | **Easy** — registered as an installable feature by this extension (see below) |
| **flash-attention-legacy** ([sirCamp](https://github.com/sirCamp/flash-attention-legacy)) | The best-tested kernel port: FA2 CUDA for Pascal & Volta, GQA/MQA, varlen, real pytest suite, V100 benchmarks (84–96% attention memory savings), HF `attn_implementation="flash_attn_legacy"` | Manual pip build |
| **flash-attention-v100** ([ai-bond](https://github.com/ai-bond/flash-attention-v100)) | The original from-scratch FA2 port (self-education project, no tests). Same API as official `flash_attn` | Manual pip build |
| **flash-attention-v100** ([ZRayZzz](https://github.com/ZRayZzz/flash-attention-v100) / updated fork [Coloured-glaze](https://github.com/Coloured-glaze/flash-attention-v100)) | CUTLASS-based port. Caveats per its README: forward ~40% faster but backward ~20% slower than PyTorch (net wash), sequence lengths must be padded to multiples of 32, Linux only, needs a CUTLASS checkout | Manual pip build, most fragile |

### Recommended path: the ComfyUI node pack

This extension registers **ComfyUI Flash-Attention V100** as an installable feature in SwarmUI. After installing the extension and restarting:

1. Find it in SwarmUI's installable features list (Server area — it can be installed/updated alongside your ComfyUI backend) and install it.
2. Restart the backend. In ComfyUI, connect the **FlashAttnV100 Controller** node between your model loader and sampler (or set `FLASHATTN_V100_AUTO_PATCH=1` in the environment).
3. Keep this extension's dtype patch enabled regardless — FlashAttention fixes attention memory/speed, not bf16 compatibility. The two operate at different layers.

No `flash_attn` import shim and no `--use-flash-attention` backend arg needed for this path.

### Power-user path: flash-attention-legacy as a pip library

If you want the better-tested sirCamp kernel to back ComfyUI's `--use-flash-attention` flag:

1. Build & install it into your ComfyUI venv (CUDA toolkit + matching torch required):
   ```bash
   pip install git+https://github.com/sirCamp/flash-attention-legacy.git
   ```
   (For ai-bond or ZRayZzz instead, clone their repos and follow their build instructions — ZRayZzz additionally requires editing `setup.py` to point at a CUTLASS source checkout, and only supports Linux.)
2. ComfyUI's `--use-flash-attention` flag does a plain `import flash_attn`, but these ports install under different module names. Add a shim — create `flash_attn.py` in your venv's site-packages:
   ```python
   from flash_attn_legacy import *  # noqa: F401,F403  (or: from flash_attn_v100 import *)
   ```
3. In SwarmUI: **Server → Backends → Configure** your ComfyUI backend and add `--use-flash-attention` to the extra args (remove `--use-pytorch-cross-attention` if set).
4. Keep this extension's dtype patch enabled regardless (same reason as above).

## Notes & manual steps the extension can't do for you

- **Keep your ComfyUI's PyTorch on a cu128 or older build** (see above). cu130+ has no sm_70 kernels at all, and no workflow patch can fix that.
- Avoid ComfyUI launch args that request Volta-unsupported features, e.g. SageAttention / `torch.compile` won't work on a V100; `--use-pytorch-cross-attention` is the safe default attention backend (with the experimental FA2-V100 options described above as alternatives).
- The `dtype` widget of `ModelComputeDtype` expects a string such as `fp16`/`fp32`. If your ComfyUI version is very old and lacks this core node, update ComfyUI first.

## License

MIT (recommended for SwarmUI extensions).
