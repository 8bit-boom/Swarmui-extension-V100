# SwarmUI V100 Compatibility Extension

Patches SwarmUI's ComfyUI workflows so they run correctly on **NVIDIA Volta (sm_70)** cards — Tesla V100, Titan V.

## Why the V100 needs special handling

The V100 is still a capable card (16/32 GB HBM2, fast FP16 tensor cores), but it predates a few things modern pipelines assume:

1. **No bf16 compute.** Volta only supports FP16/FP32. bf16 tensors get *silently emulated via fp32*, which is ~4x slower, spikes VRAM, and some nodes just crash with errors like `RuntimeError: expected scalar type Half but found BFloat16`. Some newer models (FLUX-family, video models) whitelist only bf16/fp32, so on a V100 ComfyUI silently falls back to fp32 and crawls.
2. **cu130 PyTorch wheels dropped sm_70.** If your torch is a cu130 build, every CUDA op fails with `no kernel image is available for execution on the device`, even though `torch.cuda.is_available()` returns True. Use a **cu128 (or older)** build:
   ```bash
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
   ```
3. **Triton 3.3+ and FlashAttention-3 dropped Volta.** SageAttention, `torch.compile`, and FA3-dependent nodes will fail on a V100.

## What this extension does

- At startup, runs `nvidia-smi` to detect a Volta-class GPU (compute capability 7.x).
- Adds an **Advanced Options** group "V100 / Volta Compatibility" to the generation page:
  - **V100 Compatibility Patch** — on by default when a Volta GPU is detected. Inserts ComfyUI's core `ModelComputeDtype` node immediately after base model loading (model-gen step priority `-50`, after the loader at `-100` but before sampling), so every downstream node (sampler, refiner, etc.) gets a V100-safe dtype.
  - **V100 Precision** — `fp16` (default, ~4x faster on Volta tensor cores) or `fp32` (safe fallback for the few models that NaN/overflow in fp16).
- On non-Volta machines it stays completely inert unless you manually enable the toggle.

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

The official FlashAttention wheels don't support Volta, but there's an unofficial hand-written FA2 port for sm_70: [ai-bond/flash-attention-v100](https://github.com/ai-bond/flash-attention-v100). It exposes the same API (`flash_attn_func`, `flash_attn_varlen_func`, ...) as the official package.

**Treat this as experimental.** It's a self-described "from scratch, for self education" project with no test suite — silently wrong attention output is worse than slow attention. Verify outputs against `--use-pytorch-cross-attention` before trusting it.

If you want to try it:

1. Build & install it into your ComfyUI venv per the repo's `install.md` (it compiles CUDA kernels for sm_70; you need the CUDA toolkit and ~matching torch version).
2. ComfyUI's `--use-flash-attention` flag does a plain `import flash_attn`, but the port installs as `flash_attn_v100`. Add a shim so the import succeeds — create `flash_attn.py` in your venv's site-packages:
   ```python
   from flash_attn_v100 import *  # noqa: F401,F403
   ```
3. In SwarmUI: **Server → Backends → Configure** your ComfyUI backend and add `--use-flash-attention` to the extra args (remove `--use-pytorch-cross-attention` if set).
4. Keep this extension's dtype patch enabled regardless — FlashAttention does not fix bf16 problems, and the two fixes operate at different layers (memory/speed vs dtype compatibility).

## Notes & manual steps the extension can't do for you

- **Keep your ComfyUI's PyTorch on a cu128 or older build** (see above). cu130+ has no sm_70 kernels at all, and no workflow patch can fix that.
- Avoid ComfyUI launch args that request Volta-unsupported features, e.g. SageAttention / `torch.compile` won't work on a V100; `--use-pytorch-cross-attention` is the safe default attention backend (with the experimental FA2-V100 route described above as an alternative).
- The `dtype` widget of `ModelComputeDtype` expects a string such as `fp16`/`fp32`. If your ComfyUI version is very old and lacks this core node, update ComfyUI first.

## License

MIT (recommended for SwarmUI extensions).
