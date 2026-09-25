using SwarmUI.Core;
using SwarmUI.Utils;
using SwarmUI.Text2Image;
using Newtonsoft.Json.Linq;
using System.Diagnostics;

// NOTE: Namespace must NOT contain "SwarmUI" (reserved for built-in extensions)
namespace V100Compat;

/// <summary>
/// SwarmUI extension that patches ComfyUI workflows so they run correctly on
/// NVIDIA Volta (sm_70) cards such as the Tesla V100 / Titan V.
///
/// Why the V100 needs this:
/// - Volta has NO bf16 compute. bf16 is silently emulated via fp32, which is
///   roughly 4x slower and can spike VRAM usage ("expected scalar type Half
///   but found BFloat16", silent fp32 fallback, OOMs on big models).
/// - Modern PyTorch cu130 wheels dropped sm_70 kernels entirely
///   ("no kernel image is available for execution on the device").
/// - Triton 3.3+ and FlashAttention-3 dropped Volta, so anything relying on
///   them (SageAttention, torch.compile, FA3) will fail.
///
/// This extension inserts ComfyUI's core "ModelComputeDtype" node right after
/// the base model is loaded, forcing a V100-native dtype (fp16 by default,
/// fp32 as a safe fallback for models that NaN in fp16).
///
/// It also registers related ComfyUI node packs as opt-in installable features:
/// - ComfyUI Flash-Attention V100 (experimental Volta FlashAttention node pack)
/// - ComfyUI-GGUF KREA-2 (RealRebelAI fork) for Krea 2 GGUF support
///   (upstream city96/ComfyUI-GGUF doesn't parse Krea 2 ops yet, city96#464,
///   and the fork shares node/class names with upstream - do NOT install both).
/// </summary>
public class V100CompatExtension : Extension
{
    public static T2IRegisteredParam<bool> EnablePatch;
    public static T2IRegisteredParam<string> Precision;
    public static T2IParamGroup V100Group;

    /// <summary>Set true when a Volta-class (compute capability 7.x) GPU is detected on this machine.</summary>
    public static bool IsVolta = false;

    public override void OnPreInit()
    {
        IsVolta = DetectVoltaGpu();
        if (IsVolta)
        {
            Logs.Init("V100Compat: Volta (sm_70) GPU detected. Enable 'V100 Compatibility Patch' in generation parameters (or Server tab > Backends) to force V100-safe precision, and make sure your PyTorch is a cu128 (or older) build - cu130 wheels dropped sm_70 support.");
        }
    }

    public override void OnInit()
    {
        // The group shows under Advanced Options in the Text2Image tab.
        V100Group = new("V100 / Volta Compatibility", Toggles: false, Open: true, IsAdvanced: true);

        // Default the patch to ON when we detected a Volta card, OFF otherwise
        // (so the extension is inert on Ampere+ machines unless the user opts in).
        EnablePatch = T2IParamTypes.Register<bool>(new("V100 Compatibility Patch",
            "Inserts a ModelComputeDtype node after base model loading to force a V100-safe compute dtype. " +
            "Use this on Tesla V100 / Titan V (Volta, no bf16 support) if you see bf16 dtype errors, " +
            "'expected scalar type Half but found BFloat16', or ~4x slowdowns from silent fp32 fallback.",
            IsVolta ? "true" : "false", Toggleable: true, Group: V100Group, FeatureFlag: "comfyui"));

        Precision = T2IParamTypes.Register<string>(new("V100 Precision",
            "Compute dtype to force when the V100 Compatibility Patch is active. " +
            "fp16 is ~4x faster on Volta tensor cores but a few models (e.g. some video/LLM-backed models) " +
            "overflow to NaN in fp16 - switch those to fp32.",
            "fp16", Toggleable: true, GetValues: _ => ["fp16", "fp32"], Group: V100Group, FeatureFlag: "comfyui"));

        // Model Gen Steps with priority below -100 run BEFORE base model loading.
        // We want to run AFTER the base model (and CLIP/VAE) have been loaded,
        // but before sampling / refiner loading, so -50 is a good slot.
        // Check the source of WorkflowGenerator for the full priority map.
        WorkflowGenerator.AddModelGenStep(g =>
        {
            if (!g.UserInput.TryGet(EnablePatch, out bool enabled) || !enabled)
            {
                return; // leave the workflow untouched unless the user opted in
            }
            if (!g.UserInput.TryGet(Precision, out string precision))
            {
                precision = "fp16";
            }
            // Guard: only patch when a model was actually loaded by this step chain
            if (g.LoadingModel is null)
            {
                return;
            }
            string dtypeNode = g.CreateNode("ModelComputeDtype", new JObject()
            {
                ["model"] = g.LoadingModel, // Take in the freshly loaded MODEL
                ["dtype"] = precision
            });
            g.LoadingModel = [dtypeNode, 0]; // Every later step (sampler, refiner) now uses the fixed model
        }, -50);

        // Register the ComfyUI custom-node FlashAttention pack as an installable feature,
        // so users can install it from SwarmUI (Server area / alongside the ComfyUI backend)
        // instead of manually cloning into custom_nodes. Not auto-installed: it is an
        // unofficial kernel, so the user should opt in deliberately.
        InstallableFeatures.RegisterInstallableFeature(new("ComfyUI Flash-Attention V100",
            "comfyui-flash-attention-v100",
            "https://github.com/NetVoobrazhenia/ComfyUI_Flash-Attention_v100",
            "NetVoobrazhenia"));

        // Krea 2 GGUF support: SwarmUI core already detects Krea 2 models (Raw and Turbo),
        // but upstream city96/ComfyUI-GGUF doesn't parse Krea 2's ops yet (city96#464).
        // The RealRebelAI/ComfyUI-GGUF_KREA-2 fork is a drop-in replacement with Krea 2
        // support (incl. the Qwen3-VL GGUF text encoder). It shares node/class names with
        // the upstream pack - DO NOT install both; this one supersedes it.
        InstallableFeatures.RegisterInstallableFeature(new("ComfyUI-GGUF KREA-2 (fork)",
            "comfyui-gguf-krea2",
            "https://github.com/RealRebelAI/ComfyUI-GGUF_KREA-2",
            "RealRebelAI"));

        Logs.Init("V100Compat extension loaded.");
    }

    /// <summary>Detects any Volta-class GPU (compute capability 7.x, e.g. V100/Titan V) via nvidia-smi.</summary>
    public static bool DetectVoltaGpu()
    {
        try
        {
            ProcessStartInfo psi = new("nvidia-smi", "--query-gpu=compute_cap --format=csv,noheader")
            {
                RedirectStandardOutput = true,
                RedirectStandardError = true,
                UseShellExecute = false,
                CreateNoWindow = true
            };
            using Process proc = Process.Start(psi);
            if (proc is null)
            {
                return false;
            }
            string output = proc.StandardOutput.ReadToEnd();
            proc.WaitForExit(5000);
            foreach (string line in output.Split('\n', StringSplitOptions.RemoveEmptyEntries | StringSplitOptions.TrimEntries))
            {
                // Volta = 7.0 (V100/Titan V). Anything 8.0+ (Ampere+) has native bf16 and needs no patch.
                if (line.StartsWith("7."))
                {
                    return true;
                }
            }
        }
        catch
        {
            // nvidia-smi missing, not on PATH, or timed out - fail silent and stay inert
        }
        return false;
    }
}
