"""Opt-in compatibility for FastWAM training with torch_musa 2.7.1."""
import os

import torch


def sinusoidal_embedding(dim, position):
    # MUSA 2.7 lacks the FP64 operations used by the upstream embedding.
    half = dim // 2
    exponents = -torch.arange(half, dtype=torch.float32, device=position.device) / half
    phase = torch.outer(position.float(), torch.pow(10000, exponents))
    return torch.cat((phase.cos(), phase.sin()), dim=1).to(position.dtype)


def real_frequency_table(complex_table):
    # Generate upstream frequencies on CPU, then interleave cos/sin pairs.
    # This preserves the temporal/height/width concatenation in WanVideoDiT.
    return torch.view_as_real(complex_table).flatten(-2).to(torch.float32)


def apply_real_rope(x, freqs, num_heads):
    batch, length, width = x.shape
    pair_width = width // num_heads // 2
    real, imag = x.float().reshape(batch, length, num_heads, pair_width, 2).unbind(-1)
    cos, sin = freqs.reshape(length, 1, pair_width, 2).unbind(-1)
    return torch.stack((real * cos - imag * sin, real * sin + imag * cos), dim=-1).reshape_as(x).to(x.dtype)


def musa_rank_device():
    rank = int(os.environ.get("LOCAL_RANK", "0"))
    count = torch.musa.device_count()
    if not 0 <= rank < count:
        raise RuntimeError(f"Invalid LOCAL_RANK={rank} for {count} visible MUSA devices")
    torch.musa.set_device(rank)
    return f"musa:{rank}"


def install():
    """Install the proven MUSA training path before constructing the model.

    Only scripts/train_musa.py opts in; the upstream CUDA entry is unchanged.
    """
    import torch_musa  # noqa: F401 - register the MUSA backend
    from fastwam import runtime
    from fastwam.models.wan22 import action_dit, mot, wan_video_dit
    from fastwam.models.wan22.fastwam import FastWAM

    if getattr(install, "_installed", False):
        return
    upstream_table = wan_video_dit.precompute_freqs_cis
    upstream_encode = FastWAM._encode_video_latents

    def frequency_table(*args, **kwargs):
        return real_frequency_table(upstream_table(*args, **kwargs))

    @torch.no_grad()
    def encode_video_latents(model, video, tiled=False, **kwargs):
        if model.device.type != "musa" or tiled:
            return upstream_encode(model, video, tiled=tiled, **kwargs)
        # Upstream uses the CUDA-only cudagraphs compiler backend here.
        # Keep the same batched encode, normalization and output clone.
        return model.vae.model.encode(video.to(model.device), model.vae.scale).clone()

    wan_video_dit.precompute_freqs_cis = frequency_table
    action_dit.precompute_freqs_cis = frequency_table
    wan_video_dit.sinusoidal_embedding_1d = sinusoidal_embedding
    action_dit.sinusoidal_embedding_1d = sinusoidal_embedding
    wan_video_dit.rope_apply = apply_real_rope
    mot.rope_apply = apply_real_rope
    FastWAM._encode_video_latents = encode_video_latents
    runtime._resolve_train_device = musa_rank_device
    install._installed = True
