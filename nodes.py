"""
ComfyUI custom nodes for CompVis Latent Diffusion Models (LDM), 2021.
Architecture: BERT text encoder + UNet with Spatial Transformers + f8 VAE.
This is the original pre-Stable-Diffusion model from the LDM paper
(Rombach et al., "High-Resolution Image Synthesis with Latent Diffusion Models").
"""
import math
import os
import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
import folder_paths

try:
    from tqdm import tqdm as _tqdm
except Exception:
    _tqdm = None


# ════════════════════════════════════════════════════════════════
# ATTENTION BACKEND SELECTION
# ════════════════════════════════════════════════════════════════

_ATTENTION_BACKEND: str | None = None

def _get_attention_backend() -> str:
    """
    Determines attention implementation from ComfyUI CLI flags.
    Priority: flash > split > quad > pytorch > sdpa (default).
    Sage Attention is NOT supported here → falls back to SDPA.
    """
    global _ATTENTION_BACKEND
    if _ATTENTION_BACKEND is not None:
        return _ATTENTION_BACKEND

    backend = "sdpa"  # safe default

    try:
        from comfy.cli_args import args

        if getattr(args, "use_flash_attention", False):
            backend = "flash"
        elif getattr(args, "use_split_cross_attention", False):
            backend = "split"
        elif getattr(args, "use_quad_cross_attention", False):
            backend = "quad"
        elif getattr(args, "use_pytorch_cross_attention", False):
            backend = "pytorch"
        elif getattr(args, "use_sage_attention", False):
            logging.warning(
                "[LDM] Sage Attention is not supported by this model. "
                "Falling back to PyTorch SDPA."
            )
            backend = "sdpa"
        else:
            backend = "sdpa"
    except (ImportError, AttributeError):
        backend = "sdpa"

    _ATTENTION_BACKEND = backend
    logging.info(f"[LDM] Attention backend: {backend}")
    return backend


# ── Split / Quad helpers (memory-saving chunked attention) ───────

def _split_cross_attention(q, k, v, dim_head, chunk_size=512):
    """Process attention in chunks over the query sequence to save VRAM."""
    B, H, N, _ = q.shape
    scale = dim_head ** -0.5
    out = torch.empty_like(q)

    for i in range(0, N, chunk_size):
        end = min(i + chunk_size, N)
        q_chunk = q[:, :, i:end]                       # [B, H, chunk, D]
        attn = (q_chunk @ k.transpose(-2, -1)) * scale  # [B, H, chunk, Nk]
        attn = attn.softmax(dim=-1)
        out[:, :, i:end] = attn @ v

    return out


def _quad_cross_attention(q, k, v, dim_head, chunk_size=256):
    """Quad-tree style chunked attention (smaller chunks than split)."""
    return _split_cross_attention(q, k, v, dim_head, chunk_size=chunk_size)


# ── Unified dispatch ─────────────────────────────────────────────

def _attention_dispatch(q, k, v, dim_head):
    """
    q, k, v: [B, heads, N, dim_head]
    Returns: [B, heads, N, dim_head]
    """
    backend = _get_attention_backend()

    if backend == "flash":
        # Force flash-attention kernel via SDPA context
        try:
            with torch.nn.attention.sdpa_kernel(
                [torch.nn.attention.SDPBackend.FLASH_ATTENTION,
                 torch.nn.attention.SDPBackend.EFFICIENT_ATTENTION]
            ):
                return F.scaled_dot_product_attention(q, k, v)
        except (AttributeError, RuntimeError):
            # PyTorch < 2.2 or flash unavailable → plain SDPA
            return F.scaled_dot_product_attention(q, k, v)

    elif backend == "pytorch":
        # Explicit PyTorch SDPA (math backend allowed)
        try:
            with torch.nn.attention.sdpa_kernel(
                [torch.nn.attention.SDPBackend.MATH]
            ):
                return F.scaled_dot_product_attention(q, k, v)
        except (AttributeError, RuntimeError):
            return F.scaled_dot_product_attention(q, k, v)

    elif backend == "split":
        return _split_cross_attention(q, k, v, dim_head)

    elif backend == "quad":
        return _quad_cross_attention(q, k, v, dim_head)

    else:
        # "sdpa" — default: let PyTorch pick the best kernel
        return F.scaled_dot_product_attention(q, k, v)


# ════════════════════════════════════════════════════════════════
# TEXT ENCODER  (cond_stage_model.transformer.*)
# Keys: token_emb, pos_emb.emb, attn_layers.layers.{0..63},
#       norm, to_logits
# ════════════════════════════════════════════════════════════════

class _PositionalEmbedding(nn.Module):
    """Produces key pos_emb.emb.weight"""
    def __init__(self, max_len, dim):
        super().__init__()
        self.emb = nn.Embedding(max_len, dim)


class _Attention(nn.Module):
    """
    to_q/to_k/to_v: [512, 1280] no bias; to_out: [1280, 512] with bias.
    8 heads × 64 = 512 (as in the original BERTEmbedder).
    """
    def __init__(self, dim=1280, dim_head=64, num_heads=8):
        super().__init__()
        self.h, self.d = num_heads, dim_head
        inner = num_heads * dim_head
        self.to_q = nn.Linear(dim, inner, bias=False)
        self.to_k = nn.Linear(dim, inner, bias=False)
        self.to_v = nn.Linear(dim, inner, bias=False)
        self.to_out = nn.Linear(inner, dim, bias=True)

    def forward(self, x):
        B, N, _ = x.shape
        h, d = self.h, self.d

        q = self.to_q(x).view(B, N, h, d).transpose(1, 2)
        k = self.to_k(x).view(B, N, h, d).transpose(1, 2)
        v = self.to_v(x).view(B, N, h, d).transpose(1, 2)

        out = _attention_dispatch(q, k, v, d)

        out = out.transpose(1, 2).reshape(B, N, h * d)
        return self.to_out(out)


class _FFN(nn.Module):
    """
    net.0.0 = Linear(1280→5120), net.0.1 = GELU (no weights),
    net.1 = Dropout (no weights), net.2 = Linear(5120→1280)
    """
    def __init__(self, dim=1280, mult=4):
        super().__init__()
        inner = dim * mult
        self.net = nn.Sequential(
            nn.Sequential(nn.Linear(dim, inner), nn.GELU()),
            nn.Dropout(0.0),
            nn.Linear(inner, dim),
        )

    def forward(self, x):
        return self.net(x)


class LDMTextEncoder(nn.Module):
    def __init__(self, vocab_size=30522, dim=1280, max_seq_len=77,
                 num_layers=64, dim_head=64, num_heads=8, ff_mult=4):
        super().__init__()
        self.token_emb = nn.Embedding(vocab_size, dim)
        self.pos_emb = _PositionalEmbedding(max_seq_len, dim)

        layers = []
        for i in range(num_layers):
            norm = nn.LayerNorm(dim)
            sub = _Attention(dim, dim_head, num_heads) if i % 2 == 0 \
                else _FFN(dim, ff_mult)
            layers.append(nn.ModuleList([norm, sub]))
        self.attn_layers = nn.ModuleDict({"layers": nn.ModuleList(layers)})
        self.norm = nn.LayerNorm(dim)
        self.to_logits = nn.Linear(dim, vocab_size, bias=True)

    def forward(self, token_ids):
        B, N = token_ids.shape
        pos = torch.arange(N, device=token_ids.device).unsqueeze(0)
        x = self.token_emb(token_ids) + self.pos_emb.emb(pos)
        for norm, sub in self.attn_layers["layers"]:
            x = sub(norm(x)) + x
        return self.norm(x)


# ════════════════════════════════════════════════════════════════
# UNET  (model.diffusion_model.*)
# ════════════════════════════════════════════════════════════════

def _timestep_embedding(t, dim, max_period=10000):
    half = dim // 2
    freqs = torch.exp(-math.log(max_period)
                      * torch.arange(half, device=t.device,
                                     dtype=torch.float32) / half)
    args = t[:, None].float() * freqs[None]
    return torch.cat([torch.cos(args), torch.sin(args)], dim=-1)


class _GEGLU(nn.Module):
    """ff.net.0.proj = Linear(C → 2*4C)"""
    def __init__(self, dim_in, dim_out):
        super().__init__()
        self.proj = nn.Linear(dim_in, dim_out * 2)

    def forward(self, x):
        x, gate = self.proj(x).chunk(2, dim=-1)
        return x * F.gelu(gate)


class _UNetFF(nn.Module):
    """ff.net.0 = GEGLU, ff.net.1 = Dropout, ff.net.2 = Linear"""
    def __init__(self, dim, mult=4):
        super().__init__()
        inner = dim * mult
        self.net = nn.Sequential(
            _GEGLU(dim, inner),
            nn.Dropout(0.0),
            nn.Linear(inner, dim),
        )

    def forward(self, x):
        return self.net(x)


class _UNetAttn(nn.Module):
    """to_q/k/v no bias; to_out.0 with bias (Sequential → index .0)"""
    def __init__(self, dim, ctx_dim=None, num_heads=8):
        super().__init__()
        self.heads = num_heads
        ctx_dim = ctx_dim or dim
        self.to_q = nn.Linear(dim, dim, bias=False)
        self.to_k = nn.Linear(ctx_dim, dim, bias=False)
        self.to_v = nn.Linear(ctx_dim, dim, bias=False)
        self.to_out = nn.Sequential(nn.Linear(dim, dim))

    def forward(self, x, ctx=None):
        ctx = ctx if ctx is not None else x
        B, N, C = x.shape
        h = self.heads
        d = C // h

        q = self.to_q(x).view(B, N, h, d).transpose(1, 2)
        k = self.to_k(ctx).view(B, -1, h, d).transpose(1, 2)
        v = self.to_v(ctx).view(B, -1, h, d).transpose(1, 2)

        out = _attention_dispatch(q, k, v, d)

        out = out.transpose(1, 2).reshape(B, N, C)
        return self.to_out[0](out)


class _BasicTransformerBlock(nn.Module):
    def __init__(self, dim, ctx_dim, num_heads=8):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn1 = _UNetAttn(dim, dim, num_heads)
        self.norm2 = nn.LayerNorm(dim)
        self.attn2 = _UNetAttn(dim, ctx_dim, num_heads)
        self.norm3 = nn.LayerNorm(dim)
        self.ff = _UNetFF(dim)

    def forward(self, x, ctx):
        x = self.attn1(self.norm1(x)) + x
        x = self.attn2(self.norm2(x), ctx) + x
        x = self.ff(self.norm3(x)) + x
        return x


class _SpatialTransformer(nn.Module):
    def __init__(self, channels, ctx_dim, num_heads=8):
        super().__init__()
        self.norm = nn.GroupNorm(32, channels)
        self.proj_in = nn.Conv2d(channels, channels, 1)
        self.transformer_blocks = nn.ModuleList(
            [_BasicTransformerBlock(channels, ctx_dim, num_heads)])
        self.proj_out = nn.Conv2d(channels, channels, 1)

    def forward(self, x, ctx):
        B, C, H, W = x.shape
        h = self.proj_in(self.norm(x))
        h = h.view(B, C, -1).permute(0, 2, 1)
        for blk in self.transformer_blocks:
            h = blk(h, ctx)
        h = h.permute(0, 2, 1).view(B, C, H, W)
        return self.proj_out(h) + x


class _ResBlock(nn.Module):
    """
    in_layers.0/2, emb_layers.1, out_layers.0/3,
    skip_connection (only when channel count changes)
    """
    def __init__(self, channels, emb_ch, out_ch):
        super().__init__()
        self.in_layers = nn.Sequential(
            nn.GroupNorm(32, channels), nn.SiLU(),
            nn.Conv2d(channels, out_ch, 3, padding=1))
        self.emb_layers = nn.Sequential(nn.SiLU(), nn.Linear(emb_ch, out_ch))
        self.out_layers = nn.Sequential(
            nn.GroupNorm(32, out_ch), nn.SiLU(), nn.Dropout(0.0),
            nn.Conv2d(out_ch, out_ch, 3, padding=1))
        self.skip_connection = (nn.Conv2d(channels, out_ch, 1)
                                if channels != out_ch else nn.Identity())

    def forward(self, x, emb):
        h = self.in_layers(x)
        h = h + self.emb_layers(emb)[:, :, None, None]
        return self.out_layers(h) + self.skip_connection(x)


class _Downsample(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.op = nn.Conv2d(ch, ch, 3, stride=2, padding=1)

    def forward(self, x):
        return self.op(x)


class _Upsample(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.conv = nn.Conv2d(ch, ch, 3, padding=1)

    def forward(self, x):
        return self.conv(F.interpolate(x, scale_factor=2, mode="nearest"))


class LDMUNet(nn.Module):
    def __init__(self, ctx_dim=1280, num_heads=8):
        super().__init__()
        ch, emb_ch = 320, 1280

        self.time_embed = nn.Sequential(
            nn.Linear(ch, emb_ch), nn.SiLU(), nn.Linear(emb_ch, emb_ch))

        T = _SpatialTransformer
        self.input_blocks = nn.ModuleList([
            nn.ModuleList([nn.Conv2d(4, ch, 3, padding=1)]),
            nn.ModuleList([_ResBlock(ch, emb_ch, ch),   T(ch, ctx_dim, num_heads)]),
            nn.ModuleList([_ResBlock(ch, emb_ch, ch),   T(ch, ctx_dim, num_heads)]),
            nn.ModuleList([_Downsample(ch)]),
            nn.ModuleList([_ResBlock(ch, emb_ch, 640),  T(640, ctx_dim, num_heads)]),
            nn.ModuleList([_ResBlock(640, emb_ch, 640), T(640, ctx_dim, num_heads)]),
            nn.ModuleList([_Downsample(640)]),
            nn.ModuleList([_ResBlock(640, emb_ch, 1280),  T(1280, ctx_dim, num_heads)]),
            nn.ModuleList([_ResBlock(1280, emb_ch, 1280), T(1280, ctx_dim, num_heads)]),
            nn.ModuleList([_Downsample(1280)]),
            nn.ModuleList([_ResBlock(1280, emb_ch, 1280)]),
            nn.ModuleList([_ResBlock(1280, emb_ch, 1280)]),
        ])

        self.middle_block = nn.ModuleList([
            _ResBlock(1280, emb_ch, 1280),
            T(1280, ctx_dim, num_heads),
            _ResBlock(1280, emb_ch, 1280),
        ])

        self.output_blocks = nn.ModuleList([
            nn.ModuleList([_ResBlock(2560, emb_ch, 1280)]),
            nn.ModuleList([_ResBlock(2560, emb_ch, 1280)]),
            nn.ModuleList([_ResBlock(2560, emb_ch, 1280), _Upsample(1280)]),
            nn.ModuleList([_ResBlock(2560, emb_ch, 1280), T(1280, ctx_dim, num_heads)]),
            nn.ModuleList([_ResBlock(2560, emb_ch, 1280), T(1280, ctx_dim, num_heads)]),
            nn.ModuleList([_ResBlock(1920, emb_ch, 1280), T(1280, ctx_dim, num_heads),
                           _Upsample(1280)]),
            nn.ModuleList([_ResBlock(1920, emb_ch, 640),  T(640, ctx_dim, num_heads)]),
            nn.ModuleList([_ResBlock(1280, emb_ch, 640),  T(640, ctx_dim, num_heads)]),
            nn.ModuleList([_ResBlock(960, emb_ch, 640),   T(640, ctx_dim, num_heads),
                           _Upsample(640)]),
            nn.ModuleList([_ResBlock(960, emb_ch, 320),   T(320, ctx_dim, num_heads)]),
            nn.ModuleList([_ResBlock(640, emb_ch, 320),   T(320, ctx_dim, num_heads)]),
            nn.ModuleList([_ResBlock(640, emb_ch, 320),   T(320, ctx_dim, num_heads)]),
        ])

        self.out = nn.Sequential(
            nn.GroupNorm(32, ch), nn.SiLU(), nn.Conv2d(ch, 4, 3, padding=1))

    def forward(self, x, timesteps, context):
        emb = self.time_embed(
            _timestep_embedding(timesteps, 320).to(x.dtype))

        skips = []
        h = x
        for block in self.input_blocks:
            for layer in block:
                if isinstance(layer, _SpatialTransformer):
                    h = layer(h, context)
                elif isinstance(layer, _ResBlock):
                    h = layer(h, emb)
                else:
                    h = layer(h)
            skips.append(h)

        for layer in self.middle_block:
            if isinstance(layer, _SpatialTransformer):
                h = layer(h, context)
            elif isinstance(layer, _ResBlock):
                h = layer(h, emb)
            else:
                h = layer(h)

        for block in self.output_blocks:
            h = torch.cat([h, skips.pop()], dim=1)
            for layer in block:
                if isinstance(layer, _SpatialTransformer):
                    h = layer(h, context)
                elif isinstance(layer, _ResBlock):
                    h = layer(h, emb)
                else:
                    h = layer(h)

        return self.out(h)


# ════════════════════════════════════════════════════════════════
# VAE (standard AutoencoderKL from ComfyUI — keys match 1:1)
# ════════════════════════════════════════════════════════════════

def _build_vae(dtype):
    from comfy.ldm.models.autoencoder import AutoencoderKL
    vae = AutoencoderKL(
        embed_dim=4,
        ddconfig=dict(double_z=True, z_channels=4, resolution=256,
                      in_channels=3, out_ch=3, ch=128,
                      ch_mult=[1, 2, 4, 4], num_res_blocks=2,
                      attn_resolutions=[], dropout=0.0))
    return vae.to(dtype).eval()


# ════════════════════════════════════════════════════════════════
# NOISE SCHEDULE + SIGMA UTILITIES
# ════════════════════════════════════════════════════════════════

class NoiseSchedule:
    def __init__(self, other):
        ac = other["alphas_cumprod"]
        self.alphas_cumprod = ac
        self.T = len(other["betas"])
        self.sigma_min = max(float(((1 - ac[0]) / ac[0]).sqrt()), 1e-4)
        self.sigma_max = float(((1 - ac[-1]) / ac[-1]).sqrt())

    def sigma_to_t(self, sigma):
        """Nearest integer DDPM timestep for the given sigma."""
        a = 1.0 / (1.0 + float(sigma) ** 2)
        return int((self.alphas_cumprod - a).abs().argmin())


def make_sigmas(scheduler, steps, ns):
    """Returns a sigma tensor of length steps+1, descending to 0."""
    ac, T = ns.alphas_cumprod, ns.T

    def sig_ts(ts):
        a = ac[torch.as_tensor(ts, dtype=torch.long)].clamp(min=1e-8)
        return ((1 - a) / a).sqrt()

    if scheduler == "ddim_uniform":
        ts = list(range(T - 1, -1, -(T // steps)))
        s = sig_ts(ts)
        return torch.cat([s, s.new_zeros(1)])

    if scheduler == "linear":
        s = sig_ts(torch.linspace(T - 1, 0, steps + 1).round().long())
        s[-1] = 0
        return s

    if scheduler == "beta":
        ts = (torch.linspace(0, 1, steps + 1).sqrt() * (T - 1)).round().long().flip(0)
        s = sig_ts(ts)
        s[-1] = 0
        return s

    if scheduler == "karras":
        rho, lo, hi = 7.0, ns.sigma_min, ns.sigma_max
        r = hi ** (1 / rho) + torch.linspace(0, 1, steps) * (lo ** (1 / rho) - hi ** (1 / rho))
        return torch.cat([r ** rho, torch.zeros(1)])

    if scheduler == "exponential":
        lo, hi = ns.sigma_min, ns.sigma_max
        s = hi * (lo / hi) ** torch.linspace(0, 1, steps)
        return torch.cat([s, torch.zeros(1)])

    # cosine (Nichol & Dhariwal)
    u = torch.linspace(1, 0, steps + 1)
    a = torch.cos((u + 0.008) / 1.008 * math.pi / 2) ** 2
    s = ((1 - a) / a.clamp(min=1e-8)).sqrt()
    s[0] = min(float(s[0]), ns.sigma_max)
    s[-1] = 0
    return s


def _make_predict(unet, cond, ns, device):
    def predict_eps(x, sigma):
        t = ns.sigma_to_t(sigma)
        tt = torch.full((x.shape[0],), t, device=device, dtype=torch.long)
        return unet(x, tt, context=cond)

    def denoise(y, sigma):
        """y is in x0 + sigma*eps space; returns predicted x0."""
        s = float(sigma)
        scale = math.sqrt(1.0 + s * s)
        eps = predict_eps(y / scale, sigma)
        return y - s * eps

    return predict_eps, denoise


# ── Alpha-space samplers (ddim / ddpm / plms) ────────────────────

def _sample_ancestral(predict_eps, x, sigmas, eta, pbar_update):
    n = len(sigmas) - 1
    for i in range(n):
        s, s2 = float(sigmas[i]), float(sigmas[i + 1])
        if s <= 0.0:
            break
        a_t = 1.0 / (1.0 + s * s)
        a_p = 1.0 / (1.0 + s2 * s2)
        eps = predict_eps(x, s)
        x0 = (x - math.sqrt(1.0 - a_t) * eps) / math.sqrt(a_t)
        var = max(0.0, (1.0 - a_p) / (1.0 - a_t) * (1.0 - a_t / a_p))
        sigma_k = eta * math.sqrt(var)
        x = math.sqrt(a_p) * x0 + math.sqrt(max(0.0, 1.0 - a_p - sigma_k ** 2)) * eps
        if eta > 0 and i < n - 1:
            x = x + sigma_k * torch.randn_like(x)
        pbar_update(i)
    return x


def _sample_plms(predict_eps, x, sigmas, eta, pbar_update):
    old = []
    n = len(sigmas) - 1
    for i in range(n):
        s, s2 = float(sigmas[i]), float(sigmas[i + 1])
        if s <= 0.0:
            break
        a_t = 1.0 / (1.0 + s * s)
        a_p = 1.0 / (1.0 + s2 * s2)
        eps = predict_eps(x, s)

        def step(e):
            x0 = (x - math.sqrt(1.0 - a_t) * e) / math.sqrt(a_t)
            return math.sqrt(a_p) * x0 + math.sqrt(1.0 - a_p) * e

        if len(old) == 0:
            e2 = predict_eps(step(eps), s2) if s2 > 0 else eps
            ep = (eps + e2) / 2
        elif len(old) == 1:
            ep = (3 * eps - old[-1]) / 2
        elif len(old) == 2:
            ep = (23 * eps - 16 * old[-1] + 5 * old[-2]) / 12
        else:
            ep = (55 * eps - 59 * old[-1] + 37 * old[-2] - 9 * old[-3]) / 24

        x = step(ep)
        old.append(eps)
        if len(old) >= 4:
            old.pop(0)
        pbar_update(i)
    return x


# ── Sigma-space samplers ─────────────────────────────────────────

def _sample_euler(denoise, x, sigmas, pbar_update):
    for i in range(len(sigmas) - 1):
        s, s2 = float(sigmas[i]), float(sigmas[i + 1])
        if s <= 0.0:
            break
        d = (x - denoise(x, s)) / s
        x = x + d * (s2 - s)
        pbar_update(i)
    return x


def _sample_euler_ancestral(denoise, x, sigmas, pbar_update):
    for i in range(len(sigmas) - 1):
        s, s2 = float(sigmas[i]), float(sigmas[i + 1])
        if s <= 0.0:
            break
        den = denoise(x, s)
        d = (x - den) / s
        su = math.sqrt(max(0.0, (s2 * s2 * (s * s - s2 * s2)) / (s * s)))
        sd = math.sqrt(max(0.0, s2 * s2 - su * su))
        x = den + d * sd
        if s2 > 0:
            x = x + su * torch.randn_like(x)
        pbar_update(i)
    return x


def _sample_heun(denoise, x, sigmas, pbar_update):
    for i in range(len(sigmas) - 1):
        s, s2 = float(sigmas[i]), float(sigmas[i + 1])
        if s <= 0.0:
            break
        d1 = (x - denoise(x, s)) / s
        x2 = x + d1 * (s2 - s)
        if s2 > 0:
            d2 = (x2 - denoise(x2, s2)) / s2
            x = x + (d1 + d2) / 2 * (s2 - s)
        else:
            x = x + d1 * (s2 - s)
        pbar_update(i)
    return x


def _sample_dpmpp_2m(denoise, x, sigmas, pbar_update):
    old_den, s_prev = None, None
    for i in range(len(sigmas) - 1):
        s, s2 = float(sigmas[i]), float(sigmas[i + 1])
        if s <= 0.0:
            break
        den = denoise(x, s)
        ratio = s2 / s
        if old_den is None or s2 == 0 or s2 == s:
            x = ratio * x + (1.0 - ratio) * den
        else:
            r = math.log(s_prev / s) / math.log(s / s2)
            den_d = (1.0 + 1.0 / (2.0 * r)) * den - (1.0 / (2.0 * r)) * old_den
            x = ratio * x + (1.0 - ratio) * den_d
        old_den, s_prev = den, s
        pbar_update(i)
    return x


ALPHA_SCHEDULERS = ("ddim_uniform", "linear", "beta")
SIGMA_SCHEDULERS = ("ddim_uniform", "linear", "beta", "karras", "exponential", "cosine")

SAMPLER_TABLE = {
    "ddim":            ("alpha", _sample_ancestral),
    "ddpm":            ("alpha", _sample_ancestral),
    "plms":            ("alpha", _sample_plms),
    "euler":           ("sigma", _sample_euler),
    "euler_ancestral": ("sigma", _sample_euler_ancestral),
    "heun":            ("sigma", _sample_heun),
    "dpmpp_2m":        ("sigma", _sample_dpmpp_2m),
}


@torch.no_grad()
def run_sampler(name, scheduler, unet, cond, ns, x_in,
                steps, eta, denoise, device, dtype=torch.float32):
    sigmas = make_sigmas(scheduler, steps, ns).to(device=device, dtype=dtype)
    sigmas = torch.cat([sigmas[sigmas > 0], sigmas.new_zeros(1)])
    cut = int(round((1.0 - denoise) * (len(sigmas) - 1)))
    sigmas = sigmas[cut:]

    predict_eps, denoise_fn = _make_predict(unet, cond, ns, device)

    n = len(sigmas) - 1
    ui_pbar = None
    try:
        import comfy.utils
        ui_pbar = comfy.utils.ProgressBar(n)
    except Exception:
        ui_pbar = None

    console_pbar = None
    if _tqdm is not None:
        console_pbar = _tqdm(
            total=n,
            desc=f"[LDM] {name}",
            unit="it",
            dynamic_ncols=True,
            leave=True,
        )

    def pbar_update(i):
        if console_pbar is not None:
            console_pbar.set_postfix(sigma=f"{float(sigmas[i]):.4f}", refresh=False)
            console_pbar.update(1)
        else:
            print(f"[LDM] {name}: step {i + 1}/{n}  sigma={float(sigmas[i]):.4f}")
        if ui_pbar is not None:
            ui_pbar.update_absolute(i + 1, n)

    noise = torch.randn_like(x_in)
    family, fn = SAMPLER_TABLE[name]
    s0 = float(sigmas[0])

    try:
        if family == "alpha":
            eff_eta = 1.0 if name == "ddpm" else eta
            if denoise >= 1.0:
                x = noise
            else:
                a0 = 1.0 / (1.0 + s0 * s0)
                x = math.sqrt(a0) * x_in + math.sqrt(1.0 - a0) * noise
            result = fn(predict_eps, x, sigmas, eff_eta, pbar_update)
        else:
            if denoise >= 1.0:
                y = s0 * noise
            else:
                y = x_in + s0 * noise
            result = fn(denoise_fn, y, sigmas, pbar_update)
    finally:
        if console_pbar is not None:
            console_pbar.close()

    return result


# ════════════════════════════════════════════════════════════════
# MEMORY MANAGEMENT
# ════════════════════════════════════════════════════════════════

import contextlib


class _Component:
    """Pipeline component: master weights on CPU + placement flag."""
    def __init__(self, name, module):
        self.name = name
        self.module = module
        self.on_device = False
        self.bytes = sum(p.numel() * p.element_size() for p in module.parameters())


class MemoryManager:
    """
    Strategies:
      keep    — load to GPU once and keep (max VRAM, max speed);
      auto    — keep on GPU if VRAM allows, otherwise offload after stage;
      offload — always return to CPU after stage (min VRAM);
      cpu     — no GPU usage at all.
    """
    MARGIN = 512 * 1024 * 1024

    def __init__(self, strategy="auto", device="cuda"):
        self.strategy = strategy
        self.device = device
        self.components = []

    def register(self, comp):
        self.components.append(comp)
        return comp

    @contextlib.contextmanager
    def use(self, comp):
        if self.device == "cpu":
            yield comp.module
            return
        self._load(comp)
        try:
            yield comp.module
        finally:
            self._maybe_release(comp)

    def _free(self):
        free, _ = torch.cuda.mem_get_info()
        return free

    def _load(self, comp):
        if comp.on_device:
            return
        if self._free() < comp.bytes + self.MARGIN:
            for other in self.components:
                if other is not comp and other.on_device:
                    self._offload(other)
        comp.module.to(self.device)
        comp.on_device = True
        logging.info(f"LDM {comp.name} → {self.device} ({comp.bytes / 2**20:.0f} MB)")

    def _offload(self, comp):
        comp.module.to("cpu")
        comp.on_device = False
        torch.cuda.empty_cache()
        logging.info(f"LDM {comp.name} → cpu (offloaded)")

    def _maybe_release(self, comp):
        if self.strategy == "keep":
            return
        if self.strategy == "offload":
            self._offload(comp)
            return
        if self._free() < comp.bytes + self.MARGIN:
            self._offload(comp)


# ════════════════════════════════════════════════════════════════
# LOADER / TOKENIZER
# ════════════════════════════════════════════════════════════════

_TOKENIZER = None

def _get_tokenizer():
    global _TOKENIZER
    if _TOKENIZER is None:
        from transformers import BertTokenizerFast
        _TOKENIZER = BertTokenizerFast.from_pretrained("bert-base-uncased")
    return _TOKENIZER


def _load_checkpoint(path, dtype):
    if path.endswith(".safetensors"):
        from safetensors.torch import load_file
        sd = load_file(path)
    else:
        raw = torch.load(path, map_location="cpu", weights_only=True)
        sd = raw.get("state_dict", raw)

    parts = {"text_encoder": {}, "vae": {}, "unet": {}, "other": {}}
    for k, v in sd.items():
        if not isinstance(v, torch.Tensor):
            continue
        if k.startswith("cond_stage_model.transformer."):
            parts["text_encoder"][k[len("cond_stage_model.transformer."):]] = v
        elif k.startswith("first_stage_model."):
            parts["vae"][k[len("first_stage_model."):]] = v
        elif k.startswith("model.diffusion_model."):
            parts["unet"][k[len("model.diffusion_model."):]] = v
        elif k.startswith("model_ema."):
            pass
        else:
            parts["other"][k] = v

    # Convert model weights to target dtype
    for g in parts.values():
        for k2 in g:
            g[k2] = g[k2].to("cpu", dtype)

    # Noise schedule stays fp32 regardless (numerical stability)
    for k2 in parts["other"]:
        parts["other"][k2] = parts["other"][k2].to(torch.float32)

    return parts


# ════════════════════════════════════════════════════════════════
# NODE: Load Checkpoint
# ════════════════════════════════════════════════════════════════

class LDMCheckpointLoader:
    DESCRIPTION = (
        "Loads a CompVis Latent Diffusion (LDM) checkpoint from ComfyUI/models/ldm.\n\n"
        "This is the original 2021 LDM architecture (Rombach et al.) with a BERT-based "
        "text encoder. The model operates in a 4-channel latent "
        "space with 8× spatial downsampling (f8 VAE).\n\n"
        "Outputs: MODEL / BERT / VAE.\n\n"
        "precision controls compute dtype:\n"
        "• fp32 — full precision (default, most stable);\n"
        "• fp16 — half precision (2× less VRAM, faster on modern GPUs);\n"
        "• bf16 — bfloat16 (better dynamic range than fp16, requires Ampere+).\n\n"
        "memory_mode controls VRAM usage:\n"
        "• keep — all components stay on GPU (fastest, highest VRAM);\n"
        "• auto — keep on GPU when VRAM permits, otherwise offload after stage;\n"
        "• offload — each component lives on GPU only during its stage;\n"
        "• cpu — entire inference runs on CPU (no VRAM required)."
    )

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "ckpt_name": (folder_paths.get_filename_list("ldm"), {
                "tooltip": "Checkpoint file from the ComfyUI/models/ldm directory."}),
            "precision": (["fp32", "fp16", "bf16"], {
                "default": "fp32",
                "tooltip": "Compute precision. fp16 halves VRAM usage and speeds up "
                           "inference on modern GPUs. bf16 offers better numerical "
                           "stability than fp16 but requires Ampere+ (RTX 30xx+)."}),
            "memory_mode": (["auto", "keep", "offload", "cpu"], {
                "tooltip": "VRAM management strategy. 'auto' balances speed and memory; "
                           "'offload' minimizes VRAM; 'cpu' avoids GPU entirely."}),
        }}

    RETURN_TYPES = ("LDM_MODEL", "LDM_BERT", "LDM_VAE")
    RETURN_NAMES = ("MODEL", "BERT", "VAE")
    FUNCTION = "load"
    CATEGORY = "LDM"

    def load(self, ckpt_name, precision, memory_mode):
        device = "cpu" if memory_mode == "cpu" else (
            "cuda" if torch.cuda.is_available() else "cpu")
        offload = "cpu"
        dtype_map = {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}
        dtype = dtype_map[precision]

        if dtype == torch.bfloat16 and device == "cuda":
            cc = torch.cuda.get_device_capability()
            if cc[0] < 8:
                logging.warning(
                    f"bf16 requires Ampere+ GPU (sm_{cc[0]}{cc[1]}); falling back to fp16")
                dtype = torch.float16

        parts = _load_checkpoint(
            folder_paths.get_full_path("ldm", ckpt_name), dtype)

        mm = MemoryManager(strategy=memory_mode, device=device)

        logging.info(f"VAE load device: {device}, offload device: {offload}, dtype: {dtype}")
        vae = _build_vae(dtype)
        m, u = vae.load_state_dict(parts["vae"], strict=False)
        logging.info(f"VAE weights loaded: missing={len(m)} unexpected={len(u)}")

        logging.info(f"BERT/text encoder model load device: {device}, "
                     f"offload device: {offload}, current: cpu, dtype: {dtype}")
        tenc = LDMTextEncoder().to(dtype).eval()
        m, u = tenc.load_state_dict(parts["text_encoder"], strict=False)
        logging.info(f"BERT weights loaded: missing={len(m)} unexpected={len(u)}")

        unet = LDMUNet(ctx_dim=1280, num_heads=8).to(dtype).eval()
        m, u = unet.load_state_dict(parts["unet"], strict=False)
        logging.info(f"UNet weights loaded: missing={len(m)} unexpected={len(u)}")

        logging.info(f"model weight dtype {dtype}, manual cast: None")
        logging.info("model_type EPS (DDPM epsilon-prediction)")

        comp_unet = mm.register(_Component("unet", unet))
        comp_bert = mm.register(_Component("bert", tenc))
        comp_vae  = mm.register(_Component("vae", vae))

        model = {"comp": comp_unet, "mm": mm, "dtype": dtype,
                 "noise_schedule": NoiseSchedule(parts["other"])}
        bert  = {"comp": comp_bert, "mm": mm, "dtype": dtype}
        vae_o = {"comp": comp_vae,  "mm": mm, "dtype": dtype}

        return (model, bert, vae_o)


# ════════════════════════════════════════════════════════════════
# NODE: BERT Text Encode
# ════════════════════════════════════════════════════════════════

class LDMBERTTextEncode:
    DESCRIPTION = (
        "Encodes a text prompt into a conditioning tensor for the UNet.\n\n"
        "Uses the BERT-base-uncased tokenizer (77 tokens max). The encoder is a "
        "64-layer transformer with 1280-dim hidden states and 8 attention heads.\n\n"
        "Prompting tips for this model:\n"
        "• Keep prompts short and concrete (BERT has limited compositional understanding).\n"
        "• Lowercase is fine — the tokenizer is uncased.\n"
        "• Avoid complex syntax, negations, or long narratives.\n"
        "• Single-subject descriptions work best: 'a red car', 'mountain landscape'.\n"
        "• This model was trained without classifier-free guidance — there is no "
        "negative prompt."
    )

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "bert": ("LDM_BERT", {
                "tooltip": "BERT output from the Load LDM Checkpoint node."}),
            "text": ("STRING", {
                "multiline": True, "default": "",
                "tooltip": "Text prompt. BERT tokenizer, 77 tokens max. "
                           "Keep it short and descriptive."}),
        }}

    RETURN_TYPES = ("LDM_CONDITIONING",)
    RETURN_NAMES = ("CONDITIONING",)
    FUNCTION = "encode"
    CATEGORY = "LDM"

    @torch.no_grad()
    def encode(self, bert, text):
        mm = bert["mm"]
        dtype = bert.get("dtype", torch.float32)
        with mm.use(bert["comp"]) as tenc:
            ids = _get_tokenizer()(
                text, padding="max_length", max_length=77,
                truncation=True, return_tensors="pt",
            ).input_ids.to(mm.device)
            cond = tenc(ids)
        return ({"cond": cond.to(dtype).cpu(), "dtype": dtype},)


# ════════════════════════════════════════════════════════════════
# NODE: Empty Latent
# ════════════════════════════════════════════════════════════════

class LDMEmptyLatent:
    DESCRIPTION = (
        "Creates an empty (zero-initialized) latent tensor for text-to-image generation.\n\n"
        "The LDM f8 VAE uses 4 latent channels with 8× spatial downsampling "
        "(e.g. 256×256 pixels → 32×32 latent).\n\n"
        "Recommended resolutions: 256×256 (native training resolution). "
        "Higher resolutions (512, 1024) are possible but may produce artifacts "
        "since the model was trained at 256."
    )

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "width": ("INT", {
                "default": 256, "min": 64, "max": 1024, "step": 64,
                "tooltip": "Image width in pixels. Must be a multiple of 8. "
                           "Native training resolution is 256."}),
            "height": ("INT", {
                "default": 256, "min": 64, "max": 1024, "step": 64,
                "tooltip": "Image height in pixels. Must be a multiple of 8. "
                           "Native training resolution is 256."}),
            "batch_size": ("INT", {
                "default": 1, "min": 1, "max": 8,
                "tooltip": "Number of images to generate in a single pass."}),
        }}

    RETURN_TYPES = ("LDM_LATENT",)
    RETURN_NAMES = ("LATENT",)
    FUNCTION = "generate"
    CATEGORY = "LDM"

    def generate(self, width, height, batch_size):
        return ({"samples": torch.zeros([batch_size, 4, height // 8, width // 8])},)


# ════════════════════════════════════════════════════════════════
# NODE: Sampler
# ════════════════════════════════════════════════════════════════

class LDMSampler:
    DESCRIPTION = (
        "Runs the diffusion sampling loop to denoise the latent.\n\n"
        "Samplers (alpha-space, native to the LDM paper):\n"
        "• ddim — deterministic DDIM (eta controls stochasticity: 0=det, 1≈ddpm);\n"
        "• ddpm — ancestral DDPM (always stochastic);\n"
        "• plms — pseudo-linear multistep.\n"
        "  Compatible schedulers: ddim_uniform, linear, beta.\n\n"
        "Samplers (sigma-space, k-diffusion family):\n"
        "• euler / euler_ancestral / heun / dpmpp_2m.\n"
        "  Compatible schedulers: all (ddim_uniform, linear, beta, karras, "
        "exponential, cosine).\n\n"
        "denoise=1.0 generates from pure noise (txt2img). Values < 1.0 partially "
        "denoise an existing latent (img2img).\n\n"
        "NOTE: This model was trained without classifier-free guidance. "
        "There is no negative prompt and no CFG scale."
    )

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "model": ("LDM_MODEL", {
                "tooltip": "MODEL output from Load LDM Checkpoint."}),
            "positive": ("LDM_CONDITIONING", {
                "tooltip": "CONDITIONING from BERT Text Encode node."}),
            "latent_image": ("LDM_LATENT", {
                "tooltip": "Starting latent: empty (txt2img) or encoded image (img2img)."}),
            "seed": ("INT", {
                "default": 42, "min": 0, "max": 0xffffffffffffffff,
                "tooltip": "Random seed for initial noise."}),
            "steps": ("INT", {
                "default": 50, "min": 1, "max": 1000,
                "tooltip": "Number of denoising steps. k-diffusion samplers converge "
                           "well at 20–30 steps."}),
            "sampler_name": (list(SAMPLER_TABLE.keys()), {
                "tooltip": "ddim/ddpm/plms are alpha-space (LDM paper native). "
                           "euler/heun/dpmpp_2m are sigma-space (k-diffusion)."}),
            "scheduler": (list(SIGMA_SCHEDULERS), {
                "tooltip": "Sigma schedule. Alpha-space samplers (ddim/ddpm/plms) "
                           "accept: ddim_uniform, linear, beta. "
                           "Sigma-space samplers accept all schedulers."}),
            "eta": ("FLOAT", {
                "default": 0.0, "min": 0.0, "max": 1.0, "step": 0.05,
                "tooltip": "Stochasticity for DDIM only. 0 = deterministic, 1 ≈ DDPM. "
                           "Ignored by other samplers."}),
            "denoise": ("FLOAT", {
                "default": 1.0, "min": 0.0, "max": 1.0, "step": 0.05,
                "tooltip": "1.0 = full generation from noise; < 1.0 = partial denoising "
                           "(img2img)."}),
        }}

    RETURN_TYPES = ("LDM_LATENT",)
    RETURN_NAMES = ("LATENT",)
    FUNCTION = "sample"
    CATEGORY = "LDM"

    def sample(self, model, positive, latent_image, seed, steps,
               sampler_name, scheduler, eta, denoise):
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

        mm = model["mm"]
        dtype = model.get("dtype", torch.float32)

        # ── scheduler compatibility check ──
        family, fn = SAMPLER_TABLE[sampler_name]
        valid = ALPHA_SCHEDULERS if family == "alpha" else SIGMA_SCHEDULERS
        if scheduler not in valid:
            fallback = "ddim_uniform" if family == "alpha" else "karras"
            print(f"[LDM] WARNING: scheduler '{scheduler}' is incompatible with "
                  f"sampler '{sampler_name}'. Falling back to '{fallback}'.")
            scheduler = fallback

        with mm.use(model["comp"]) as unet:
            dev = mm.device
            x_in = latent_image["samples"].to(dev, dtype=dtype)
            cond = positive["cond"].to(dev, dtype=dtype)
            if cond.shape[0] != x_in.shape[0]:
                cond = cond.repeat(x_in.shape[0], 1, 1)

            lat = run_sampler(
                sampler_name, scheduler,
                unet, cond, model["noise_schedule"], x_in,
                steps=steps, eta=eta, denoise=denoise,
                device=dev, dtype=dtype)

        return ({"samples": lat.cpu()},)


# ════════════════════════════════════════════════════════════════
# NODE: VAE Decode
# ════════════════════════════════════════════════════════════════

class LDMVAEDecode:
    DESCRIPTION = (
        "Decodes a latent tensor back into pixel space using the f8 VAE decoder.\n\n"
        "The latent is divided by the training scale factor (0.18215 for f8 VAE) "
        "before decoding. If the output appears too dark or washed out, "
        "adjust vae_scale.\n\n"
        "Output: IMAGE tensor in [0, 1] range, BHWC layout."
    )

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "samples": ("LDM_LATENT", {
                "tooltip": "Latent tensor from the Sampler node."}),
            "vae": ("LDM_VAE", {
                "tooltip": "VAE output from Load LDM Checkpoint."}),
            "vae_scale": ("FLOAT", {
                "default": 0.18215, "min": 0.01, "max": 1.0, "step": 0.00001,
                "tooltip": "Latent scaling factor. Standard f8 VAE value is 0.18215. "
                           "Adjust if output brightness looks incorrect."}),
        }}

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("IMAGE",)
    FUNCTION = "decode"
    CATEGORY = "LDM"

    def decode(self, samples, vae, vae_scale):
        mm = vae["mm"]
        dtype = vae.get("dtype", torch.float32)
        with mm.use(vae["comp"]) as v:
            with torch.no_grad():
                z = samples["samples"].to(mm.device, dtype=dtype) / vae_scale
                img = v.decode(z)
        # Always output in fp32 for downstream ComfyUI nodes
        return (img.float().clamp(0, 1).permute(0, 2, 3, 1).cpu(),)


# ════════════════════════════════════════════════════════════════
# NODE: VAE Encode (for img2img)
# ════════════════════════════════════════════════════════════════

class LDMVAEEncode:
    DESCRIPTION = (
        "Encodes an image into latent space using the f8 VAE encoder.\n\n"
        "The input image (expected in [0, 1]) is converted to [-1, 1] range, "
        "passed through the encoder, and the mean of the posterior distribution "
        "is taken (deterministic, no stochastic sampling).\n\n"
        "The resulting latent is multiplied by vae_scale (0.18215) to match "
        "the UNet's operating space. Feed the output into the Sampler with "
        "denoise < 1.0 for img2img workflows."
    )

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "pixels": ("IMAGE", {
                "tooltip": "Input image in [0, 1] range (e.g. from Load Image). "
                           "Conversion to [-1, 1] is handled internally."}),
            "vae": ("LDM_VAE", {
                "tooltip": "VAE output from Load LDM Checkpoint."}),
            "vae_scale": ("FLOAT", {
                "default": 0.18215, "min": 0.01, "max": 1.0, "step": 0.00001,
                "tooltip": "Latent scaling factor. Must match the value used in "
                           "VAE Decode (default 0.18215 for f8 VAE)."}),
        }}

    RETURN_TYPES = ("LDM_LATENT",)
    RETURN_NAMES = ("LATENT",)
    FUNCTION = "encode"
    CATEGORY = "LDM"

    @torch.no_grad()
    def encode(self, pixels, vae, vae_scale):
        mm = vae["mm"]
        dtype = vae.get("dtype", torch.float32)
        with mm.use(vae["comp"]) as v:
            x = (pixels * 2.0 - 1.0).permute(0, 3, 1, 2).to(mm.device, dtype=dtype)
            h = v.encoder(x)
            moments = v.quant_conv(h)
            mean, _logvar = torch.chunk(moments, 2, dim=1)
            z = mean * vae_scale
        return ({"samples": z.cpu()},)


# ════════════════════════════════════════════════════════════════
# NODE: Save Checkpoint (safetensors)
# ════════════════════════════════════════════════════════════════

def _rebuild_other(ns):
    """Reconstructs all 12 noise-schedule keys from alphas_cumprod."""
    ac = ns.alphas_cumprod
    ac_prev = torch.cat([torch.ones(1, dtype=ac.dtype), ac[:-1]])
    betas = 1.0 - ac / ac_prev
    post_var = betas * (1 - ac_prev) / (1 - ac)
    return {
        "betas": betas,
        "alphas_cumprod": ac,
        "alphas_cumprod_prev": ac_prev,
        "sqrt_alphas_cumprod": ac.sqrt(),
        "sqrt_one_minus_alphas_cumprod": (1 - ac).sqrt(),
        "log_one_minus_alphas_cumprod": (1 - ac).log(),
        "sqrt_recip_alphas_cumprod": 1.0 / ac.sqrt(),
        "sqrt_recipm1_alphas_cumprod": (1.0 / ac - 1).sqrt(),
        "posterior_variance": post_var,
        "posterior_log_variance_clipped": post_var.clamp(min=1e-20).log(),
        "posterior_mean_coef1": betas * ac_prev.sqrt() / (1 - ac),
        "posterior_mean_coef2": (1 - ac_prev) * ac.sqrt() / (1 - ac),
    }


class LDMSaveCheckpoint:
    DESCRIPTION = (
        "Saves the full LDM pipeline (UNet + BERT encoder + VAE + noise schedule) "
        "as a single .safetensors file in ComfyUI/models/ldm.\n\n"
        "Keys are written with original CompVis prefixes:\n"
        "• model.diffusion_model.* (UNet)\n"
        "• cond_stage_model.transformer.* (BERT text encoder)\n"
        "• first_stage_model.* (VAE)\n"
        "• 12 noise-schedule tensors (betas, alphas_cumprod, etc.)\n\n"
        "All tensors are saved in the precision the model was loaded with "
        "(fp32 / fp16 / bf16). model_ema weights are not saved."
    )
    OUTPUT_NODE = True

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "model": ("LDM_MODEL", {
                "tooltip": "MODEL output from Load LDM Checkpoint."}),
            "bert": ("LDM_BERT", {
                "tooltip": "BERT output from Load LDM Checkpoint."}),
            "vae": ("LDM_VAE", {
                "tooltip": "VAE output from Load LDM Checkpoint."}),
            "filename": ("STRING", {
                "default": "ldm_f8",
                "tooltip": "Output filename in models/ldm. The .safetensors "
                           "extension is appended automatically."}),
        }}

    RETURN_TYPES = ()
    FUNCTION = "save"
    CATEGORY = "LDM"

    def save(self, model, bert, vae, filename):
        from safetensors.torch import save_file

        dtype = model.get("dtype", torch.float32)
        sd = {}

        for k, v in model["comp"].module.state_dict().items():
            sd["model.diffusion_model." + k] = v.detach().cpu().to(dtype).contiguous()
        for k, v in bert["comp"].module.state_dict().items():
            sd["cond_stage_model.transformer." + k] = v.detach().cpu().to(dtype).contiguous()
        for k, v in vae["comp"].module.state_dict().items():
            sd["first_stage_model." + k] = v.detach().cpu().to(dtype).contiguous()

        other = model.get("other")
        if other is None:
            other = _rebuild_other(model["noise_schedule"])
        for k, v in other.items():
            sd[k] = v.detach().cpu().to(dtype).contiguous()

        if not filename.endswith(".safetensors"):
            filename += ".safetensors"

        dest = os.path.join(folder_paths.get_folder_paths("ldm")[0], filename)
        save_file(sd, dest, metadata={
            "format": "compvis-ldm-f8",
            "source": "comfyui-compvis-ldm",
            "precision": str(dtype),
        })
        print(f"[LDM] saved {dest} ({len(sd)} keys, dtype={dtype})")
        return ()


# ════════════════════════════════════════════════════════════════
# REGISTRY
# ════════════════════════════════════════════════════════════════

NODE_CLASS_MAPPINGS = {
    "LDMCheckpointLoader": LDMCheckpointLoader,
    "LDMBERTTextEncode":   LDMBERTTextEncode,
    "LDMEmptyLatent":      LDMEmptyLatent,
    "LDMSampler":          LDMSampler,
    "LDMVAEDecode":        LDMVAEDecode,
    "LDMVAEEncode":        LDMVAEEncode,
    "LDMSaveCheckpoint":   LDMSaveCheckpoint,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "LDMCheckpointLoader": "🏰 Load LDM Checkpoint",
    "LDMBERTTextEncode":   "📜 LDM BERT Text Encode",
    "LDMEmptyLatent":      "🌫️ LDM Empty Latent",
    "LDMSampler":          "🎲 LDM Sampler",
    "LDMVAEDecode":        "🖼️ LDM VAE Decode",
    "LDMVAEEncode":        "📥 LDM VAE Encode",
    "LDMSaveCheckpoint":   "💾 Save LDM Checkpoint",
}