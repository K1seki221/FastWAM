# This is a very awesome vision encoder
# you give the name of the encoder, and the dim you want to project to
# it will return the encoder and corresponding projector
# the output of the encoder is a tensor of shape [batch_size, seq_len, dim]

from typing import Optional

import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from transformers import AutoModel, AutoProcessor

# Method prefixes that select alternative compression pipelines on top of
# SigLIP / SigLIP-2 backbones. Anything before the first ":" is treated as a
# method tag; the rest is the underlying HF model id passed to AutoModel.
_KNOWN_METHOD_PREFIXES = ('groot_pixel_shuffle', 'perceiver', 'direct', 'dino')


def _parse_method(model_id: str) -> tuple[str, str]:
    """Split a `vision_encoder` string into ``(method, real_model_id)``.

    - ``"groot_pixel_shuffle:google/siglip2-so400m-patch14-224"`` ->
      ``("groot_pixel_shuffle", "google/siglip2-so400m-patch14-224")``
    - ``"timm/vit_so400m_patch14_siglip_384"`` -> ``("siglip", same)``
    - ``"microsoft/resnet-152"`` -> ``("resnet", same)``
    """
    if ':' in model_id:
        prefix, rest = model_id.split(':', 1)
        if prefix in _KNOWN_METHOD_PREFIXES:
            return prefix, rest
    lowered = model_id.lower()
    if 'siglip' in lowered:
        return 'siglip', model_id
    if 'resnet' in lowered:
        return 'resnet', model_id
    raise ValueError(f'Unsupported encoder: {model_id}')


class PerceiverResampler(nn.Module):
    """Flamingo-style Perceiver Resampler.

    A small stack of cross-attention + FFN blocks that compresses a variable
    number of input tokens into ``num_queries`` learned latents. Inside each
    block the keys/values are formed by concatenating the input tokens with
    the current latents (Flamingo trick) so the latents can also attend to
    each other.
    """

    def __init__(
        self,
        dim: int,
        depth: int = 2,
        num_heads: int = 8,
        head_dim: int = 64,
        num_queries: int = 64,
        ff_mult: int = 4,
    ):
        super().__init__()
        inner_dim = num_heads * head_dim
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.scale = head_dim ** -0.5

        self.latents = nn.Parameter(torch.randn(num_queries, dim) * 0.02)

        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(
                nn.ModuleList(
                    [
                        nn.LayerNorm(dim),
                        nn.LayerNorm(dim),
                        nn.Linear(dim, inner_dim, bias=False),
                        nn.Linear(dim, inner_dim * 2, bias=False),
                        nn.Linear(inner_dim, dim, bias=False),
                        nn.LayerNorm(dim),
                        nn.Linear(dim, dim * ff_mult, bias=False),
                        nn.Linear(dim * ff_mult, dim, bias=False),
                    ]
                )
            )
        self.norm = nn.LayerNorm(dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """``x``: ``[B, N, D]`` -> latents ``[B, num_queries, D]``."""
        b = x.shape[0]
        latents = self.latents.unsqueeze(0).expand(b, -1, -1).to(x.dtype)

        for norm_x, norm_l, to_q, to_kv, to_out, norm_ff, ff1, ff2 in self.layers:
            xn = norm_x(x)
            ln = norm_l(latents)
            kv_input = torch.cat([xn, ln], dim=1)

            q = to_q(ln)
            k, v = to_kv(kv_input).chunk(2, dim=-1)

            q = q.reshape(b, -1, self.num_heads, self.head_dim).transpose(1, 2)
            k = k.reshape(b, -1, self.num_heads, self.head_dim).transpose(1, 2)
            v = v.reshape(b, -1, self.num_heads, self.head_dim).transpose(1, 2)

            attn = (q @ k.transpose(-2, -1)) * self.scale
            attn = attn.softmax(dim=-1)
            out = attn @ v
            out = out.transpose(1, 2).reshape(b, -1, self.num_heads * self.head_dim)
            latents = latents + to_out(out)

            latents = latents + ff2(F.gelu(ff1(norm_ff(latents))))

        return self.norm(latents)


class VisionEncoder(nn.Module):
    """
    VisionEncoder: A flexible vision encoder that supports multiple backbones

    Built-in pipelines (selected by ``model_id``):

    - ``microsoft/resnet-...``                      -> ResNet-152 global features
    - ``timm/vit_so400m_patch14_siglip_384``        -> SigLIP via timm (384x384)
    - ``<hf_id_containing_'siglip'>``               -> SigLIP via HF transformers
    - ``groot_pixel_shuffle:<hf_id>``               -> SigLIP-2 So400m 224 + 2x2 PixelShuffle (256 -> 64 tokens)
    - ``perceiver:<hf_id>``                         -> SigLIP So400m 224 + Perceiver Resampler (256 -> 64 tokens)
    - ``direct:<hf_id>``                            -> SigLIP-2 Base patch32 256 (64 tokens, no compression)
    - ``dino:<timm_model_id>``                      -> DINO v1 ViT via timm (224x224, 196 tokens, e.g. ViT-B/16 768-dim)
    """

    def __init__(self, model_id: str, output_dim: int, history=None):
        """``history``: optional VisionHistoryConfig (enabled) — installs the
        relative-age time embedding for multi-frame (T>1) inputs."""
        super().__init__()
        self.model_id = model_id
        self.output_dim = output_dim

        # vision_history: zero-init additive per-frame time tag. Zero-init is
        # safe because the term is additive on full-content tokens (full
        # gradient from step 0), unlike multiplicative zero-init gates.
        # - learned_table: nn.Embedding indexed by SLOT age (0 = current).
        # - sincos_mlp: sinusoid of the TRUTHFUL per-frame age in raw steps
        #   (post-jitter, post-clamp; batch key 'video_delta_indices') -> MLP.
        self.history_mode: Optional[str] = None
        self.time_embed: Optional[nn.Embedding] = None
        if history is not None and getattr(history, 'enabled', False):
            assert history.time_embed in ('learned_table', 'sincos_mlp'), (
                f'vision_history.time_embed="{history.time_embed}" not implemented'
            )
            self.history_mode = history.time_embed
            self._max_frames = history.max_frames
            if history.time_embed == 'learned_table':
                self.time_embed = nn.Embedding(history.max_frames, output_dim)
                nn.init.zeros_(self.time_embed.weight)
            else:
                from models.common.moe import SinusoidalPositionalEncoding

                self.time_sincos = SinusoidalPositionalEncoding(64)
                self.time_mlp = nn.Sequential(
                    nn.Linear(64, output_dim),
                    nn.SiLU(),
                    nn.Linear(output_dim, output_dim),
                )
                nn.init.zeros_(self.time_mlp[-1].weight)
                nn.init.zeros_(self.time_mlp[-1].bias)

        self.method, self._real_model_id = _parse_method(model_id)

        if self.method == 'groot_pixel_shuffle':
            self._init_groot_pixel_shuffle()
        elif self.method == 'perceiver':
            self._init_perceiver()
        elif self.method == 'direct':
            self._init_direct_patch32()
        elif self.method == 'siglip':
            self._init_siglip()
        elif self.method == 'resnet':
            self._init_resnet()
        elif self.method == 'dino':
            self._init_dino()
        else:
            raise ValueError(f'Unsupported encoder: {model_id}')

    def _init_siglip(self):
        """Initialize SigLIP model"""
        # Load SigLIP model and processor
        if 'timm' in self.model_id.lower():
            self.is_timm_model = True
            self.model_id = self.model_id.split('/')[-1]
            self.encoder = timm.create_model(
                self.model_id,
                pretrained=True,
                num_classes=0,
            )
            self.data_cfg = timm.data.resolve_model_data_config(self.encoder)
            default_image_transform = timm.data.create_transform(**self.data_cfg, is_training=False)
            self.processor = default_image_transform.transforms[3]
        else:
            self.is_timm_model = False
            self.encoder = AutoModel.from_pretrained(self.model_id, torch_dtype=torch.bfloat16, trust_remote_code=True)
            self.processor = AutoProcessor.from_pretrained(self.model_id)

        self._hidden_size = 1152

        # Create projector if needed
        if self.output_dim != self._hidden_size:
            self.projector = nn.Linear(self._hidden_size, self.output_dim)
        else:
            self.projector = nn.Identity()

        # Set default image resolution for SigLIP
        self._default_image_resolution = (3, 384, 384)

        # Calculate number of patches (for SigLIP base patch16-224: 14x14 = 196)
        self._num_patches = (384 // 14) ** 2

    def _init_resnet(self):
        """Initialize ResNet model"""
        # self.encoder = AutoFeatureExtractor.from_pretrained(self.model_id)
        self.encoder = torchvision.models.resnet152(pretrained=True)
        self.encoder = torch.nn.Sequential(*(list(self.encoder.children())[:-2]))
        # ResNet typically outputs 2048 features for ResNet-50/101/152
        self._hidden_size = 2048

        # Create projector
        if self.output_dim != self._hidden_size:
            self.projector = nn.Linear(self._hidden_size, self.output_dim)
        else:
            self.projector = nn.Identity()

        # Set default image resolution for ResNet
        self._default_image_resolution = (3, 224, 224)
        self._num_patches = 1  # ResNet outputs global features

    def _init_dino(self):
        """Initialize DINO v1 ViT-B/16 via timm.

        Output per camera: ``[B, 196, output_dim]`` from a 14x14 patch grid
        at 224x224 input resolution (patch size 16, 86M params).
        """
        self.encoder = timm.create_model(self._real_model_id, pretrained=True, num_classes=0)
        self._hidden_size = self.encoder.num_features  # 768 for ViT-B
        self._default_image_resolution = (3, 224, 224)
        self._num_patches = (224 // 16) ** 2  # 196

        if self.output_dim != self._hidden_size:
            self.projector = nn.Linear(self._hidden_size, self.output_dim)
        else:
            self.projector = nn.Identity()

    def _load_hf_vision_model(self, hf_id: str) -> nn.Module:
        """Load a HF SigLIP/SigLIP-2 model and return only its vision tower.
        """
        full_model = AutoModel.from_pretrained(
            hf_id, torch_dtype=torch.bfloat16, trust_remote_code=True
        )
        vision_model = (
            full_model.vision_model
            if hasattr(full_model, 'vision_model')
            else full_model
        )
        if hasattr(vision_model, 'head'):
            vision_model.head = nn.Identity()
        return vision_model

    def _init_groot_pixel_shuffle(self):
        """GR00T-style Pixel Shuffle compression on SigLIP-2 So400m 224.

        Output per camera: ``[B, 64, output_dim]`` from a 16x16 patch grid.
        """
        self.encoder = self._load_hf_vision_model(self._real_model_id)
        self._hidden_size = 1152
        self._grid = 16  # 224 / 14
        self._num_patches = (self._grid // 2) ** 2  # 64 after 2x2 shuffle
        self._default_image_resolution = (3, 224, 224)

        # PixelShuffle 2x2 packs 4 spatial tokens into the channel dim,
        # then a single linear projects 4*1152 -> output_dim.
        self.pixel_shuffle_proj = nn.Linear(self._hidden_size * 4, self.output_dim)
        # Final projector is identity because pixel_shuffle_proj already produced output_dim.
        self.projector = nn.Identity()

    def _init_perceiver(self):
        """pi0-style Perceiver Resampler on SigLIP So400m 224.

        Output per camera: ``[B, 64, output_dim]``.
        """
        self.encoder = self._load_hf_vision_model(self._real_model_id)
        self._hidden_size = 1152
        self._grid = 16
        self._num_patches = 64
        self._default_image_resolution = (3, 224, 224)

        self.resampler = PerceiverResampler(
            dim=self._hidden_size,
            depth=2,
            num_heads=8,
            head_dim=64,
            num_queries=64,
            ff_mult=4,
        )

        if self.output_dim != self._hidden_size:
            self.projector = nn.Linear(self._hidden_size, self.output_dim)
        else:
            self.projector = nn.Identity()

    def _init_direct_patch32(self):
        """Direct SigLIP-2 Base patch32 at 256 -> 64 tokens, no compression."""
        self.encoder = self._load_hf_vision_model(self._real_model_id)
        self._hidden_size = 768  # SigLIP-2 Base
        self._grid = 8  # 256 / 32
        self._num_patches = 64
        self._default_image_resolution = (3, 256, 256)

        if self.output_dim != self._hidden_size:
            self.projector = nn.Linear(self._hidden_size, self.output_dim)
        else:
            self.projector = nn.Identity()

    def forward(
        self, pixel_values: torch.Tensor, frame_offsets: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass through the vision encoder.

        Args:
            pixel_values: Collated action-head images ``[B, T, V, C, H, W]``
                (T = 1 + history frames ordered oldest -> current; V = cameras).
            frame_offsets: ``[B, T]`` truthful per-frame offsets in raw steps
                (<= 0, 0 = current frame). Required for ``sincos_mlp``;
                ignored by ``learned_table``.

        Returns:
            ``[B, T*V*P, output_dim]`` tokens, frame-major with cameras
            concatenated inside each frame — identical to the legacy
            ``[B, V*P, output_dim]`` layout when T == 1.
        """
        b, t, v = pixel_values.shape[:3]
        if self.history_mode is None:
            assert t == 1, (
                f'vision encoder got {t} frames but vision_history is not enabled — '
                'time-anonymous multi-frame conditions are almost certainly a '
                'misconfiguration (dataset.temporal_sampling.history without '
                'heads.action_predict.vision_history)'
            )
        else:
            assert t <= self._max_frames, (
                f'{t} frames exceed vision_history.max_frames={self._max_frames}'
            )

        # The trunk is time-blind: every frame/camera is encoded identically.
        flat = pixel_values.reshape(b * t * v, *pixel_values.shape[3:])
        features = self._forward_single_camera(flat)             # [B*T*V, P, hidden]
        p = features.shape[1]
        features = features.reshape(b, t, v * p, features.shape[-1])
        features = self.projector(features)                      # [B, T, V*P, output_dim]

        if self.history_mode == 'learned_table':
            # Slot-age tag: age 0 = current (last) frame, age k = k slots
            # back. Frames arrive oldest-first from the sampler.
            assert self.time_embed is not None
            ages = torch.arange(t - 1, -1, -1, device=features.device)
            features = features + self.time_embed(ages)[None, :, None, :]
        elif self.history_mode == 'sincos_mlp':
            assert frame_offsets is not None, (
                'vision_history.time_embed=sincos_mlp needs the batch key '
                "'video_delta_indices' (truthful per-frame offsets)"
            )
            # Truthful age in raw steps (>= 0): clamped episode-start frames
            # carry their REAL dt (e.g. [-2,-2,-2,0] at step 2 -> ages [2,2,2,0]).
            ages = (-frame_offsets).to(dtype=features.dtype, device=features.device)
            emb = self.time_mlp(self.time_sincos(ages).to(features.dtype))  # [B, T, D]
            features = features + emb[:, :, None, :]

        return features.reshape(b, t * v * p, -1)

    def _forward_single_camera(self, pixel_values: torch.Tensor) -> torch.Tensor:
        if self.method == 'groot_pixel_shuffle':
            return self._forward_groot_pixel_shuffle(pixel_values)
        if self.method == 'perceiver':
            return self._forward_perceiver(pixel_values)
        if self.method == 'direct':
            return self._forward_direct_patch32(pixel_values)
        if self.method == 'siglip':
            return self._forward_siglip(pixel_values)
        if self.method == 'resnet':
            return self._forward_resnet(pixel_values)
        if self.method == 'dino':
            return self._forward_dino(pixel_values)
        raise ValueError(f'Unsupported encoder: {self.model_id}')

    def _forward_siglip(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """Forward pass for SigLIP"""
        # Handle different SigLIP implementations
        if self.is_timm_model:
            # For timm models, forward pass returns patch embeddings directly
            # timm ViT models return [batch_size, num_patches + 1, embed_dim] by default
            features = self.encoder.forward_features(pixel_values)
            # Remove CLS token to get [batch_size, num_patches, embed_dim]
            features = features[:, 1:, :]
        else:
            # For Hugging Face models, use output_hidden_states
            outputs = self.encoder(pixel_values, output_hidden_states=True)
            # Get patch embeddings (excluding CLS token)
            # For SigLIP, the last_hidden_state contains [CLS, patch1, patch2, ..., patchN]
            # We want to return [patch1, patch2, ..., patchN] to get [bs, seqlen, 1152]
            features = outputs.last_hidden_state[:, 1:, :]  # Remove CLS token

        return features

    def _forward_resnet(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """Forward pass for ResNet"""
        # ResNet forward pass
        features = self.encoder(pixel_values)

        # ResNet outputs global features, add sequence dimension
        if len(features.shape) == 2:
            features = features.unsqueeze(1)
        bs, hidden_dim, h, w = features.shape
        features = features.reshape(bs, hidden_dim, h * w)
        features = features.permute(0, 2, 1)

        return features

    def _forward_dino(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """Forward pass for DINO v1 ViT via timm.

        timm returns ``[B, N+1, D]`` where index 0 is the CLS token; we strip
        it to yield ``[B, 196, 768]`` for ViT-B/16 at 224x224.
        """
        features = self.encoder.forward_features(pixel_values)
        return features[:, 1:, :]

    def _hf_siglip_patch_features(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """Run a HF SigLIP/SigLIP-2 vision model and return patch tokens.

        HF SigLIP vision models do not prepend a CLS token, so we keep the
        full ``last_hidden_state``.
        """
        outputs = self.encoder(pixel_values=pixel_values)
        return outputs.last_hidden_state

    def _forward_groot_pixel_shuffle(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """SigLIP-2 features -> 2x2 spatial pixel shuffle -> linear -> 64 tokens."""
        features = self._hf_siglip_patch_features(pixel_values)  # [B, 256, 1152]
        b, n, d = features.shape
        g = self._grid
        assert n == g * g, f'Expected {g * g} patches, got {n}'
        # Group each 2x2 spatial block into the channel dim:
        # [B, g, g, D] -> [B, g/2, 2, g/2, 2, D] -> [B, g/2, g/2, 4*D] -> [B, 64, 4D]
        features = features.reshape(b, g // 2, 2, g // 2, 2, d)
        features = features.permute(0, 1, 3, 2, 4, 5).contiguous()
        features = features.reshape(b, (g // 2) * (g // 2), 4 * d)
        features = self.pixel_shuffle_proj(features)
        return features

    def _forward_perceiver(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """SigLIP features -> Perceiver Resampler -> 64 tokens."""
        features = self._hf_siglip_patch_features(pixel_values)  # [B, 256, 1152]
        features = self.resampler(features)
        return features

    def _forward_direct_patch32(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """SigLIP-2 Base patch32 at 256: already 64 tokens, no compression."""
        features = self._hf_siglip_patch_features(pixel_values)  # [B, 64, 768]
        return features
