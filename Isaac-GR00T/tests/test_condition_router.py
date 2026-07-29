# SPDX-License-Identifier: Apache-2.0
"""CPU smoke tests for the condition router (v1 static)."""

import torch

from gr00t.configs.model.gr00t_n1d7 import Gr00tN1d7Config
from gr00t.model.gr00t_n1d7.gr00t_n1d7 import ConditionRouter, Gr00tN1d7ActionHead
from gr00t.model.modules.dit import AlternateVLDiT

B, S, D = 2, 17, 2048


def _small_dit():
    torch.manual_seed(0)
    return AlternateVLDiT(
        num_layers=4,
        num_attention_heads=4,
        attention_head_dim=8,
        output_dim=16,
        norm_type="ada_norm",
        interleave_self_attention=True,
        cross_attention_dim=D,
        attend_text_every_n_blocks=2,
    ).eval()


def test_dit_per_block_equals_shared():
    """4-D encoder states with identical per-block tensors == stock 3-D path."""
    dit = _small_dit()
    sa = torch.randn(B, 5, 4 * 8)
    enc = torch.randn(B, S, D)
    image_mask = torch.zeros(B, S, dtype=torch.bool)
    image_mask[:, S // 2 :] = True
    attn_mask = torch.ones(B, S, dtype=torch.bool)
    t = torch.zeros(B, dtype=torch.long)
    with torch.no_grad():
        out_shared = dit(sa, enc, timestep=t, image_mask=image_mask, backbone_attention_mask=attn_mask)
        num_cross = (4 + 1) // 2
        enc_stacked = enc.unsqueeze(1).expand(B, num_cross, S, D)
        out_per_block = dit(
            sa, enc_stacked, timestep=t, image_mask=image_mask, backbone_attention_mask=attn_mask
        )
    # Strided per-block slices can take a different SDPA kernel path than the
    # contiguous shared tensor -> tiny float noise, not a semantic difference.
    assert torch.allclose(out_shared, out_per_block, atol=1e-5)


def test_router_identity_at_high_bias():
    """With a huge incumbent bias, routing == LayerNorm of the deepest candidate."""
    router = ConditionRouter(num_cross_blocks=8, num_candidates=13, dim=D, init_bias=40.0)
    feats = torch.randn(B, 13, S, D)
    out = router(feats)
    assert out.shape == (B, 8, S, D)
    expected = router.norms[-1](feats[:, -1])
    assert torch.allclose(out[:, 0], expected, atol=1e-5)
    assert torch.allclose(out[:, 7], expected, atol=1e-5)


def test_router_pool_and_stats():
    router = ConditionRouter(num_cross_blocks=8, num_candidates=3, dim=D, init_bias=4.0)
    assert len(router.norms) == 3
    assert router.logits.shape == (8, 3)
    stats = router.mixture_stats()
    assert stats["router_weights"].shape == (8, 3)
    w = stats["router_weights"]
    assert torch.allclose(w.sum(-1), torch.ones(8))
    assert (w[:, -1] > 0.8).all()  # incumbent dominates at init
    assert stats["router_entropy"].ndim == 0


def test_action_head_routed_process_backbone_output():
    from transformers.feature_extraction_utils import BatchFeature

    cfg = Gr00tN1d7Config(use_condition_router=True, router_candidate_layers=[0, 6, 12])
    head = Gr00tN1d7ActionHead(cfg).eval()
    head.init_condition_router_from_vlln()
    assert torch.equal(head.condition_router.norms[-1].weight, head.vlln.weight)
    num_cross = (cfg.diffusion_model_cfg["num_layers"] + 1) // 2
    bo = BatchFeature(
        data={
            "backbone_features": torch.randn(B, S, D),
            "backbone_features_all": torch.randn(B, 3, S, D),
        }
    )
    out = head.process_backbone_output(bo)
    assert out["backbone_features"].shape == (B, num_cross, S, D)


def test_router_forward_backward_gradients():
    """Full head forward + backward with router: loss is finite, gradients reach
    the router logits, every candidate norm, and the DiT."""
    from transformers.feature_extraction_utils import BatchFeature

    torch.manual_seed(0)
    cfg = Gr00tN1d7Config(use_condition_router=True, router_candidate_layers=[0, 6, 12])
    head = Gr00tN1d7ActionHead(cfg)
    head.train()
    head.init_condition_router_from_vlln()

    K = 3
    backbone_output = BatchFeature(
        data={
            "backbone_features": torch.randn(B, S, D),
            "backbone_features_all": torch.randn(B, K, S, D),
            "backbone_attention_mask": torch.ones(B, S, dtype=torch.bool),
            "image_mask": torch.cat(
                [torch.zeros(B, S // 2, dtype=torch.bool), torch.ones(B, S - S // 2, dtype=torch.bool)],
                dim=1,
            ),
        }
    )
    action_input = BatchFeature(
        data={
            "state": torch.randn(B, 1, cfg.max_state_dim),
            "action": torch.randn(B, cfg.action_horizon, cfg.max_action_dim),
            "action_mask": torch.ones(B, cfg.action_horizon, cfg.max_action_dim),
            "embodiment_id": torch.zeros(B, dtype=torch.long),
        }
    )
    out = head(backbone_output, action_input)
    loss = out["loss"]
    assert torch.isfinite(loss), "loss is not finite"
    assert "router_entropy" in out and "router_weights" in out
    loss.backward()

    r = head.condition_router
    assert r.logits.grad is not None and r.logits.grad.abs().sum() > 0, "no grad on router logits"
    for k, norm in enumerate(r.norms):
        assert norm.weight.grad is not None and norm.weight.grad.abs().sum() > 0, f"no grad on norm {k}"
    dit_grads = [p.grad for p in head.model.parameters() if p.grad is not None]
    assert len(dit_grads) > 0 and any(g.abs().sum() > 0 for g in dit_grads), "no grad in DiT"
    # vlln must NOT receive gradients on the routed path (it is bypassed).
    assert head.vlln.weight.grad is None or head.vlln.weight.grad.abs().sum() == 0
