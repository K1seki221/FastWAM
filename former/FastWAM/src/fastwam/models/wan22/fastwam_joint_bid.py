import torch

from .fastwam_joint import FastWAMJoint


class FastWAMJointBid(FastWAMJoint):
    """Joint variant with bidirectional attention between future video and action tokens."""

    @torch.no_grad()
    def _build_mot_attention_mask(
        self,
        video_seq_len: int,
        action_seq_len: int,
        video_tokens_per_frame: int,
        device: torch.device,
    ) -> torch.Tensor:
        mask = super()._build_mot_attention_mask(
            video_seq_len=video_seq_len,
            action_seq_len=action_seq_len,
            video_tokens_per_frame=video_tokens_per_frame,
            device=device,
        )

        # future video -> action. Keep the conditioned first-frame queries
        # independent of action while making future video/action bidirectional.
        first_frame_tokens = min(video_tokens_per_frame, video_seq_len)
        mask[first_frame_tokens:video_seq_len, video_seq_len:] = True
        return mask
