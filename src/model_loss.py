import torch.nn.functional as F
import torch


def CLIP_loss(logits: torch.Tensor) -> torch.Tensor:
    """
    Calculate a custom cross-entropy loss.

    Args:
    - logits (torch.Tensor): The input tensor containing unnormalized logits.

    Returns:
    - torch.Tensor: The computed custom cross-entropy loss.

    Example:
    >>> logits = torch.rand((batch_size, num_classes))
    >>> loss = CLIP_loss(logits)
    """

    n = logits.shape[1]

    # Create labels tensor
    labels = torch.arange(n)

    # bring logits to cpu
    logits = logits.to("cpu")

    # Calculate cross entropy losses along axis 0 and 1
    loss_i = F.cross_entropy(logits.transpose(0, 1), labels, reduction="mean")
    loss_t = F.cross_entropy(logits, labels, reduction="mean")

    # Calculate the final loss
    loss = (loss_i + loss_t) / 2

    return loss


def metrics(similarity: torch.Tensor):
    y = torch.arange(len(similarity)).to(similarity.device)
    img2cap_match_idx = similarity.argmax(dim=1)
    cap2img_match_idx = similarity.argmax(dim=0)

    img_acc = (img2cap_match_idx == y).float().mean()
    cap_acc = (cap2img_match_idx == y).float().mean()

    return img_acc, cap_acc


# Example usage
# logits = torch.rand(8, 8)  # Simulated logits
# loss = CLIP_loss(logits)
# print(f"Contrastive loss: {loss}")
