"""
Used to generate by calculation the entrophy of the distribution of the Gaussian Primitives.
"""

# uncertainty.py

import torch
import torch.nn.functional as F

@torch.no_grad()
def compute_uncertainty_map(
    alphas: torch.Tensor, 
    n_contrib: torch.Tensor,
    height: int,
    width: int
) -> torch.Tensor:
    """
    Computes the uncertainty map based on the entropy of Gaussian weights per pixel.

    Args:
        alphas (torch.Tensor): A tensor of shape [N_points,], containing the alpha values
                               of all Gaussians that contributed to any pixel.
        n_contrib (torch.Tensor): A tensor of shape [H, W], indicating the number of
                                  contributing Gaussians for each pixel.
        height (int): The height of the output image.
        width (int): The width of the output image.

    Returns:
        torch.Tensor: An uncertainty map (entropy map) of shape [H, W].
    """
    
    # Add a small epsilon for numerical stability to avoid log(0) and division by zero.
    eps = 1e-10

    # The rasterizer outputs a flattened list of alphas. We need to reconstruct the
    # per-pixel list. We can do this using the n_contrib tensor.
    # We create a mask for valid contributions.
    max_contrib = n_contrib.max()
    
    # Create an index tensor to map flattened alphas to a padded [H, W, max_contrib] tensor
    cum_contrib = torch.cumsum(n_contrib.flatten(), dim=0)
    indices = torch.arange(alphas.shape[0], device=alphas.device)
    
    # Find which pixel each alpha belongs to
    pixel_indices = torch.searchsorted(cum_contrib, indices, right=True)
    
    # Find the intra-pixel index for each alpha
    intra_pixel_indices = indices - torch.cat([torch.tensor([0], device=alphas.device), cum_contrib[:-1]])[pixel_indices]

    # Create the padded alpha tensor
    padded_alphas = torch.zeros(height, width, max_contrib, device=alphas.device)
    
    # Unflatten pixel indices to 2D
    pixel_coords_h = pixel_indices // width
    pixel_coords_w = pixel_indices % width
    
    # Use advanced indexing to fill the padded tensor
    padded_alphas[pixel_coords_h, pixel_coords_w, intra_pixel_indices] = alphas
    
    # Create a mask to ignore padded values
    mask = (torch.arange(max_contrib, device=alphas.device)[None, None, :] < n_contrib[..., None]).float()

    # Calculate transmittance T_i = product_{j=1}^{i-1} (1 - alpha_j)
    # The first Gaussian has a transmittance of 1.
    transmittance = torch.cumprod(
        torch.cat([torch.ones(height, width, 1, device=alphas.device), 1. - padded_alphas + eps], dim=-1), 
        dim=-1
    )[:, :, :-1]

    # Calculate weights w_i = T_i * alpha_i
    weights = transmittance * padded_alphas * mask
    
    # Normalize weights per pixel to get a probability distribution P
    # Sum of weights can be zero for pixels with no contributions.
    weights_sum = torch.sum(weights, dim=-1, keepdim=True)
    
    # Probability P = weights / sum(weights)
    probabilities = weights / (weights_sum + eps)
    
    # Calculate entropy: H(P) = - sum(P * log(P))
    # We multiply by the mask again to ensure padded values do not contribute.
    entropy = -torch.sum(probabilities * torch.log(probabilities + eps) * mask, dim=-1)
    
    return entropy