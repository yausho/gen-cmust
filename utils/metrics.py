import torch
import numpy as np

def masked_mse(preds, labels, null_val=np.nan):
    if np.isnan(null_val):
        mask = ~torch.isnan(labels)
    else:
        mask = (labels != null_val)
    mask = mask.float()
    mask /= torch.mean(mask).clamp(min=1e-8)
    mask = torch.where(torch.isnan(mask), torch.zeros_like(mask), mask)
    loss = (preds - labels) ** 2
    loss = loss * mask
    loss = torch.where(torch.isnan(loss), torch.zeros_like(loss), loss)
    return torch.mean(loss)

def masked_rmse(preds, labels, null_val=np.nan):
    if np.isnan(null_val):
        mask = ~torch.isnan(labels)
    else:
        mask = (labels != null_val)
    mask = mask.float()
    mask /= torch.mean(mask).clamp(min=1e-8)
    mask = torch.where(torch.isnan(mask), torch.zeros_like(mask), mask)
    loss = (preds - labels) ** 2
    loss = loss * mask
    loss = torch.where(torch.isnan(loss), torch.zeros_like(loss), loss)
    return torch.sqrt(torch.mean(loss))

def masked_mae(preds, labels, null_val=np.nan):
    if np.isnan(null_val):
        mask = ~torch.isnan(labels)
    else:
        mask = (labels != null_val)
    mask = mask.float()
    mask /= torch.mean(mask).clamp(min=1e-8)
    mask = torch.where(torch.isnan(mask), torch.zeros_like(mask), mask)
    loss = torch.abs(preds - labels)
    loss = loss * mask
    loss = torch.where(torch.isnan(loss), torch.zeros_like(loss), loss)
    return torch.mean(loss)

def masked_mape(preds, labels, null_val=np.nan, threshold=0.01):
    if np.isnan(null_val):
        mask = ~torch.isnan(labels)
    else:
        mask = (labels != null_val)
    # Exclude near-zero nodes to avoid long-tail distortion.
    if threshold > 0:
        mask = mask & (labels.abs() >= threshold)
    mask = mask.float()
    mask /= torch.mean(mask).clamp(min=1e-8)
    mask = torch.where(torch.isnan(mask), torch.zeros_like(mask), mask)

    safe_labels = torch.clamp(labels.abs(), min=1e-3)
    loss = torch.abs(preds - labels) / safe_labels

    loss = loss * mask
    loss = torch.where(torch.isnan(loss), torch.zeros_like(loss), loss)
    return torch.mean(loss)

def metric(preds, labels, null_val=np.nan):
    """
    统一指标调用接口
    """
    mae = masked_mae(preds, labels, null_val)
    rmse = masked_rmse(preds, labels, null_val)
    mape = masked_mape(preds, labels, null_val)
    return mae, rmse, mape


def masked_ssim(preds, labels, global_mask=None, data_range=None):
    """
    Spatial Structural Similarity Index (SSIM) averaged over all samples and
    time steps.  Operates on the 2D spatial grid (H, W) so it captures the
    topological fidelity of the prediction — complementary to point-wise
    MAE/RMSE.

    Args:
        preds:  (N, T, C, H, W) or (N, C, H, W)  numpy array / tensor
        labels: same shape as preds
        global_mask: optional (1,1,1,H,W) valid-region mask from auto_pad_to_32.
                     Padding pixels are zeroed out before SSIM to prevent them
                     from inflating structural similarity.
        data_range:  if None, estimated as labels.max() - labels.min().
    """
    if isinstance(preds, torch.Tensor):
        preds = preds.numpy()
    if isinstance(labels, torch.Tensor):
        labels = labels.numpy()

    preds = preds.astype(np.float64)
    labels = labels.astype(np.float64)

    # Zero out padding region so it doesn't inflate SSIM.
    if global_mask is not None:
        if isinstance(global_mask, torch.Tensor):
            global_mask = global_mask.cpu().numpy()
        mask_broadcast = global_mask.astype(np.float64)
        # Broadcast to match preds shape (handles 4D and 5D).
        preds = preds * mask_broadcast
        labels = labels * mask_broadcast

    if data_range is None:
        data_range = float(labels.max() - labels.min())
        if data_range < 1e-8:
            return 1.0  # identical constant fields → perfect similarity

    # Flatten to (num_frames, 1, H, W) for uniform 2D processing.
    if preds.ndim == 5:  # (N, T, C, H, W)
        N, T, C, H, W = preds.shape
        preds = preds.reshape(N * T * C, 1, H, W)
        labels = labels.reshape(N * T * C, 1, H, W)
    elif preds.ndim == 4:  # (N, C, H, W)
        N, C, H, W = preds.shape
        preds = preds.reshape(N * C, 1, H, W)
        labels = labels.reshape(N * C, 1, H, W)
    else:
        return 0.0  # unfamiliar shape after masking; cannot compute spatial SSIM

    # --- Pure-numpy SSIM (no extra dependency) ---
    # Uses the simplified form without gaussian weighting; window = min(7, H, W).
    k1, k2 = 0.01, 0.03
    C1 = (k1 * data_range) ** 2
    C2 = (k2 * data_range) ** 2
    win = min(7, H, W)
    if win < 2:
        # Spatial grid too small for meaningful SSIM; return simple correlation.
        return 1.0 if np.allclose(preds, labels) else 0.0

    # Uniform box filter via cumulative sum (fast, no scipy needed).
    def _box_filter_2d(img, r):
        """Mean filter with kernel size r on last two dims of (M, 1, H, W)."""
        # Pad-reflect to avoid border artifacts.
        out = img
        # cumsum along H
        pad_h = np.pad(out, ((0,0),(0,0),(r//2, r - r//2),(0,0)), mode='reflect')
        cs = np.cumsum(pad_h, axis=2)
        out = (cs[:, :, r:, :] - cs[:, :, :-r, :]) / r
        # cumsum along W
        pad_w = np.pad(out, ((0,0),(0,0),(0,0),(r//2, r - r//2)), mode='reflect')
        cs = np.cumsum(pad_w, axis=3)
        out = (cs[:, :, :, r:] - cs[:, :, :, :-r]) / r
        return out

    mu_x = _box_filter_2d(preds, win)
    mu_y = _box_filter_2d(labels, win)
    mu_x_sq = mu_x ** 2
    mu_y_sq = mu_y ** 2
    mu_xy = mu_x * mu_y
    sigma_x_sq = _box_filter_2d(preds ** 2, win) - mu_x_sq
    sigma_y_sq = _box_filter_2d(labels ** 2, win) - mu_y_sq
    sigma_xy = _box_filter_2d(preds * labels, win) - mu_xy

    # Clamp negative variance from numerical precision.
    sigma_x_sq = np.maximum(sigma_x_sq, 0.0)
    sigma_y_sq = np.maximum(sigma_y_sq, 0.0)

    ssim_map = ((2 * mu_xy + C1) * (2 * sigma_xy + C2)) / \
               ((mu_x_sq + mu_y_sq + C1) * (sigma_x_sq + sigma_y_sq + C2))

    return float(np.mean(ssim_map))