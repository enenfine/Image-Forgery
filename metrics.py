# src/metrics.py
import numpy as np
import json
import scipy.optimize
import numba

@numba.jit(nopython=True)
def _rle_encode_jit(x: np.ndarray, fg_val: int = 1) -> list[int]:
    dots = np.where(x.T.flatten() == fg_val)[0]
    run_lengths = []
    prev = -2
    for b in dots:
        if b > prev + 1:
            run_lengths.extend((b + 1, 0))
        run_lengths[-1] += 1
        prev = b
    return run_lengths


def rle_encode_mask(mask: np.ndarray, fg_val: int = 1) -> str:
    return json.dumps(_rle_encode_jit(mask, fg_val))


@numba.njit
def _rle_decode_jit(mask_rle: np.ndarray, height: int, width: int) -> np.ndarray:
    if len(mask_rle) % 2 != 0:
        raise ValueError('One or more rows has an odd number of values.')
    starts, lengths = mask_rle[0::2], mask_rle[1::2]
    starts -= 1
    ends = starts + lengths
    for i in range(len(starts) - 1):
        if ends[i] > starts[i + 1]:
            raise ValueError('Pixels must not be overlapping.')
    img = np.zeros(height * width, dtype=np.bool_)
    for lo, hi in zip(starts, ends):
        img[lo:hi] = 1
    return img


def rle_decode_mask(mask_rle: str, shape: tuple[int, int]) -> np.ndarray:
    mask_rle = json.loads(mask_rle)
    mask_rle = np.asarray(mask_rle, dtype=np.int32)
    starts = mask_rle[0::2]
    if sorted(starts) != list(starts):
        raise RuntimeError('Submitted values must be in ascending order.')
    try:
        return _rle_decode_jit(mask_rle, shape[0], shape[1]).reshape(shape, order='F')
    except ValueError as e:
        raise RuntimeError(str(e)) from e


def calculate_f1_score(pred_mask: np.ndarray, gt_mask: np.ndarray):
    pred_flat = pred_mask.flatten()
    gt_flat = gt_mask.flatten()
    tp = np.sum((pred_flat == 1) & (gt_flat == 1))
    fp = np.sum((pred_flat == 1) & (gt_flat == 0))
    fn = np.sum((pred_flat == 0) & (gt_flat == 1))
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    return 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0


def calculate_f1_matrix(pred_masks: list[np.ndarray], gt_masks: list[np.ndarray]):
    num_pred = len(pred_masks)
    num_gt = len(gt_masks)
    f1_matrix = np.zeros((num_pred, num_gt))
    for i in range(num_pred):
        for j in range(num_gt):
            f1_matrix[i, j] = calculate_f1_score(pred_masks[i], gt_masks[j])
    if num_pred < num_gt:
        f1_matrix = np.vstack((f1_matrix, np.zeros((num_gt - num_pred, num_gt))))
    return f1_matrix


def oF1_score(pred_masks: list[np.ndarray], gt_masks: list[np.ndarray]):
    if len(pred_masks) == 0 or len(gt_masks) == 0:
        return 0.0
    f1_matrix = calculate_f1_matrix(pred_masks, gt_masks)
    row_ind, col_ind = scipy.optimize.linear_sum_assignment(-f1_matrix)
    excess_penalty = len(gt_masks) / max(len(pred_masks), len(gt_masks))
    return np.mean(f1_matrix[row_ind, col_ind]) * excess_penalty


def rle_encode(mask):
    pixels = mask.T.flatten()
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    return json.dumps([int(x) for x in runs])