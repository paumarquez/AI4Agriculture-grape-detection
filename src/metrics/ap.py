import numpy as np

def get_envelope(precisions: np.ndarray) -> np.array:
    """Compute the envelope of the precision curve.
    Args:
      precisions:
    Returns: enveloped precision
    """
    for i in range(precisions.size - 1, 0, -1):
        precisions[i - 1] = np.maximum(precisions[i - 1], precisions[i])
    return precisions


def get_ap(recalls: np.ndarray, precisions: np.ndarray) -> float:
    """Calculate area under precision/recall curve.
    Args:
      recalls: recalls sorted per score
      precisions: precisions sorted per score
    Returns:
    """
    # correct AP calculation
    # first append sentinel values at the end
    recalls = np.concatenate(([0.0], recalls, [1.0]))
    precisions = np.concatenate(([0.0], precisions, [0.0]))

    precisions = get_envelope(precisions)

    # to calculate area under PR curve, look for points where X axis (recall) changes value
    i = np.where(recalls[1:] != recalls[:-1])[0]

    # and sum (\Delta recall) * prec
    ap = np.sum((recalls[i + 1] - recalls[i]) * precisions[i + 1])
    return ap
