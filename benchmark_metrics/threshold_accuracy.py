import numpy as np

def delta1_metric(ground_truth, prediction, threshold=1.25):
    """
    Calculate the δ1↑ metric for depth estimation.

    Args:
        ground_truth (numpy.ndarray): Ground truth depth map.
        prediction (numpy.ndarray): Predicted depth map.
        threshold (float): The δ threshold (default is 1.25 for δ1).

    Returns:
        float: The δ1↑ metric as a percentage.
    """
    # Ensure inputs are NumPy arrays
    ground_truth = np.asarray(ground_truth)
    prediction = np.asarray(prediction)
    
    # Avoid invalid values (e.g., divide by zero)
    mask = ground_truth > 0
    
    # Calculate ratio max(d_p/d_t, d_t/d_p)
    ratio = np.maximum(prediction[mask] / ground_truth[mask], ground_truth[mask] / prediction[mask])
    
    # Compute δ1
    delta = np.mean(ratio < threshold)
    
    return delta * 100  # Return as a percentage