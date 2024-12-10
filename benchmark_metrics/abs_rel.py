import numpy as np

def abs_rel_metric(ground_truth, prediction):
    """
    Calculate the Absolute Relative Difference (Abs Rel) metric.
    
    Args:
        ground_truth (numpy.ndarray): Ground truth depth map.
        prediction (numpy.ndarray): Predicted depth map.
    
    Returns:
        float: The Abs Rel metric.
    """
    # Ensure inputs are NumPy arrays
    ground_truth = np.asarray(ground_truth)
    prediction = np.asarray(prediction)
    
    # Avoid division by zero by masking invalid values
    mask = ground_truth > 0
    abs_rel = np.mean(np.abs(ground_truth[mask] - prediction[mask]) / ground_truth[mask])
    return abs_rel