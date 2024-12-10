import numpy as np

def square_relative_metric(ground_truth, prediction):
    """
    Calculate the Square Relative Metric (Sq Rel) metric.

    Args:
        ground_truth (numpy.ndarray): Ground truth depth map.
        prediction (numpy.ndarray): Predicted depth map.

    Returns:
        float: The Sq Rel metric.
    """
    # Ensure inputs are NumPy arrays
    ground_truth = np.asarray(ground_truth)
    prediction = np.asarray(prediction)
    
    # Avoid division by zero and handle invalid values
    mask = ground_truth > 0
    
    # Compute Square Relative Metric
    sq_rel = np.mean(((ground_truth[mask] - prediction[mask]) ** 2) / ground_truth[mask])
    
    return sq_rel