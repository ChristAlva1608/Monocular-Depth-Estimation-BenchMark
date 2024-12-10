import numpy as np

def rms_metric(ground_truth, prediction):
    """
    Calculate the Root Mean Square Error (RMS) metric.
    
    Args:
        ground_truth (numpy.ndarray): Ground truth depth map.
        prediction (numpy.ndarray): Predicted depth map.
    
    Returns:
        float: The RMS metric.
    """
    # Ensure inputs are NumPy arrays
    ground_truth = np.asarray(ground_truth)
    prediction = np.asarray(prediction)
    
    # Avoid invalid values by masking ground truth
    mask = ground_truth > 0
    rms = np.sqrt(np.mean((ground_truth[mask] - prediction[mask]) ** 2))
    return rms