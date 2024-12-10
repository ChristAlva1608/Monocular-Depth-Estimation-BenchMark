import numpy as np

def log_rms_metric(ground_truth, prediction):
    """
    Calculate the Log Root Mean Square Error (Log RMS) metric.
    
    Args:
        ground_truth (numpy.ndarray): Ground truth depth map.
        prediction (numpy.ndarray): Predicted depth map.
    
    Returns:
        float: The Log RMS metric.
    """
    # Ensure inputs are NumPy arrays
    ground_truth = np.asarray(ground_truth)
    prediction = np.asarray(prediction)
    
    # Avoid invalid values by masking ground truth and prediction
    mask = (ground_truth > 0) & (prediction > 0)
    
    # Compute log differences and the Log RMS
    log_diff = np.log(ground_truth[mask]) - np.log(prediction[mask])
    log_rms = np.sqrt(np.mean(log_diff ** 2))
    
    return log_rms