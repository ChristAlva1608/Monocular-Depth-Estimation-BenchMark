import time
import numpy as np
import torch
from torch.utils.data import DataLoader
from benchmark_metrics import *

models = {
    'marigold': {
        'class': MariGoldPipeline,
        'checkpoint': 'path/to/marigold_checkpoint.pth',  
        'batch_size': 2  
    },
    'adabins': {
        'class': Adabin,
        'checkpoint': 'path/to/adabins_checkpoint.pth',  
        'batch_size': 2  
    }
}

datasets = {
    'kitti': KITTIDataset,
    'vkitti': VirtualKITTIDataset,
    'nyu': NYUDataset
}

metrics = {
    'abs_rel': abs_rel_metric,
    'rms': rms_metric,
    'log_rms': log_rms_metric,
    'sq_rel': square_relative_metric,
    'threshold_accuracy': delta1_metric
}

def load_checkpoint(model_class, checkpoint):
    try:
        model = model_class()
        ckpt = torch.load(checkpoint)
        model.load_state_dict(ckpt['model_state_dict'])
    except KeyError:
        model.from_pretrained(
            checkpoint, torch_dtype=torch.float16
        )
    return model

def evaluate_model(model, dataloader, metric_function):
    # Loop through the dataset and evaluate using the metric function
    results = []
    for data, ground_truth in dataloader:
        prediction = model.predict(data)
        metric_value = metric_function(ground_truth, prediction)
        results.append(metric_value)
    
    # Return the average of the results
    return np.mean(results)

def main():

    '''
        Results: a list of dict contrains metric values for each model in each benchmark dataset

        result = [
            {
                'model': adabins,
                'dataset': kitti,
                'metric': {
                    'abs_rel': 0.9,
                    'rms': 0.7
                }
            }
        ]
    '''
    result = []
    for model_name, model_info in models.items():
        model_class = model_info['class']
        checkpoint = model_info['checkpoint']
        batch_size = model_info['batch_size']
        
        for dataset_name, dataset in datasets.items():
            for metric_name, metric_function in metrics.items():
                print(f"Evaluating Model: {model_name}, Dataset: {dataset_name}, Metric: {metric_name}")

                # Evaluate the model
                model = load_checkpoint(model_class, checkpoint)
                model.eval()

                dataset = dataset()
                dataloader = DataLoader(
                    dataset=dataset,
                    batch_size=batch_size
                )
                avg_metric_value = evaluate_model(model, dataloader, metric_function)

                result.append({
                    'model': model_name,
                    'dataset': dataset_name,
                    'metric': {
                        metric_name: avg_metric_value
                    }
                })
                print(f"Avg {metric_name} for {model_name} on {dataset_name}: {avg_metric_value:.4f}\n")