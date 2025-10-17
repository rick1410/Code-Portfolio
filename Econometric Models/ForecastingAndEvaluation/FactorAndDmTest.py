import numpy as np
from collections import defaultdict

class ForecastUtils:
    @staticmethod
    def calculate_factor(log_returns_in_sample, kernel_in_sample):
        """
        Calculate the factor for scaling forecasts.
        """
        sample_variance = np.var(log_returns_in_sample, ddof=1)
        sample_mean_kernel = np.mean(kernel_in_sample)
        return sample_variance / sample_mean_kernel

  

def merge_by_key(runs, top_key):
    merged = defaultdict(dict)
    for run in runs:
        if top_key not in run:
            continue  # skip runs missing the key
        for h, models in run[top_key].items():
            for model_name, value in models.items():
                merged[h][model_name] = value
    return dict(merged)


def collect_test_data_by_horizon(runs):
    """
    Return a realized_kernel dictionary that depends on horizon only.
    Assumes all runs share the same realized_kernel per horizon.
    """
    result = {}
    for h in runs[0]["realized_kernel"]:
        # Just take the first available one per horizon
        result[h] = runs[0]["realized_kernel"][h]
    return result