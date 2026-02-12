import numpy as np
from scipy.spatial.distance import euclidean, cityblock, cosine, minkowski
from typing import List


class MetricFactory:
    
    SUPPORTED_METRICS = [
        'euclidean',
        'manhattan', 
        'cosine',
        'minkowski',
        'mahalanobis'
    ]
    @staticmethod
    def get_metric(metric_name: str, **params):
        if metric_name not in MetricFactory.SUPPORTED_METRICS:
            raise ValueError(
                "Unsupported metric"
                f"Supported metrics: {':'.join(MetricFactory.SUPPORTED_METRICS)}"
            )
        if metric_name == 'euclidean':
            return euclidean

        elif metric_name == 'manhattan':
            return cityblock
        
        elif metric_name == 'cosine':
            return cosine
        
        elif metric_name == 'minkowski':
            p = params.get('p', 2)
            return lambda x, y: minkowski(x, y, p=p)
        
        elif metric_name == 'mahalanobis':
            cov_inv = params.get('cov_inv')
            if cov_inv is None:
                raise ValueError(
                    "Mahalanobis metric requires 'cov_inv' parameter "
                )
            
            def mahalanobis_distance(x: np.ndarray, y: np.ndarray) -> float:
                diff = x - y
                return np.sqrt(diff.T @ cov_inv @ diff)
            
            return mahalanobis_distance
    @staticmethod
    def list_metrics() -> List[str]:
        return list(MetricFactory.SUPPORTED_METRICS)
