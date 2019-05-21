from .classification import accuracy, precision, recall, f1, fbeta
from .regression import mean_squared_error, mean_absolute_error, root_mean_squared_error, r2, quantile_loss

__all__ = ['mean_squared_error', 'mean_absolute_error', 'root_mean_squared_error', 'accuracy', 'r2', 'quantile_loss',
           'precision', 'recall', 'f1', 'fbeta']
