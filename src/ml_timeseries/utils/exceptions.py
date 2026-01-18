"""
Custom exceptions for the ML Time Series project
"""


class TimeSeriesException(Exception):
    """Base exception for all project-specific errors"""
    pass


class DataValidationError(TimeSeriesException):
    """Raised when input data validation fails"""
    pass


class PreprocessingError(TimeSeriesException):
    """Raised during preprocessing failures"""
    pass


class FeatureEngineeringError(TimeSeriesException):
    """Raised during feature engineering failures"""
    pass


class ModelTrainingError(TimeSeriesException):
    """Raised when model training fails"""
    pass


class ModelPredictionError(TimeSeriesException):
    """Raised during prediction failures"""
    pass


class EvaluationError(TimeSeriesException):
    """Raised during evaluation failures"""
    pass
