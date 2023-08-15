import warnings
from functools import wraps
from inspect import signature


def _index_warning_default_none(func):
    """Decorator function that warns if index_col None is not set explicit."""

    @wraps(func)  # copy all metadata
    def wrapper(*args, **kwargs):
        bound_values = signature(func).bind(*args, **kwargs)  # binds only available args and kwargs
        if "index_col" not in bound_values.arguments:
            warnings.warn(
                "Assuming default index as unique identifier. "
                "Pass 'index_col=None' as explicit argument to avoid a warning."
            )
        return func(*args, **kwargs)

    return wrapper
