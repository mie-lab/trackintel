from functools import partial, update_wrapper


def _copy_docstring(wrapped, assigned=("__doc__",), updated=[]):
    """Thin wrapper for `functools.update_wrapper` to mimic `functools.wraps` but to only copy the docstring."""
    return partial(update_wrapper, wrapped=wrapped, assigned=assigned, updated=updated)
