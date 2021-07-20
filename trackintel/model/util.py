from functools import partial, update_wrapper


def copy_docstring(wrapped, assigned=("__doc__",), updated=[]):
    return partial(update_wrapper, wrapped=wrapped, assigned=assigned, updated=updated)
