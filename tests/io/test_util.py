import pytest
from trackintel.io.util import _index_warning_default_none


class Test_index_warning_default_none:
    def test_index_set(self):
        """Test if a set index creates no warning."""

        @_index_warning_default_none
        def foo(index_col=None):
            return

        foo(index_col=None)

    def test_index_default(self):
        """Test if default index creates a warning."""

        @_index_warning_default_none
        def foo(index_col=None):
            return

        with pytest.warns(UserWarning):
            foo()
