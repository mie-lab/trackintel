from trackintel.model.util import copy_docstring
from functools import WRAPPER_ASSIGNMENTS
from trackintel.io.postgis import read_trips_postgis


class TestCopy_Docstring:
    def test_default(self):
        @copy_docstring(read_trips_postgis)
        def bar(b: int) -> int:
            """Old docstring."""
            pass

        old_docs = """Old docstring."""
        print(type(old_docs))

        for wa in WRAPPER_ASSIGNMENTS:
            attr_foo = getattr(read_trips_postgis, wa)
            attr_bar = getattr(bar, wa)
            if wa == "__doc__":
                assert attr_foo == attr_bar
                assert attr_bar != old_docs
            else:
                assert attr_foo != attr_bar
