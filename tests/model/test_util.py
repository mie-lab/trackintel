import pandas as pd
import pytest
from geopandas import GeoDataFrame
from shapely.geometry import Point

import trackintel as ti
from trackintel.model.util import (
    NonCachedAccessor,
    doc,
    _register_trackintel_accessor,
    _wrapped_gdf_method,
    TrackintelGeoDataFrame,
    TrackintelDataFrame,
)


@pytest.fixture
def example_positionfixes():
    """Positionfixes for tests."""
    p1 = Point(8.5067847, 47.4)
    p2 = Point(8.5067847, 47.5)
    p3 = Point(8.5067847, 47.6)

    t1 = pd.Timestamp("1971-01-01 00:00:00", tz="utc")
    t2 = pd.Timestamp("1971-01-01 05:00:00", tz="utc")
    t3 = pd.Timestamp("1971-01-02 07:00:00", tz="utc")

    list_dict = [
        {"user_id": 0, "tracked_at": t1, "geometry": p1},
        {"user_id": 0, "tracked_at": t2, "geometry": p2},
        {"user_id": 1, "tracked_at": t3, "geometry": p3},
    ]
    pfs = GeoDataFrame(data=list_dict, geometry="geometry", crs="EPSG:4326")
    pfs.index.name = "id"

    # assert validity of positionfixes.
    pfs.as_positionfixes
    return pfs


class Test_wrapped_gdf_method:
    """Test if _wrapped_gdf_method conditionals work"""

    def test_no_dataframe(self, example_positionfixes):
        """Test if function return value does not subclass DataFrame then __class__ is not touched"""

        @_wrapped_gdf_method
        def foo(gdf: GeoDataFrame) -> pd.Series:
            return pd.Series(gdf.iloc[0])

        assert type(foo(example_positionfixes)) is pd.Series

    def test_keep_class(self, example_positionfixes):
        """Test if original class is restored if return value subclasses GeoDataFrame"""

        class A(GeoDataFrame):
            fallback_class = None

        @_wrapped_gdf_method
        def foo(a: A) -> GeoDataFrame:
            return GeoDataFrame(a)

        a = A(example_positionfixes)
        assert type(foo(a)) is A

    def test_fallback(self, example_positionfixes):
        """Test if fallback to fallback_class if no GeoDataFrame"""

        class B(pd.DataFrame):
            pass

        class A(GeoDataFrame):
            fallback_class = B

        @_wrapped_gdf_method
        def foo(a: A) -> pd.DataFrame:
            return pd.DataFrame(a)

        a = A(example_positionfixes)
        assert type(foo(a)) is B

    def test_no_fallback(self, example_positionfixes):
        """Test if fallback_class is not set then fallback_class is not used."""

        class A(GeoDataFrame):
            fallback_class = None

        @_wrapped_gdf_method
        def foo(a: A) -> pd.DataFrame:
            return pd.DataFrame(a)

        a = A(example_positionfixes)
        assert type(foo(a)) is pd.DataFrame


class TestTrackintelGeoDataFrame:
    """Test helper class TrackintelGeoDataFrame."""

    class A(TrackintelGeoDataFrame):
        """Mimic TrackintelGeoDataFrame subclass by taking the same arguments"""

        class AFallback(TrackintelDataFrame):
            def __init__(self, *args, validate=True, **kwargs):
                super().__init__(*args, **kwargs)

        fallback_class = AFallback

        def __init__(self, *args, validate=True, validate_geometry=True, **kwargs):
            super().__init__(*args, **kwargs)

    def test_getitem(self, example_positionfixes):
        """Test if loc on all columns returns original class."""
        a = self.A(example_positionfixes)
        b = a.loc[[True for _ in a.columns]]
        assert type(b) is self.A

    def test_copy(self, example_positionfixes):
        """Test if copy maintains class."""
        a = self.A(example_positionfixes)
        b = a.copy()
        assert type(b) is self.A

    def test_merge(self, example_positionfixes):
        """Test if merge maintains class"""
        a = self.A(example_positionfixes)
        b = a.merge(a, on="user_id", suffixes=("", "_other"))
        assert type(b) is self.A

    def test_constructor_fallback_class(self, example_positionfixes):
        """Test if _constructor gets can fallback to fallback_class"""
        a = self.A(example_positionfixes)
        a = a.drop(columns=a.geometry.name)
        assert type(a) is self.A.fallback_class

    def test_constructor_no_fallback_class(self, example_positionfixes):
        """Test if _constructor does not fallback to fallback_class if not set"""
        a = self.A(example_positionfixes)
        a.fallback_class = None  # unset it again
        a = a.drop(columns=a.geometry.name)
        assert type(a) is pd.DataFrame

    def test_constructor_calls_init(self, example_positionfixes):
        """Test if _constructor gets GeoDataFrame and fulfills test then builds class"""
        a = self.A(example_positionfixes)
        assert type(a._constructor(a)) is self.A


class TestTrackintelDataFrame:
    """Test helper class TrackintelDataFrame."""

    class A(TrackintelDataFrame):
        """Mimic TrackintelDataFrame subclass by taking the same arguments"""

        def __init__(self, *args, validate=True, **kwargs):
            super().__init__(*args, **kwargs)

    def test_constructor_calls_init(self, example_positionfixes):
        """Test if _constructor gets DataFrame and fulfills test then builds class"""
        a = self.A(example_positionfixes)
        assert type(a._constructor(a)) is self.A


class TestNonCachedAccessor:
    """Test if NonCachedAccessor works"""

    def test_accessor(self):
        """Test accessor on class object and class instance."""

        def foo(val):
            return val

        class A:
            nca = NonCachedAccessor("nca_test", foo)

        a = A()
        assert A.nca == foo  # class object
        assert a.nca == a  # class instance


class Test_register_trackintel_accessor:
    """Test if accessors are correctly registered."""

    def test_register(self):
        """Test if accessor is registered in DataFrame"""

        def foo(val):
            return val

        bar = _register_trackintel_accessor("foo")(foo)
        assert foo == bar
        assert "foo" in pd.DataFrame._accessors
        assert foo == pd.DataFrame.foo
        # remove accessor again to make tests independent
        pd.DataFrame._accesors = pd.DataFrame._accessors.remove("foo")
        del pd.DataFrame.foo

    def test_duplicate_name_warning(self):
        """Test that duplicate name raises warning"""

        def foo(val):
            return val

        _register_trackintel_accessor("foo")(foo)
        with pytest.warns(UserWarning):
            _register_trackintel_accessor("foo")(foo)
        # remove accessor again to make tests independent
        pd.DataFrame._accesors = pd.DataFrame._accessors.remove("foo")
        del pd.DataFrame.foo


class TestDoc:
    """Test doc decorator"""

    def test_default_docstring(self):
        """Test that default docstring is kept."""

        def foo():
            pass

        default = "I am a docstring"
        foo.__doc__ = default
        foo = doc()(foo)
        assert foo.__doc__ == default

    def test_None(self):
        """Test that None in args create no docstring"""

        def foo():
            pass

        foo = doc(None)(foo)
        assert foo.__doc__ == ""

    def test_docstring_component(self):
        """Test that docstring component is formatable"""
        d = "this is a {adjective} function"

        def foo():
            pass

        foo._docstring_components = [d]
        foo = doc(foo, adjective="cool")(foo)
        assert foo.__doc__ == d.format(adjective="cool")

    def test_string(self):
        """Test if string can be supplied"""
        d = "this is a {adjective} function"

        def foo():
            pass

        foo = doc(d, adjective="cool")(foo)
        assert foo.__doc__ == d.format(adjective="cool")

    def test_fall_through_case(self):
        """Test fall through case in for loop"""

        def foo():
            pass

        foo = doc(foo)(foo)
        assert foo.__doc__ == ""
