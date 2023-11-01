import geopandas as gpd
import pandas as pd

import trackintel as ti
from trackintel.model.util import (
    TrackintelBase,
    TrackintelGeoDataFrame,
    _register_trackintel_accessor,
    _shared_docs,
    doc,
)

_required_columns = ["user_id", "tracked_at"]


@_register_trackintel_accessor("as_positionfixes")
class Positionfixes(TrackintelBase, TrackintelGeoDataFrame, gpd.GeoDataFrame):
    """Trackintel class to treat GeoDataFrames as collections of `Positionfixes`.

    Requires at least the following columns:
    ['user_id', 'tracked_at']

    Requires valid point geometries; the 'index' of the GeoDataFrame will be treated as unique identifier
    of the `Positionfixes`.

    For several usecases, the following additional columns are required:
    ['elevation', 'accuracy' 'staypoint_id', 'tripleg_id']

    Notes
    -----
    In GPS based movement data analysis `Positionfixes` are the smallest unit of tracking and
    represent timestamped locations.

    'tracked_at' is a timezone aware pandas datetime object.

    Examples
    --------
    >>> positionfixes.generate_staypoints()
    """

    def __init__(self, *args, validate=True, **kwargs):
        # could be moved to super class
        # validate kwarg is necessary as the object is not fully initialised if we call it from _constructor
        # (geometry-link is missing). thus we need a way to stop validating too early.
        super().__init__(*args, **kwargs)
        if validate:
            self.validate(self)

    # create circular reference directly -> avoid second call of init via accessor
    @property
    def as_positionfixes(self):
        return self

    @staticmethod
    def validate(obj):
        assert obj.shape[0] > 0, f"Geodataframe is empty with shape: {obj.shape}"
        # check columns
        if any([c not in obj.columns for c in _required_columns]):
            raise AttributeError(
                "To process a DataFrame as a collection of positionfixes, it must have the properties"
                f" {_required_columns}, but it has [{', '.join(obj.columns)}]."
            )
        # check timestamp dtypes
        assert isinstance(
            obj["tracked_at"].dtype, pd.DatetimeTZDtype
        ), f"dtype of tracked_at is {obj['tracked_at'].dtype} but has to be datetime64 and timezone aware"

        # check geometry
        assert (
            obj.geometry.is_valid.all()
        ), "Not all geometries are valid. Try x[~ x.geometry.is_valid] where x is you GeoDataFrame"

        if obj.geometry.iloc[0].geom_type != "Point":
            raise AttributeError("The geometry must be a Point (only first checked).")

    @property
    def center(self):
        """Return the center coordinate of this collection of positionfixes."""
        lat = self.geometry.y
        lon = self.geometry.x
        return (float(lon.mean()), float(lat.mean()))

    def generate_staypoints(
        self,
        method="sliding",
        distance_metric="haversine",
        dist_threshold=100,
        time_threshold=5.0,
        gap_threshold=15.0,
        include_last=False,
        print_progress=False,
        exclude_duplicate_pfs=True,
        n_jobs=1,
    ):
        """
        Generate staypoints based on positionfixes.

        See :func:`trackintel.preprocessing.generate_staypoints` for full documentation.
        """
        return ti.preprocessing.generate_staypoints(
            self,
            method=method,
            distance_metric=distance_metric,
            dist_threshold=dist_threshold,
            time_threshold=time_threshold,
            gap_threshold=gap_threshold,
            include_last=include_last,
            print_progress=print_progress,
            exclude_duplicate_pfs=exclude_duplicate_pfs,
            n_jobs=n_jobs,
        )

    def generate_triplegs(
        self,
        staypoints=None,
        method="between_staypoints",
        gap_threshold=15,
        print_progress=False,
    ):
        """
        Generate triplegs from positionfixes.

        See :func:`trackintel.preprocessing.generate_triplegs` for full documentation.
        """
        return ti.preprocessing.generate_triplegs(
            self,
            staypoints=staypoints,
            method=method,
            gap_threshold=gap_threshold,
            print_progress=print_progress,
        )

    def to_csv(self, filename, *args, **kwargs):
        """
        Write positionfixes to csv file.

        See :func:`trackintel.io.write_positionfixes_csv` for full documentation.
        """
        ti.io.write_positionfixes_csv(self, filename, *args, **kwargs)

    @doc(_shared_docs["write_postgis"], first_arg="", long="positionfixes", short="pfs")
    def to_postgis(
        self, name, con, schema=None, if_exists="fail", index=True, index_label=None, chunksize=None, dtype=None
    ):
        ti.io.write_positionfixes_postgis(self, name, con, schema, if_exists, index, index_label, chunksize, dtype)

    def calculate_distance_matrix(self, Y=None, dist_metric="haversine", n_jobs=0, **kwds):
        """
        Calculate a distance matrix based on a specific distance metric.

        See :func:`trackintel.geogr.calculate_distance_matrix` for full documentation.
        """
        return ti.geogr.calculate_distance_matrix(self, Y=Y, dist_metric=dist_metric, n_jobs=n_jobs, **kwds)

    def get_speed(self):
        """
        Compute speed per positionfix (in m/s)

        See :func:`trackintel.geogr.get_speed_positionfixes` for full documentation.
        """
        return ti.geogr.get_speed_positionfixes(self)
