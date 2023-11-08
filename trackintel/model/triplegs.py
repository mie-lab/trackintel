import pandas as pd

import trackintel as ti
from trackintel.model.util import (
    TrackintelBase,
    TrackintelGeoDataFrame,
    _register_trackintel_accessor,
    _shared_docs,
    doc,
)

_required_columns = ["user_id", "started_at", "finished_at"]


@_register_trackintel_accessor("as_triplegs")
class Triplegs(TrackintelBase, TrackintelGeoDataFrame):
    """Trackintel class to treat a GeoDataFrame as a collections of `Tripleg`.

    Requires at least the following columns:
    ['user_id', 'started_at', 'finished_at']

    Requires valid line geometries; the 'index' of the GeoDataFrame will be treated as unique identifier
    of the `triplegs`

    For several usecases, the following additional columns are required:
    ['mode', 'trip_id']

    Notes
    -----
    A `Tripleg` (also called `stage`) is defined as continuous movement without changing the mode of transport.

    'started_at' and 'finished_at' are timezone aware pandas datetime objects.

    Examples
    --------
    >>> triplegs.generate_trips()
    """

    def __init__(self, *args, validate=True, **kwargs):
        super().__init__(*args, **kwargs)
        if validate:
            self.validate(self)

    # create circular reference directly -> avoid second call of init via accessor
    @property
    def as_triplegs(self):
        return self

    @staticmethod
    def validate(obj):
        assert obj.shape[0] > 0, f"Geodataframe is empty with shape: {obj.shape}"
        # check columns
        if any([c not in obj.columns for c in _required_columns]):
            raise AttributeError(
                "To process a DataFrame as a collection of triplegs, it must have the properties"
                f" {_required_columns}, but it has [{', '.join(obj.columns)}]."
            )

        # check timestamp dtypes
        assert isinstance(
            obj["started_at"].dtype, pd.DatetimeTZDtype
        ), f"dtype of started_at is {obj['started_at'].dtype} but has to be datetime64 and timezone aware"
        assert isinstance(
            obj["finished_at"].dtype, pd.DatetimeTZDtype
        ), f"dtype of finished_at is {obj['finished_at'].dtype} but has to be datetime64 and timezone aware"

        # check geometry
        assert (
            obj.geometry.is_valid.all()
        ), "Not all geometries are valid. Try x[~ x.geometry.is_valid] where x is you GeoDataFrame"
        if obj.geometry.iloc[0].geom_type != "LineString":
            raise AttributeError("The geometry must be a LineString (only first checked).")

    @doc(_shared_docs["write_csv"], first_arg="", long="triplegs", short="tpls")
    def to_csv(self, filename, *args, **kwargs):
        ti.io.write_triplegs_csv(self, filename, *args, **kwargs)

    @doc(_shared_docs["write_postgis"], first_arg="", long="triplegs", short="tpls")
    def to_postgis(
        self, name, con, schema=None, if_exists="fail", index=True, index_label=None, chunksize=None, dtype=None
    ):
        ti.io.write_triplegs_postgis(self, name, con, schema, if_exists, index, index_label, chunksize, dtype)

    def calculate_distance_matrix(self, Y=None, dist_metric="haversine", n_jobs=0, **kwds):
        """
        Calculate a distance matrix based on a specific distance metric.

        See :func:`trackintel.geogr.calculate_distance_matrix` for full documentation.
        """
        return ti.geogr.calculate_distance_matrix(self, Y=Y, dist_metric=dist_metric, n_jobs=n_jobs, **kwds)

    def spatial_filter(self, areas, method="within", re_project=False):
        """
        Filter Triplegs on a geo extent.

        See :func:`trackintel.preprocessing.spatial_filter` for full documentation.
        """
        return ti.preprocessing.spatial_filter(self, areas, method=method, re_project=re_project)

    def generate_trips(self, staypoints, gap_threshold=15, add_geometry=True):
        """
        Generate trips based on staypoints and triplegs.

        See :func:`trackintel.preprocessing.generate_trips` for full documentation.
        """
        return ti.preprocessing.generate_trips(staypoints, self, gap_threshold=gap_threshold, add_geometry=add_geometry)

    def predict_transport_mode(self, method="simple-coarse", **kwargs):
        """
        Predict the transport mode of triplegs.

        See :func:`trackintel.analysis.predict_transport_mode` for full documentation.
        """
        return ti.analysis.predict_transport_mode(self, method=method, **kwargs)

    def calculate_modal_split(self, freq=None, metric="count", per_user=False, norm=False):
        """
        Calculate the modal split of triplegs.

        See :func:`trackintel.analysis.calculate_modal_split` for full documentation.
        """
        return ti.analysis.calculate_modal_split(self, freq=freq, metric=metric, per_user=per_user, norm=norm)

    def temporal_tracking_quality(self, granularity="all"):
        """
        Calculate per-user temporal tracking quality (temporal coverage).

        See :func:`trackintel.analysis.temporal_tracking_quality` for full documentation.
        """
        return ti.analysis.temporal_tracking_quality(self, granularity=granularity)

    def get_speed(self, positionfixes=None, method="tpls_speed"):
        """
        Compute the average speed per positionfix for each tripleg (in m/s)

        See :func:`trackintel.geogr.get_speed_triplegs` for full documentation.
        """
        return ti.geogr.get_speed_triplegs(self, positionfixes=positionfixes, method=method)
