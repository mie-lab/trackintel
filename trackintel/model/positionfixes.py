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
    """A pandas accessor to treat (Geo)DataFrames as collections of `Positionfixes`.

    This will define certain methods and accessors, as well as make sure that the DataFrame
    adheres to some requirements.

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
    >>> df.as_positionfixes.generate_staypoints()
    """

    def __init__(self, *args, validate_geometry=True, **kwargs):
        # could be moved to super class
        # validate kwarg is necessary as the object is not fully initialised if we call it from _constructor
        # (geometry-link is missing). thus we need a way to stop validating too early.
        super().__init__(*args, **kwargs)
        self._validate(self, validate_geometry=validate_geometry)

    # create circular reference directly -> avoid second call of init via accessor
    @property
    def as_positionfixes(self):
        return self

    @staticmethod
    def _validate(obj, validate_geometry=True):
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
        if validate_geometry:
            assert (
                obj.geometry.is_valid.all()
            ), "Not all geometries are valid. Try x[~ x.geometry.is_valid] where x is you GeoDataFrame"

            if obj.geometry.iloc[0].geom_type != "Point":
                raise AttributeError("The geometry must be a Point (only first checked).")

    @staticmethod
    def _check(obj, validate_geometry=True):
        """Check does the same as _validate but returns bool instead of potentially raising an error."""
        if any([c not in obj.columns for c in _required_columns]):
            return False
        if obj.shape[0] <= 0:
            return False
        if not isinstance(obj["tracked_at"].dtype, pd.DatetimeTZDtype):
            return False
        if validate_geometry:
            return obj.geometry.is_valid.all() and obj.geometry.iloc[0].geom_type == "Point"
        return True

    @property
    def center(self):
        """Return the center coordinate of this collection of positionfixes."""
        lat = self.geometry.y
        lon = self.geometry.x
        return (float(lon.mean()), float(lat.mean()))

    @doc(first_arg="")
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
        Generate staypoints from positionfixes.

        Parameters
        ----------{first_arg}
        method : {{'sliding'}}
            Method to create staypoints. 'sliding' applies a sliding window over the data.

        distance_metric : {{'haversine'}}
            The distance metric used by the applied method.

        dist_threshold : float, default 100
            The distance threshold for the 'sliding' method, i.e., how far someone has to travel to
            generate a new staypoint. Units depend on the dist_func parameter. If 'distance_metric' is 'haversine' the
            unit is in meters

        time_threshold : float, default 5.0 (minutes)
            The time threshold for the 'sliding' method in minutes.

        gap_threshold : float, default 15.0 (minutes)
            The time threshold of determine whether a gap exists between consecutive pfs. Consecutive pfs with
            temporal gaps larger than 'gap_threshold' will be excluded from staypoints generation.
            Only valid in 'sliding' method.

        include_last: boolean, default False
            The algorithm in Li et al. (2008) only detects staypoint if the user steps out
            of that staypoint. This will omit the last staypoint (if any). Set 'include_last'
            to True to include this last staypoint.

        print_progress: boolean, default False
            Show per-user progress if set to True.

        exclude_duplicate_pfs: boolean, default True
            Filters duplicate positionfixes before generating staypoints. Duplicates can lead to problems in later
            processing steps (e.g., when generating triplegs). It is not recommended to set this to False.

        n_jobs: int, default 1
            The maximum number of concurrently running jobs. If -1 all CPUs are used. If 1 is given, no parallel
            computing code is used at all, which is useful for debugging. See
            https://joblib.readthedocs.io/en/latest/parallel.html#parallel-reference-documentation
            for a detailed description

        Returns
        -------
        pfs: GeoDataFrame (as trackintel positionfixes)
            The original positionfixes with a new column ``[`staypoint_id`]``.

        sp: GeoDataFrame (as trackintel staypoints)
            The generated staypoints.

        Notes
        -----
        The 'sliding' method is adapted from Li et al. (2008). In the original algorithm, the 'finished_at'
        time for the current staypoint lasts until the 'tracked_at' time of the first positionfix outside
        this staypoint. Users are assumed to be stationary during this missing period and potential tracking
        gaps may be included in staypoints. To avoid including too large missing signal gaps, set 'gap_threshold'
        to a small value, e.g., 15 min.

        Examples
        --------
        >>> pfs.as_positionfixes.generate_staypoints('sliding', dist_threshold=100)

        References
        ----------
        Zheng, Y. (2015). Trajectory data mining: an overview. ACM Transactions on Intelligent Systems
        and Technology (TIST), 6(3), 29.

        Li, Q., Zheng, Y., Xie, X., Chen, Y., Liu, W., & Ma, W. Y. (2008, November). Mining user
        similarity based on location history. In Proceedings of the 16th ACM SIGSPATIAL international
        conference on Advances in geographic information systems (p. 34). ACM.
        """
        return ti.preprocessing.positionfixes.generate_staypoints(
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

    @doc(first_arg="")
    def generate_triplegs(
        self,
        staypoints=None,
        method="between_staypoints",
        gap_threshold=15,
        print_progress=False,
    ):
        """
        Generate triplegs from positionfixes.

        Parameters
        ----------{first_arg}
        staypoints : GeoDataFrame (as trackintel staypoints), optional
            The staypoints (corresponding to the positionfixes). If this is not passed, the
            positionfixes need 'staypoint_id' associated with them.

        method: {{'between_staypoints'}}
            Method to create triplegs. 'between_staypoints' method defines a tripleg as all positionfixes
            between two staypoints (no overlap). This method requires either a column 'staypoint_id' on
            the positionfixes or passing staypoints as an input.

        gap_threshold: float, default 15 (minutes)
            Maximum allowed temporal gap size in minutes. If tracking data is missing for more than
            `gap_threshold` minutes, a new tripleg will be generated.

        print_progress: boolean, default False
            Show the progress bar for assigning staypoints to positionfixes if set to True.

        Returns
        -------
        pfs: GeoDataFrame (as trackintel positionfixes)
            The original positionfixes with a new column ``[`tripleg_id`]``.

        tpls: GeoDataFrame (as trackintel triplegs)
            The generated triplegs.

        Notes
        -----
        Methods 'between_staypoints' requires either a column 'staypoint_id' on the
        positionfixes or passing some staypoints that correspond to the positionfixes!
        This means you usually should call ``generate_staypoints()`` first.

        The first positionfix after a staypoint is regarded as the first positionfix of the
        generated tripleg. The generated tripleg will not have overlapping positionfix with
        the existing staypoints. This means a small temporal gap in user's trace will occur
        between the first positionfix of staypoint and the last positionfix of tripleg:
        pfs_stp_first['tracked_at'] - pfs_tpl_last['tracked_at'].

        Examples
        --------
        >>> pfs.as_positionfixes.generate_triplegs('between_staypoints', gap_threshold=15)
        """
        return ti.preprocessing.positionfixes.generate_triplegs(
            self, staypoints=staypoints, method=method, gap_threshold=gap_threshold, print_progress=print_progress
        )

    @doc(first_arg="")
    def to_csv(self, filename, *args, **kwargs):
        """
        Write positionfixes to csv file.

        Wraps the pandas to_csv function, but strips the geometry column and
        stores the longitude and latitude in respective columns.

        Parameters
        ----------{first_arg}
        filename : str
            The file to write to.

        args
            Additional arguments passed to pd.DataFrame.to_csv().

        kwargs
            Additional keyword arguments passed to pd.DataFrame.to_csv().

        Notes
        -----
        "longitude" and "latitude" is extracted from the geometry column and the original
        geometry column is dropped.

        Examples
        ---------
        >>> pfs.as_positionfixes.to_csv("export_pfs.csv")
        """
        ti.io.file.write_positionfixes_csv(self, filename, *args, **kwargs)

    @doc(_shared_docs["write_postgis"], first_arg="", short="pfs", long="positionfixes")
    def to_postgis(
        self, name, con, schema=None, if_exists="fail", index=True, index_label=None, chunksize=None, dtype=None
    ):
        ti.io.postgis.write_positionfixes_postgis(
            self, name, con, schema, if_exists, index, index_label, chunksize, dtype
        )

    @doc(_shared_docs["calculate_distance_matrix"], first_arg="")
    def calculate_distance_matrix(self, Y=None, dist_metric="haversine", n_jobs=0, **kwargs):
        return ti.geogr.calculate_distance_matrix(self, Y=Y, dist_metric=dist_metric, n_jobs=n_jobs, **kwargs)

    def get_speed(self):
        """
        Compute speed per positionfix (in m/s)

        Returns
        -------
        pfs: GeoDataFrame (as trackintel positionfixes)
            The original positionfixes with a new column ``[`speed`]``. The speed is given in m/s

        Notes
        -----
        The speed at one positionfix is computed from the distance and time since the previous positionfix. For the first
        positionfix, the speed is set to the same value as for the second one.
        """
        return ti.geogr.get_speed_positionfixes(self)
