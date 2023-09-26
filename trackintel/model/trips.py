import geopandas as gpd
import pandas as pd

import trackintel as ti
from trackintel.model.util import (
    TrackintelBase,
    TrackintelDataFrame,
    TrackintelGeoDataFrame,
    _register_trackintel_accessor,
    _shared_docs,
    doc,
)


@_register_trackintel_accessor("as_trips")
class Trips:
    """A pandas accessor to treat (Geo)DataFrames as collections of trips.

    The class constructor will create a TripsDataFrame or a TripsGeoDataFrame depending if a geometry column is present.

    Requires at least the following columns:
    ['user_id', 'started_at', 'finished_at', 'origin_staypoint_id', 'destination_staypoint_id']

    The 'index' of the (Geo)DataFrame will be treated as unique identifier of the `Trips`

    Trips have an optional geometry of type MultiPoint which describes the start and the end point of the trip

    For several usecases, the following additional columns are required:
    ['origin_purpose', 'destination_purpose', 'modes', 'primary_mode', 'tour_id']

    Notes
    -----
    `Trips` are an aggregation level in transport planning that summarize all movement and all non-essential actions
    (e.g., waiting) between two relevant activities. The following assumptions are implemented
        - If we do not record a person for more than `gap_threshold` minutes, we assume that the person performed an \
          activity in the recording gap and split the trip at the gap.
        - Trips that start/end in a recording gap can have an unknown origin/destination staypoint id.
        - If the origin (or destination) staypoint is unknown (and a geometry column exists), the origin/destination
          geometry is set as the first coordinate of the first tripleg (or the last coordinate of the last tripleg)
        - There are no trips without a (recorded) tripleg.

    'started_at' and 'finished_at' are timezone aware pandas datetime objects.

    Examples
    --------
    >>> df.as_trips.generate_tours()
    """

    def __new__(cls, *args, **kwargs):
        is_gdf = (
            (len(args) > 0 and isinstance(args[0], gpd.GeoDataFrame))
            or "geometry" in kwargs
            or ("data" in kwargs and isinstance(kwargs["data"], gpd.GeoDataFrame))
        )
        if is_gdf:
            return TripsGeoDataFrame(*args, **kwargs)
        return TripsDataFrame(*args, **kwargs)


_required_columns = ["user_id", "started_at", "finished_at", "origin_staypoint_id", "destination_staypoint_id"]


class TripsDataFrame(TrackintelBase, TrackintelDataFrame):
    """Class to treat a DataFrame as collections of trips.

    Requires at least the following columns:
    ['user_id', 'started_at', 'finished_at', 'origin_staypoint_id', 'destination_staypoint_id']

    The 'index' of the DataFrame will be treated as unique identifier of the `Trips`

    For several usecases, the following additional columns are required:
    ['origin_purpose', 'destination_purpose', 'modes', 'primary_mode', 'tour_id']

    Notes
    -----
    `Trips` are an aggregation level in transport planning that summarize all movement and all non-essential actions
    (e.g., waiting) between two relevant activities. The following assumptions are implemented
        - If we do not record a person for more than `gap_threshold` minutes, we assume that the person performed an \
          activity in the recording gap and split the trip at the gap.
        - Trips that start/end in a recording gap can have an unknown origin/destination staypoint id.
        - If the origin (or destination) staypoint is unknown (and a geometry column exists), the origin/destination
          geometry is set as the first coordinate of the first tripleg (or the last coordinate of the last tripleg)
        - There are no trips without a (recorded) tripleg.

    'started_at' and 'finished_at' are timezone aware pandas datetime objects.

    Examples
    --------
    >>> df.as_trips.generate_tours()
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        TripsDataFrame._validate(self)  # static call

    @staticmethod
    def _validate(obj):
        if any([c not in obj.columns for c in _required_columns]):
            raise AttributeError(
                "To process a DataFrame as a collection of trips, it must have the properties"
                f" {_required_columns}, but it has [{', '.join(obj.columns)}]."
            )

        # check timestamp dtypes
        assert isinstance(
            obj["started_at"].dtype, pd.DatetimeTZDtype
        ), f"dtype of started_at is {obj['started_at'].dtype} but has to be datetime64 and timezone aware"
        assert isinstance(
            obj["finished_at"].dtype, pd.DatetimeTZDtype
        ), f"dtype of finished_at is {obj['finished_at'].dtype} but has to be datetime64 and timezone aware"

    @staticmethod
    def _check(obj):
        if any([c not in obj.columns for c in _required_columns]):
            return False
        if not isinstance(obj["started_at"].dtype, pd.DatetimeTZDtype):
            return False
        if not isinstance(obj["finished_at"].dtype, pd.DatetimeTZDtype):
            return False
        return True

    @doc(_shared_docs["write_csv"], first_arg="", long="trips", short="trips")
    def to_csv(self, filename, *args, **kwargs):
        ti.io.file.write_trips_csv(self, filename, *args, **kwargs)

    @doc(_shared_docs["write_postgis"], first_arg="", long="trips", short="trips")
    def to_postgis(
        self, name, con, schema=None, if_exists="fail", index=True, index_label=None, chunksize=None, dtype=None
    ):
        ti.io.postgis.write_trips_postgis(self, name, con, schema, if_exists, index, index_label, chunksize, dtype)

    def temporal_tracking_quality(self, granularity="all"):
        # if you update this docstring update ti.analysis.tracking_quality as well.
        """
        Calculate per-user temporal tracking quality (temporal coverage).

        Parameters
        ----------
        granularity : {"all", "day", "week", "weekday", "hour"}
            The level of which the tracking quality is calculated. The default "all" returns
            the overall tracking quality; "day" the tracking quality by days; "week" the quality
            by weeks; "weekday" the quality by day of the week (e.g, Mondays, Tuesdays, etc.) and
            "hour" the quality by hours.

        Returns
        -------
        quality: DataFrame
            A per-user per-granularity temporal tracking quality dataframe.

        Notes
        -----
        Requires at least the following columns:
        ``['user_id', 'started_at', 'finished_at']``
        which means the function supports trackintel ``staypoints``, ``triplegs``, ``trips`` and ``tours``
        datamodels and their combinations (e.g., staypoints and triplegs sequence).

        The temporal tracking quality is the ratio of tracking time and the total time extent. It is
        calculated and returned per-user in the defined ``granularity``. The time extents
        and the columns for the returned ``quality`` df for different ``granularity`` are:

        - ``all``:
            - time extent: between the latest "finished_at" and the earliest "started_at" for each user.
            - columns: ``['user_id', 'quality']``.

        - ``week``:
            - time extent: the whole week (604800 sec) for each user.
            - columns: ``['user_id', 'week_monday', 'quality']``.

        - ``day``:
            - time extent: the whole day (86400 sec) for each user
            - columns: ``['user_id', 'day', 'quality']``

        - ``weekday``
            - time extent: the whole day (86400 sec) * number of tracked weeks for each user for each user
            - columns: ``['user_id', 'weekday', 'quality']``

        - ``hour``:
            - time extent: the whole hour (3600 sec) * number of tracked days for each user
            - columns: ``['user_id', 'hour', 'quality']``

        Examples
        --------
        >>> # calculate overall tracking quality of staypoints
        >>> temporal_tracking_quality(sp, granularity="all")
        >>> # calculate per-day tracking quality of sp and tpls sequence
        >>> temporal_tracking_quality(sp_tpls, granularity="day")
        """
        return ti.analysis.tracking_quality.temporal_tracking_quality(self, granularity=granularity)

    def generate_tours(self, **kwargs):
        # if you update this docstring update ti.preprocessing.generate_tours as well
        """
        Generate trackintel-tours from trips

        Parameters
        ----------
        trips : GeoDataFrame (as trackintel trips)

        staypoints : GeoDataFrame (as trackintel staypoints), default None
            Must have `location_id` column to connect trips via locations to a tour.
            If None, trips will be connected based only by the set distance threshold `max_dist`.

        max_dist: float, default 100 (meters)
            Maximum distance between the end point of one trip and the start point of the next trip within a tour.
            This is parameter is only used if staypoints is `None`!
            Also, if `max_nr_gaps > 0`, a tour can contain larger spatial gaps (see Notes below for more detail)

        max_time: str or pd.Timedelta, default "1d" (1 day)
            Maximum time that a tour is allowed to take

        max_nr_gaps: int, default 0
            Maximum number of spatial gaps on the tour. Use with caution - see notes below.

        print_progress : bool, default False
            If print_progress is True, the progress bar is displayed

        Returns
        -------
        trips_with_tours: GeoDataFrame (as trackintel trips)
            Same as `trips`, but with column `tour_id`, containing a list of the tours that the trip is part of (see notes).

        tours: GeoDataFrame (as trackintel tours)
            The generated tours

        Examples
        --------
        >>> trips.as_trips.generate_tours(staypoints)

        Notes
        -------
        - Tours are defined as a collection of trips in a certain time frame that start and end at the same point
        - Tours and trips have an N:N relationship: One tour consists of multiple trips, but also one trip can be part of
        multiple tours, due to nested tours or overlapping tours.
        - This function implements two possibilities to generate tours of trips: Via the location ID in the `staypoints`
        df, or via a maximum distance. Thus, note that only one of the parameters `staypoints` or `max_dist` is used!
        - Nested tours are possible and will be regarded as 2 (or more tours).
        - It is possible to allow spatial gaps to occur on the tour, which might be useful to deal with missing data.
        Example: The two trips home-work, supermarket-home would still be detected as a tour when max_nr_gaps >= 1,
        although the work-supermarket trip is missing.
        Warning: This only counts the number of gaps, but neither temporal or spatial distance of gaps, nor the number
        of missing trips in a gap are bounded. Thus, this parameter should be set with caution, because trips that are
        hours apart might still be connected to a tour if `max_nr_gaps > 0`.
        """
        return ti.preprocessing.trips.generate_tours(trips=self, **kwargs)


# added GeoDataFrame manually afterwards such that our methods always come first
class TripsGeoDataFrame(TrackintelGeoDataFrame, TripsDataFrame, gpd.GeoDataFrame):
    """Class to treat a GeoDataFrame as collections of trips.

    Requires at least the following columns:
    ['user_id', 'started_at', 'finished_at', 'origin_staypoint_id', 'destination_staypoint_id']

    The 'index' of the GeoDataFrame will be treated as unique identifier of the `Trips`

    TripsGeoDataFrame must have a geometry of type MultiPoint which describes the start and the end point of the trip.

    For several usecases, the following additional columns are required:
    ['origin_purpose', 'destination_purpose', 'modes', 'primary_mode', 'tour_id']

    Notes
    -----
    `Trips` are an aggregation level in transport planning that summarize all movement and all non-essential actions
    (e.g., waiting) between two relevant activities. The following assumptions are implemented
        - If we do not record a person for more than `gap_threshold` minutes, we assume that the person performed an \
          activity in the recording gap and split the trip at the gap.
        - Trips that start/end in a recording gap can have an unknown origin/destination staypoint id.
        - If the origin (or destination) staypoint is unknown (and a geometry column exists), the origin/destination
          geometry is set as the first coordinate of the first tripleg (or the last coordinate of the last tripleg)
        - There are no trips without a (recorded) tripleg.

    'started_at' and 'finished_at' are timezone aware pandas datetime objects.

    Examples
    --------
    >>> df.as_trips.generate_tours()
    """

    fallback_class = TripsDataFrame

    def __init__(self, *args, validate_geometry=True, **kwargs):
        super().__init__(*args, **kwargs)
        TripsGeoDataFrame._validate(self, validate_geometry=validate_geometry)

    @staticmethod
    def _validate(self, validate_geometry=True):
        if not validate_geometry:
            return
        # _validate should not be called from the outside -> we can use fact that it is called only from __init__
        # therefore TrackintelDataFrame validated all the columns and here we only have to validate the geometry
        assert (
            self.geometry.is_valid.all()
        ), "Not all geometries are valid. Try x[~x.geometry.is_valid] where x is you GeoDataFrame"
        if self.geometry.iloc[0].geom_type != "MultiPoint":
            raise ValueError("The geometry must be a MultiPoint (only first checked).")

    @staticmethod
    def _check(self, validate_geometry=True):
        val = TripsDataFrame._check(self)
        if not validate_geometry:
            return val
        return val and self.geometry.is_valid.all() and self.geometry.iloc[0].geom_type == "MultiPoint"
