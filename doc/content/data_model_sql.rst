Data Model (SQL)
****************

For a general description of the data model, please refer to the 
:doc:`/modules/model`. 
You can download the `complete SQL script here <https://github.com/mie-lab/trackintel/blob/master/sql/create_tables_pg.sql>`_.

.. highlight:: sql

The **positionfixes** table contains all position fixes of all users. They are not 
only linked to a user, but also to a trip leg or a staypoint::

    CREATE TABLE positionfixes (
        id bigint NOT NULL,
        user_id bigint NOT NULL,
        tripleg_id bigint,
        staypoint_id bigint,

        tracked_at timestamp without time zone NOT NULL,
        latitude double precision NOT NULL,
        longitude double precision NOT NULL,
        elevation double precision,
        accuracy double precision,
        geom geometry(Point,4326),

        CONSTRAINT positionfixes_pkey PRIMARY KEY (id)
    );

The **triplegs** table contains all trip legs, i.e., journeys that have been taken 
with a single mode of transport. They are linked to both a user, as well as a trip 
and if applicable, a public transport case::

    CREATE TABLE triplegs (
        id bigint NOT NULL,
        user_id bigint NOT NULL,
        trip_id bigint,
        cust_movements_id bigint,

        started_at timestamp without time zone NOT NULL,
        finished_at timestamp without time zone NOT NULL,

        mode_detected character varying,
        mode_validated character varying,
        validated boolean,
        validated_at timestamp without time zone,

        geometry_raw geometry,
        geometry geometry,

        CONSTRAINT triplegs_pkey PRIMARY KEY (id)
    );

The **staypoints** table contains all stay points, i.e., places, where a user stayed
for a certain amount of time. They are linked to both a user, as well as a previous
and a next trip. Depending on the purpose and time spent, a staypoint can be an activity,
i.e., a meaningful destination of movement::

    CREATE TABLE staypoints (
        id bigint NOT NULL,
        user_id bigint NOT NULL,

        trip_id bigint,
        prev_trip_id bigint,
        next_trip_id bigint,

        started_at timestamp without time zone NOT NULL,
        finished_at timestamp without time zone NOT NULL,

        purpose_detected character varying,
        purpose_validated character varying,
        validated boolean,
        validated_at timestamp without time zone,

        place_id bigint,
        geometry_raw geometry,
        geometry geometry,
        activity boolean,

        CONSTRAINT staypoints_pkey PRIMARY KEY (id)
    );

The **trips** table contains all trips, i.e., collection of trip legs going from one 
activity (staypoint with ``activity==True``) to another. They are simply linked to a user.
They also have attributes (origin & destination) to link them to a table with place IDs.
Further, they can be part of one or more tours::

    CREATE TABLE trips (
        id bigint NOT NULL,
        user_id integer NOT NULL,

        started_at timestamp without time zone NOT NULL,
        finished_at timestamp without time zone NOT NULL,
        origin bigint,
        destination bigint,
        tour_id BIGINT[],

        CONSTRAINT trips_pkey PRIMARY KEY (id)
    );

The **customer movements** table contains all customer movements (Schoenfelder and 
Axhausen 2010), i.e., sequence of trip legs which use only public transport. They are 
linked to a user and a trip::

    CREATE TABLE cust_movements (
        id bigint NOT NULL,
        user_id integer NOT NULL,
        trip_id bigint NOT NULL,

        started_at timestamp without time zone NOT NULL,
        finished_at timestamp without time zone NOT NULL,
        provider varchar,

        CONSTRAINT cust_movements_pkey PRIMARY KEY (id)
    );

The **tours** table contains all tours (tours and journeys), i.e., sequence of trips 
which start and end at the same place (in case of ``journeys==home``). They are linked 
to a user::

    CREATE TABLE tours (
        id bigint NOT NULL,
        user_id integer NOT NULL,

        started_at timestamp without time zone NOT NULL,
        finished_at timestamp without time zone NOT NULL,
        origin geometry,
        journey bool,

        CONSTRAINT tours_pkey PRIMARY KEY (id)
    );

The **places** table contains all places, i.e., somehow created (e.g., from clustering
staypoints) meaningful locations::

    CREATE TABLE places (
        id bigint NOT NULL,
        geom geometry,
        user_id BIGINT,
        purpose VARCHAR,
        staypoint_id BIGINT[]
    );
