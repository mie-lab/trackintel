Data Model (SQL)
****************

For a general description of the data model, please refer to the 
:doc:`/modules/model`. You can download the 
`complete SQL script here <https://github.com/mie-lab/trackintel/blob/master/sql/create_tables_pg.sql>`_ 
in case you want to quickly set up a database.

.. highlight:: sql

The **users** table contains additional information about individual users::

    CREATE TABLE users (
        -- Common to all tables.
        id bigint NOT NULL,

        -- Specific attributes.
        -- The attributes contain additional information that might be given for each user. This
        -- could be demographic information, such as age, gender, or income. 
        attributes json,

        -- Spatial attributes.
        geom_home geometry(Point, 4326),
        geom_work geometry(Point, 4326),

        -- Constraints.
        CONSTRAINT users_pkey PRIMARY KEY (id)
    );

The **positionfixes** table contains all positionfixes of all users. They are not 
only linked to a user, but also to a trip leg or a staypoint::

    CREATE TABLE positionfixes (
        -- Common to all tables.
        id bigint NOT NULL,
        user_id bigint NOT NULL,

        -- References to foreign tables.
        tripleg_id bigint,
        staypoint_id bigint,

        -- Temporal attributes.
        tracked_at timestamp without time zone NOT NULL,

        -- Specific attributes.
        accuracy double precision,
        tracking_tech character(12),
        -- The context contains additional information that might be filled in by trackintel.
        -- This could include things such as the temperature, public transport stops in vicinity, etc.
        context json,

        -- Spatial attributes.
        elevation double precision,
        geom geometry(Point, 4326),

        -- Constraints.
        CONSTRAINT positionfixes_pkey PRIMARY KEY (id)
    );

The **staypoints** table contains all stay points, i.e., places where a user stayed
for a certain amount of time. They are linked to a user, as well as (potentially) to a trip
and place. Depending on the purpose and time spent, a staypoint can be an *activity*,
i.e., a meaningful destination of movement::

    CREATE TABLE staypoints (
        -- Common to all tables.
        id bigint NOT NULL,
        user_id bigint NOT NULL,

        -- References to foreign tables.
        trip_id bigint,
        place_id bigint,

        -- Temporal attributes.
        started_at timestamp without time zone NOT NULL,
        finished_at timestamp without time zone NOT NULL,
        
        -- Attributes related to the activity performed at the staypoint.
        purpose_detected character varying,
        purpose_validated character varying,
        validated boolean,
        validated_at timestamp without time zone,
        activity boolean,

        -- Specific attributes.
        -- The radius is an approximation of how far the positionfixes that made up this staypoint
        -- are scattered around the center (geom) of it.
        radius double precision,
        -- The context contains additional information that might be filled in by trackintel.
        -- This could include things such as the temperature, public transport stops in vicinity, etc.
        context json,

        -- Spatial attributes.
        elevation double precision,
        geom geometry(Point, 4326),

        -- Constraints.
        CONSTRAINT staypoints_pkey PRIMARY KEY (id)
    );

The **triplegs** table contains all trip legs, i.e., journeys that have been taken 
with a single mode of transport. They are linked to both a user, as well as a trip 
and if applicable, a public transport case::

    CREATE TABLE triplegs (
        -- Common to all tables.
        id bigint NOT NULL,
        user_id bigint NOT NULL,

        -- References to foreign tables.
        trip_id bigint,

        -- Temporal attributes.
        started_at timestamp without time zone NOT NULL,
        finished_at timestamp without time zone NOT NULL,

        -- Attributes related to the transport mode used for this trip leg.
        mode_detected character varying,
        mode_validated character varying,
        validated boolean,
        validated_at timestamp without time zone,

        -- Specific attributes.
        -- The context contains additional information that might be filled in by trackintel.
        -- This could include things such as the temperature, public transport stops in vicinity, etc.
        context json,

        -- Spatial attributes.
        -- The raw geometry is unprocessed, directly made up from the positionfixes. The column
        -- 'geom' contains processed (e.g., smoothened, map matched, etc.) data.
        geom_raw geometry(Linestring, 4326),
        geom geometry(Linestring, 4326),

        -- Constraints.
        CONSTRAINT triplegs_pkey PRIMARY KEY (id)
    );

The **places** table contains all places, i.e., somehow created (e.g., from clustering
staypoints) meaningful locations::

    CREATE TABLE places (
        -- Common to all tables.
        id bigint NOT NULL,
        user_id bigint,

        -- Specific attributes.
        -- The radius is an approximation of how far the staypoints that made up this place
        -- are scattered around the center (geom) of it.
        radius double precision,
        -- The context contains additional information that might be filled in by trackintel.
        -- This could include things such as the temperature, public transport stops in vicinity, etc.
        context json,
        
        -- Spatial attributes.
        elevation double precision,
        geom geometry(Polygon, 4326),

        -- Constraints.
        CONSTRAINT places_pkey PRIMARY KEY (id)
    );

The **trips** table contains all trips, i.e., collection of trip legs going from one 
activity (staypoint with ``activity==True``) to another. They are simply linked to a user::

    CREATE TABLE trips (
        -- Common to all tables.
        id bigint NOT NULL,
        user_id integer NOT NULL,

        -- References to foreign tables.
        origin_staypoint_id bigint,
        destination_staypoint_id bigint,

        -- Temporal attributes.
        started_at timestamp without time zone NOT NULL,
        finished_at timestamp without time zone NOT NULL,
        
        -- Specific attributes.
        -- The context contains additional information that might be filled in by trackintel.
        -- This could include things such as the temperature, public transport stops in vicinity, etc.
        context json,

        -- Constraints.
        CONSTRAINT trips_pkey PRIMARY KEY (id)
    );

The **tours** table contains all tours, i.e., sequence of trips which start and end 
at the same place (in case of ``journey==True`` this place is *home*). 
They are linked to a user::

    CREATE TABLE tours (
        -- Common to all tables.
        id bigint NOT NULL,
        user_id integer NOT NULL,

        -- References to foreign tables.
        origin_destination_place_id bigint,

        -- Temporal attributes.
        started_at timestamp without time zone NOT NULL,
        finished_at timestamp without time zone NOT NULL,
        
        -- Specific attributes.
        journey bool,
        -- The context contains additional information that might be filled in by trackintel.
        -- This could include things such as the temperature, public transport stops in vicinity, etc.
        context json,

        -- Constraints.
        CONSTRAINT tours_pkey PRIMARY KEY (id)
    );


References
==========

None.