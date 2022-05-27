-- SQL to create tables in PostgreSQL/PostGIS.
-- This is the same as described in docs/content/data_model_sql.rst.

CREATE EXTENSION PostGIS;

CREATE TABLE positionfixes (
    -- Common to all tables.
    id bigint NOT NULL,
    user_id bigint NOT NULL,

    -- References to foreign tables.
    tripleg_id bigint,
    staypoint_id bigint,

    -- Temporal attributes.
    tracked_at timestamp without time zone NOT NULL,

    -- Spatial attributes.
    elevation double precision,
    geom geometry(Point, 4326),

    -- Constraints.
    CONSTRAINT positionfixes_pkey PRIMARY KEY (id)
);

CREATE TABLE staypoints (
    -- Common to all tables.
    id bigint NOT NULL,
    user_id bigint NOT NULL,

    -- References to foreign tables.
    trip_id bigint,
    location_id bigint,

    -- Temporal attributes.
    started_at timestamp without time zone NOT NULL,
    finished_at timestamp without time zone NOT NULL,
    
    -- Attributes related to the activity performed at the staypoint.
    purpose_detected character varying,
    purpose_validated character varying,
    validated boolean,
    validated_at timestamp without time zone,
    activity boolean,

    -- Spatial attributes.
    elevation double precision,
    geom geometry(Point, 4326),

    -- Constraints.
    CONSTRAINT staypoints_pkey PRIMARY KEY (id)
);

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

    -- Spatial attributes.
    -- The raw geometry is unprocessed, directly made up from the positionfixes. The column
    -- 'geom' contains processed (e.g., smoothened, map matched, etc.) data.
    geom_raw geometry(Linestring, 4326),
    geom geometry(Linestring, 4326),

    -- Constraints.
    CONSTRAINT triplegs_pkey PRIMARY KEY (id)
);

CREATE TABLE locations (
    -- Common to all tables.
    id bigint NOT NULL,
    user_id bigint,
    
    -- Spatial attributes.
    elevation double precision,
    extent geometry(Polygon, 4326),
    center geometry(Point, 4326),

    -- Constraints.
    CONSTRAINT locations_pkey PRIMARY KEY (id)
);

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

    -- Constraints.
    CONSTRAINT trips_pkey PRIMARY KEY (id)
);

CREATE TABLE tours (
    -- Common to all tables.
    id bigint NOT NULL,
    user_id integer NOT NULL,

    -- References to foreign tables.
    location_id bigint,

    -- Temporal attributes.
    started_at timestamp without time zone NOT NULL,
    finished_at timestamp without time zone NOT NULL,

    -- Constraints.
    CONSTRAINT tours_pkey PRIMARY KEY (id)
);
