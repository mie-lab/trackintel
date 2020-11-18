-- SQL to create tables in PostgreSQL/PostGIS.
-- This is the same as described in docs/content/data_model_sql.rst.

CREATE EXTENSION PostGIS;

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

CREATE TABLE locations (
    -- Common to all tables.
    id bigint NOT NULL,
    user_id bigint,

    -- Specific attributes.
    -- The context contains additional information that might be filled in by trackintel.
    -- This could include things such as the temperature, public transport stops in vicinity, etc.
    context json,
    
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
    
    -- Specific attributes.
    -- The context contains additional information that might be filled in by trackintel.
    -- This could include things such as the temperature, public transport stops in vicinity, etc.
    context json,

    -- Constraints.
    CONSTRAINT trips_pkey PRIMARY KEY (id)
);

CREATE TABLE tours (
    -- Common to all tables.
    id bigint NOT NULL,
    user_id integer NOT NULL,

    -- References to foreign tables.
    origin_destination_location_id bigint,

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
