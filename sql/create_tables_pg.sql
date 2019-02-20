-- SQL to create tables in PostgreSQL/PostGIS.

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

    -- Specific attributes.
    elevation double precision,
    accuracy double precision,

    -- Spatial attributes.
    geom geometry(Point, 4326),

    -- Constraints.
    CONSTRAINT positionfixes_pkey PRIMARY KEY (id)
);

CREATE TABLE staypoints (
    id bigint NOT NULL,
    user_id bigint NOT NULL,

    trip_id bigint,
    place_id bigint,

    started_at timestamp without time zone NOT NULL,
    finished_at timestamp without time zone NOT NULL,
    
    activity boolean,

    purpose_detected character varying,
    purpose_validated character varying,
    validated boolean,
    validated_at timestamp without time zone,

    geom_raw geometry(Point, 4326),
    geom geometry(Point, 4326),

    CONSTRAINT staypoints_pkey PRIMARY KEY (id)
);

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

    geom_raw geometry(Linestring, 4326),
    geom geometry(Linestring, 4326),

    CONSTRAINT triplegs_pkey PRIMARY KEY (id)
);

CREATE TABLE trips (
    id bigint NOT NULL,
    user_id integer NOT NULL,

    origin_id bigint,
    destination_id bigint,
    tour_id BIGINT[],

    started_at timestamp without time zone NOT NULL,
    finished_at timestamp without time zone NOT NULL,

    CONSTRAINT trips_pkey PRIMARY KEY (id)
);

CREATE TABLE cust_movements (
    id bigint NOT NULL,
    user_id integer NOT NULL,

    trip_id bigint NOT NULL,

    started_at timestamp without time zone NOT NULL,
    finished_at timestamp without time zone NOT NULL,

    provider varchar,

    CONSTRAINT cust_movements_pkey PRIMARY KEY (id)
);

CREATE TABLE tours (
    id bigint NOT NULL,
    user_id integer NOT NULL,

    origin_destination_id bigint,

    started_at timestamp without time zone NOT NULL,
    finished_at timestamp without time zone NOT NULL,
    
    journey bool,

    CONSTRAINT tours_pkey PRIMARY KEY (id)
);

CREATE TABLE places (
    id bigint NOT NULL,
    user_id bigint,

    purpose VARCHAR,
    
    geom geometry(Polygon, 4326)
);

CREATE TABLE users (
    id bigint NOT NULL,

    geom_home geometry(Point, 4326),
    geom_work geometry(Point, 4326),

    CONSTRAINT users_pkey PRIMARY KEY (id)
);