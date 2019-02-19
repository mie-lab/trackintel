-- SQL to create tables in PostgreSQL/PostGIS.

CREATE EXTENSION PostGIS;

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

    started_at timestamp without time zone NOT NULL,
    finished_at timestamp without time zone NOT NULL,
    origin geometry,
    journey bool,

    CONSTRAINT tours_pkey PRIMARY KEY (id)
);

CREATE TABLE places (
    id bigint NOT NULL,
    geom geometry,
    user_id BIGINT,
    purpose VARCHAR,
    staypoint_id BIGINT[]
);