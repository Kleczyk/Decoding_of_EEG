
CREATE EXTENSION IF NOT EXISTS timescaledb;

CREATE TABLE IF NOT EXISTS training_data (
    id SERIAL PRIMARY KEY,
    cwt_data BYTEA,
    target INTEGER,
    time TIMESTAMPTZ NOT NULL
);

CREATE TABLE IF NOT EXISTS validation_data (
    id SERIAL PRIMARY KEY,
    cwt_data BYTEA,
    target INTEGER,
    time TIMESTAMPTZ NOT NULL
);

SELECT create_hypertable('training_data', 'time');
SELECT create_hypertable('validation_data', 'time');
