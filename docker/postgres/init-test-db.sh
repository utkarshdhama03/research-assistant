#!/bin/sh
# Runs once on first container start: create the test database and enable pgvector in both.
set -e
psql -v ON_ERROR_STOP=1 --username "$POSTGRES_USER" --dbname "$POSTGRES_DB" <<-SQL
	CREATE EXTENSION IF NOT EXISTS vector;
	CREATE DATABASE "$POSTGRES_TEST_DB";
SQL
psql -v ON_ERROR_STOP=1 --username "$POSTGRES_USER" --dbname "$POSTGRES_TEST_DB" <<-SQL
	CREATE EXTENSION IF NOT EXISTS vector;
SQL
