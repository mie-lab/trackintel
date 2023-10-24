import logging
import traceback

import psycopg2
from psycopg2.extensions import ISOLATION_LEVEL_AUTOCOMMIT
import trackintel as ti


# This script will remove the following table from your database!
# It will then, however, repopulate it with new data.
database_name = "trackintel-tests"
logging.basicConfig(filename="log/setup_example_database.log", level=logging.INFO, filemode="w")

try:
    con = psycopg2.connect(user="test", host="localhost", port="5432", password="1234", dbname="postgres")
    con.set_isolation_level(ISOLATION_LEVEL_AUTOCOMMIT)
    cur = con.cursor()
    cur.execute(f"""SELECT pg_terminate_backend(pid) FROM pg_stat_activity WHERE datname='{database_name}'""")
    cur.execute(f"""DROP DATABASE IF EXISTS "{database_name}" """)
    cur.execute(f"""CREATE DATABASE "{database_name}" """)
    con.commit()
    con.close()

    con = psycopg2.connect(user="test", host="localhost", port="5432", password="1234", dbname=database_name)
    cur = con.cursor()
    cur.execute(open("../sql/create_tables_pg.sql", "r").read())
    con.commit()
    con.close()
except Exception:
    print("I am unable to connect to the database")
    traceback.print_exc()

# Now we fill in some new data.
conn_string = "postgresql://test:1234@localhost:5432/" + database_name
pfs = ti.read_positionfixes_csv("data/posmo_trajectory_2.csv", sep=";")
pfs.to_postgis("positionfixes", conn_string, if_exists="append")

# We use the trackintel functionality to fill consecutive tables.
