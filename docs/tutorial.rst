Tutorial
********

Thanks for your interest in *trackintel*. This tutorial shows you the most important 
functionalities and walks you through a typical usecase. You can find additional information
in the `examples folder <https://github.com/mie-lab/trackintel/tree/master/examples>`_. 

.. highlight:: python

First, install trackintel using::

    pip install trackintel

It is recommended to have a working `PostGIS database <https://postgis.net/>`_, as this
will make data persistence a lot easier. You can work with CSV files as well, or simply
keep everything in memory (e.g., within a Jupyter notebook session). If you decide to 
work with PostGIS, look at the set `SQL script 
<https://github.com/mie-lab/trackintel/blob/master/sql/create_tables_pg.sql>`_, which
will create all necessary tables. Simply execute it within a new/empty database.

To get you started, we put some trajectory data (in the form of positionfixes) in the 
`examples/data folder <https://github.com/mie-lab/trackintel/tree/master/examples/data>`_.
You can import one of these by executing the following steps::

    import trackintel as ti

    database_name = 'trackintel-tests'
    conn_string = 'postgresql://test:1234@localhost:5432/' + database_name

    pfs = ti.read_positionfixes_csv('examples/data/posmo_trajectory_2.csv', sep=';')
    pfs.as_positionfixes.to_postgis(conn_string, 'positionfixes', if_exists='append')

This will fill the positionfixes of ``posmo_trajectory_2.csv`` into the table
``positionfixes`` of the database ``trackintel-tests``. Make sure that you update the
connection string with your proper username and password. 

We now can for example plot the positionfixes::

    pfs.as_positionfixes.plot('positionfixes.png')

Of course, we can start our analysis, for example by detecting staypoints (locations
at which the user stayed for a certain amount of time)::

    from trackintel.geogr.distances import meters_to_decimal_degrees

    stps = pfs.as_positionfixes.extract_staypoints(method='sliding', 
        dist_threshold=100, time_threshold=60)
    stps.as_staypoints.plot(out_filename='staypoints.png',
        radius=meters_to_decimal_degrees(100, 47.5), positionfixes=pfs, plot_osm=True)

This will additionally plot the original positionfixes, as well as the underlying 
street network from OSM. We can for example continue by extracting and plotting locations 
(locations that "contain" multiple staypoints, i.e., are visited often by a user)::

    locs = spts.as_staypoints.extract_locations(method='dbscan', 
        epsilon=meters_to_decimal_degrees(120, 47.5), num_samples=3)
    locs.as_locations.plot(out_filename='locations.png', 
        radius=meters_to_decimal_degrees(120, 47.5), positionfixes=pfs, staypoints=spts, 
        staypoints_radius=meters_to_decimal_degrees(100, 47.5), plot_osm=True)
    
This will extract locations and plot them to a file called ``locations.png``, additionally 
plotting the original positionfixes and staypoints, as well as the street network.

As you can see, in *trackintel*, everything starts with positionfixes. From these 
you can generate ``staypoints`` and ``triplegs``, which in turn can be aggregated into
``locations`` and ``trips``. You can find the exact model description in the next section.