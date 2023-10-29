Tutorial
********

Thanks for your interest in *trackintel*. This tutorial shows you the most important 
functionalities and walks you through a typical usecase. You can find additional information
in the `examples folder <https://github.com/mie-lab/trackintel/tree/master/examples>`_. 

.. highlight:: python

First, install trackintel using::

    pip install trackintel

Test the install using::

    import trackintel as ti
    ti.print_version()

This page focuses on setting the trackintel environment under a working 
`PostGIS database <https://postgis.net/>`_; this will make data persistence a lot easier. 
For working with CSV files or Geopandas GeoDataframes, please check the 
`trackintel_basic_tutorial <https://github.com/mie-lab/trackintel/blob/master/examples/trackintel_basic_tutorial.ipynb>`_
or run it directly in MyBinder notebook |MyBinder|.

.. |MyBinder| image:: https://mybinder.org/badge_logo.svg 
    :target: https://mybinder.org/v2/gh/mie-lab/trackintel/HEAD?filepath=%2Fexamples%2Ftrackintel_basic_tutorial.ipynb

If you decide to work with PostGIS, look at the set `SQL script 
<https://github.com/mie-lab/trackintel/blob/master/sql/create_tables_pg.sql>`_, which
will create all necessary tables. Simply execute it within a new/empty database.

To get you started, we put some trajectory data (in the form of positionfixes) in the 
`examples/data folder <https://github.com/mie-lab/trackintel/tree/master/examples/data>`_.
You can import one of these by executing the following steps::

    import trackintel as ti

    database_name = 'trackintel-tests'
    conn_string = 'postgresql://test:1234@localhost:5432/' + database_name

    pfs = ti.read_positionfixes_csv('examples/data/posmo_trajectory_2.csv', sep=';')
    pfs.to_postgis('positionfixes', conn_string, if_exists='append')

This will fill the positionfixes of ``posmo_trajectory_2.csv`` into the table
``positionfixes`` of the database ``trackintel-tests``. Make sure that you update the
connection string with your proper username and password. 

We now can for example plot the positionfixes::

    ti.plot(positionfixes=pfs, filename="positionfixes.png")

Of course, we can start our analysis, for example by detecting staypoints (aggregated positionfixes 
where the user stayed for a certain amount of time)::

    _, sp = pfs.generate_staypoints(method='sliding')
    ti.plot(filename="staypoints.png", radius_sp=10, staypoints=sp, positionfixes=pfs, plot_osm=True)

This will additionally plot the original positionfixes, as well as the underlying 
street network from OSM. We can for example continue by extracting and plotting locations 
(locations that "contain" multiple staypoints, i.e., are visited often by a user)::

    _, locs = sp.generate_locations(method='dbscan', epsilon=100, num_samples=1)
    ti.plot(filename="locations.png", locations=locs, radius_locs=125, positionfixes=pfs,
            staypoints=sp, radius_sp=100, plot_osm=True)
    
This will extract locations and plot them to a file called ``locations.png``, additionally 
plotting the original positionfixes and staypoints, as well as the street network.

As you can see, in *trackintel*, everything starts with positionfixes. From these 
you can generate ``staypoints`` and ``triplegs``, which in turn can be aggregated into
``locations`` and ``trips``. You can find the exact model description in the 
:doc:`/modules/model` page.