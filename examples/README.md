# Examples

These examples show several common usecases of *trackintel*.

## Data

The example data (found under `examples/data`) were generated or tracked using different tools:

* `gpsies_trajectory.csv` manually generated using [GPSies](https://www.gpsies.com).

* `posmo_trajectory.csv` automatically tracked using [POSMO Segments](https://posmo.datamap.io/), see also [Android App Store](https://play.google.com/store/apps/details?id=io.datamap.posmo_segments) and [Apple App Store](https://itunes.apple.com/us/app/posmo-segments/id1450602777).

* `geolife_trajectory.csv` and `pfs_tutorial.geojson` is from the widely used [Geolife trajectory dataset](https://www.microsoft.com/en-us/download/details.aspx?id=52367) [1-3].

* `google_trajectory.csv` is a random excerpt from a location track by [Google Maps](https://www.google.ch/maps).

## Usage

Run any example as:

```bash
python preprocess_trajectories.py
python import_export_postgis.py
python setup_example_database.py
```

Several examples, such as `preprocess_trajectories.py` will generate output plots in the `examples/out` directory.
For examples involving a database connection, you can adapt the `config.json` file.

## References

[1] Yu Zheng, Lizhu Zhang, Xing Xie, Wei-Ying Ma. Mining interesting locations and travel sequences from GPS trajectories. In Proceedings of International conference on World Wild Web (WWW 2009), Madrid Spain. ACM Press: 791-800.

[2] Yu Zheng, Quannan Li, Yukun Chen, Xing Xie, Wei-Ying Ma. Understanding Mobility Based on GPS Data. In Proceedings of ACM conference on Ubiquitous Computing (UbiComp 2008), Seoul, Korea. ACM Press: 312-321.

[3] Yu Zheng, Xing Xie, Wei-Ying Ma, GeoLife: A Collaborative Social Networking Service among User, location and trajectory. Invited paper, in IEEE Data Engineering Bulletin. 33, 2, 2010, pp. 32-40.