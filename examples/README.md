# Examples

These examples show several common usecases of *trackintel*.

## Data

The example data (found under `examples/data`) were generated or tracked using different tools:

* `gpsies_trajectory.csv` manually generated using [GPSies](https://www.gpsies.com).

* `posmo_trajectory.csv` automatically tracked using [POSMO Segments](https://posmo.datamap.io/), see also [Android App Store](https://play.google.com/store/apps/details?id=io.datamap.posmo_segments) and [Apple App Store](https://itunes.apple.com/us/app/posmo-segments/id1450602777).

* *Not available yet*: `moves_trajectory.csv` automatically tracked using [Moves (discontinued)](https://www.moves-app.com/).

* `geolife_trajectory.csv` is from the widely used [Geolife trajectory dataset](https://www.microsoft.com/en-us/download/details.aspx?id=52367) [1-3].

* *Not available yet*: `myway_trajectory.csv` is tracked by the [MyWay app](https://www.sbb.ch/de/fahrplan/mobile-fahrplaene/mobile-apps/myway.html) from the [Swiss Federal Railways (SBB)](https://www.sbb.ch), basically running the [MotionTag](https://motion-tag.com/en/) software.

* `google_trajectory.csv` is a random excerpt from a location track by [Google Maps](https://www.google.ch/maps).

## Usage

Run any example as:

```bash
python preprocessing.py
python import_export_postgis.py
```

Several examples, such as `preprocessing.py` will generate output plots in the `examples/out` directory. For examples involving a database connection, you can adapt the `config.json` file.

## References

[1] Yu Zheng, Lizhu Zhang, Xing Xie, Wei-Ying Ma. Mining interesting locations and travel sequences from GPS trajectories. In Proceedings of International conference on World Wild Web (WWW 2009), Madrid Spain. ACM Press: 791-800.

[2] Yu Zheng, Quannan Li, Yukun Chen, Xing Xie, Wei-Ying Ma. Understanding Mobility Based on GPS Data. In Proceedings of ACM conference on Ubiquitous Computing (UbiComp 2008), Seoul, Korea. ACM Press: 312-321.

[3] Yu Zheng, Xing Xie, Wei-Ying Ma, GeoLife: A Collaborative Social Networking Service among User, location and trajectory. Invited paper, in IEEE Data Engineering Bulletin. 33, 2, 2010, pp. 32-40.