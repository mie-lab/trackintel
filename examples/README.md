# Examples

These examples show several common usecases of *trackintel*.

## Data

The example data (found under `examples/data`) were generated or tracked using different tools:

* `gpsies_trajectory.csv` manually generated using [GPSies](https://www.gpsies.com).

* `posmo_trajectory.csv` automatically tracked using [POSMO Segments](https://posmo.datamap.io/), see also [Android App Store](https://play.google.com/store/apps/details?id=io.datamap.posmo_segments) and [Apple App Store](https://itunes.apple.com/us/app/posmo-segments/id1450602777). 

* `moves_trajectory.csv` automatically tracked using [Moves (discontinued)](https://www.moves-app.com/).

## Usage

Run any example as:
```
python visualize_trajectories.py
```

Several examples, such as `visualize_trajectories.py` will generate output plots in the `examples/out` directory. For examples involving a database connection, you can adapt the `config.json` file.