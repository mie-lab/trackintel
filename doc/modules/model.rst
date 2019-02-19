Model
*****

In trackintel, **tracking data** is split into several classes. It is not generally 
assumed that data is already available in all these classes, instead, trackintel
provides functionality to generate everything starting from the raw GPS positionfix data 
(consisting of ``(longitude, latitude, accuracy, timestamp, user_id)`` tuples).

* **positionfixes**: Raw GPS data
* **triplegs**: Segments covered with one mode of transport
* **trips**: Segments between consecutive staypoints
* **customer movements**: Sequences of triplegs which use only public transport
* **tours**: Sequences of trips which start and end at the same place (if ``journey`` 
  is set to ``True``, this place is *home*)
* **staypoints**: Locations where a user spent a minimal time
* **places**: Clustered staypoints

Additionally, some of the more time-consuming functions of trackintel generate logging 
data, as well as extracted features data, and they assume more data about geographic 
features or characteristics of transport modes are available. A detailed (and 
SQL-specific) explanation of the different classes can be found under 
:doc:`/content/data_model_sql`.
