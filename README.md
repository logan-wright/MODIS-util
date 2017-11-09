## MODIS-util

#### Additional packages:

- [matplotlib](https://matplotlib.org/)

- [cartopy](http://scitools.org.uk/cartopy/docs/v0.14/index.html)

- [NumPy](http://www.numpy.org/)

---

#### Description

GeoMeta data of MODIS provided by LAADS, which is a geographic text
file that contains MODIS granule corner-point latitudes and longitudes,
is used here to define the MODIS granule projected on Earth. The GeoMeta data
if a non-offical MODIS archive product and is restricted access only via
anonymous [ftp site](`ftp://ladsweb.nascom.nasa.gov/geoMeta`).

This code takes in the corner-point lon/lat of MODIS granule and defines a polygon
on Earth through `matplotlib.path`. Then uses the function of `matplotlib.path.contains_points`
to check whether the input data points are within the defined polygon.

In order to best define a polygon on Earth through limited 4 corner points, the orthographic
projection centered at polygon centroid is used for each MODIS granule.

---

#### Available functions

- `READ_GEOMETA` (function)

- `FTP_INIT` (class)

- `FIND_MODIS` (function)

- `EARTH_VIEW` (function)

- `DOWNLOAD_MODIS` (function)

---

#### How to use

```python
import datetime
from modis-util import FIND_MODIS, FTP_INIT

# assume we read in some flight track info here
#
#  - time in hour: tmhr
#  - longitude: lon
#  - latitude: lat
#
# on date 2014-10-17

date = datetime.datetime(2014, 9, 11)

# the following function call will find the Aqua MODIS granules (from 10:00AM to 12:00AM)
# that contain the input flight track points on 2014-10-17
granules = FIND_MODIS(date, tmhr, lon, lat, satID='aqua', tmhr_range=[10.0, 12.0])

# define a ftp instance
ftp_init = FTP_INIT(granules)

# the following function will download the found granules
DOWNLOAD_MODIS(ftp_init)
```
