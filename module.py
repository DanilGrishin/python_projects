import pandas as pd
import numpy as np
from numpy import sin, cos, arccos, pi, round
import pickle
from geopy.geocoders import Nominatim

def apart_coord(adress):

  geolocator = Nominatim(user_agent="http")
  location = geolocator.geocode(adress)

  lat_apart = location.latitude
  lon_apart = location.longitude

  dict_coord = {
      'lat_apart': np.array(lat_apart),
      'lon_apart': np.array(lon_apart),
  }

  return dict_coord

def rad2deg(radians):
    degrees = radians * 180 / pi
    return degrees

def deg2rad(degrees):
    radians = degrees * pi / 180
    return radians


def getDistanceBetweenPointsNew(latitude1, longitude1, latitude2, longitude2, unit='meters'):
    theta = longitude1 - longitude2

    distance = 60 * 1.1515 * rad2deg(
        arccos(
            np.around(
                (sin(deg2rad(latitude1)) * sin(deg2rad(latitude2))) +
                (cos(deg2rad(latitude1)) * cos(deg2rad(latitude2)) * cos(deg2rad(theta)))
                , decimals=15)
        )
    )

    if unit == 'miles':
        return round(distance, 2)
    if unit == 'meters':
        return (distance * 1.609344)*1000

def apart_cost_value(self, dict, adress):

  model = pickle.load(open("model", "rb"))

  lats = np.array(pd.read_excel('coordinates.xlsx')['lat'])
  lons = np.array(pd.read_excel('coordinates.xlsx')['lon'])

  dist = getDistanceBetweenPointsNew(latitude1 = apart_coord(adress)['lat_apart'], longitude1 = apart_coord(adress)['lon_apart'],
                                          latitude2 = lats, longitude2 = lons, unit='meters')

  dist_df = pd.DataFrame(dist, columns= ['dist'])
  dist_df['object'] = pd.read_excel('coordinates.xlsx')['type']

  min_dist = dist_df.groupby(['object']).min()['dist']

  data_model = pd.concat([pd.DataFrame(dict, index=[0]).T, min_dist]).T
  print(np.exp(model.predict(data_model)))
