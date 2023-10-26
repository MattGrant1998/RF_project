import xarray as xr
import pandas as pd

file = '/g/data/w97/mg5624/RF_project/drought_prediction/full_model/drought_prediction_dataset_full_model.nc'
ds = xr.open_dataset(file)

ds['time'] = pd.DatetimeIndex(ds['time'].values)

ds.to_netcdf('/g/data/w97/mg5624/RF_project/drought_prediction/full_model/drought_prediction_dataset_full_model1.nc')

file = '/g/data/w97/mg5624/RF_project/drought_prediction/long_ts_model/drought_prediction_dataset_long_ts_model.nc'
ds = xr.open_dataset(file)

ds['time'] = pd.DatetimeIndex(ds['time'].values)

ds.to_netcdf('/g/data/w97/mg5624/RF_project/drought_prediction/long_ts_model/drought_prediction_dataset_long_ts_model1.nc')
