import xarray as xr
import pandas as pd
import processing_functions
import os

# Load AGCD precip data
precip_ds = xr.open_dataset(
    '/g/data/w97/amu561/AGCD_drought_metrics/AGCD_1900_2021/AGCD_v1_precip_total_r005_monthly_1900_2021.nc'
)

precip = precip_ds.precip

precip_aus = processing_functions.constrain_to_australia(precip)

precip_3monthly = precip.rolling(time=3).sum()
precip_annual = precip.rolling(time=12).sum()

filepath = ('/g/data/w97/mg5624/RF_project/Precipitation/AGCD/')

if not os.path.exists(filepath):
    os.makedirs(filepath)

filename_3monthly = 'AGCD_v1_precip_total_r005_3monthly_1900_2021.nc'
filename_annual = 'AGCD_v1_precip_total_r005_annual_1900_2021.nc'

precip_3monthly.to_netcdf(path=filepath + filename_3monthly)
precip_annual.to_netcdf(path=filepath + filename_annual)
