import xarray as xr
import numpy as np
import pandas as pd
import os

# Define some filepaths that will be used a lot
home_dir = '/home/561/mg5624/RF_project/'
my_data_dir = '/g/data/w97/mg5624/'
shared_data_dir = '/g/data/w97/Shared_data/Observations/'

def rename_coord_titles_to_lat_long(dataset):
    """
    Changes the titles of the coordinates to lat long to keep it consistent
    Args:
    dataset (xr.DataSet): dataset with incorrect coordinate titles
    """
    # Define mapping from old to new name
    
    mapping = {
        'longitude': 'lon',
        'latitude': 'lat'
    }
    
    renamed_dataset = dataset.rename(mapping)

    return renamed_dataset


def constrain_to_australia(dataarray):
    """
    Takes an xarray data array and constrains it to just cover Australia - helping save on computation
    Args:
    dataset (xr.DataArray): dataset which has australia included in it
    """
    
    if dataarray.coords['lat'][0].data < 0:
        lat1 = -45
        lat2 = -9
    else:
        lat1 = -9
        lat2 = -45
        
    # Constrain runoff data to cover just Australia
    aus_dataarray = dataarray.sel(lon=slice(112, 155), lat=slice(lat1, lat2))

    return aus_dataarray


def regrid_to_5km_grid(dataarray):
    """
    Regrids data to 0.05 degree grid.
    Args:
    dataarray (xr.DataArray): xarray array over Australia that requires regridding
    """
    # Load in precip dataset - which we want to regrid to
    precip_ds = xr.open_dataset('/g/data/w97/amu561/AGCD_drought_metrics/AGCD_1900_2021/AGCD_v1_precip_total_r005_monthly_1900_2021.nc')
    precip = precip_ds.precip
    
    # regrid runoff to precip
    interpolated_dataarray = dataarray.interp_like(precip, method='nearest')

    return interpolated_dataarray


def convert_average_mm_per_day_to_mm_per_month(runoff_dataarray):
    """
    Converts runoff data from mm/day to mm/month.
    Args:
    runoff_dataarray (xr.DataArray): Runoff data array in mm/day
    """
    days_in_month_dict = {1:31, 2:28, 3:31, 4:30, 5:31, 6:30, 7:31, 8:31, 9:30, 10:31, 11:30, 12:31}
    time = runoff.coords['time'].data
    dates = pd.to_datetime(time)
    months = dates.month
    years = dates.year

    # Create a list of the days in the month of each time stamp in the runoff data
    leap_year = [check_leap_year(year) for year in years]
    days_in_month = [days_in_month_dict[month] for month in months]
    febs = [days==28 for days in days_in_month]
    days_in_month = [29 if is_leap and feb else days for is_leap, feb, days in zip(leap_year, febs, days_in_month)]

    # Find monthly runoff data by multiplying runoff daily data by the days in the month
    runoff_monthly_data = runoff.data * np.array(days_in_month)[:, np.newaxis, np.newaxis]

    # Save this data in a data array
    runoff_monthly = runoff.copy(data=runoff_monthly_data)
    runoff_monthly.attrs['units'] = 'mm/month'

    return runoff_monthly


def check_leap_year(year):
    """
    Checks if a year is a leap year or not. Returns True if so, Fasle if not.
    Args:
    year (int): year of interest
    """
    if (year % 4) == 0:
        if (year % 100) == 0:
            if (year % 400) == 0:
                return True
            else:
                return False
        else:
            return True
    else:
        return False
        