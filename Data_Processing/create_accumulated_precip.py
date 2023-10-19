import xarray as xr
import pandas as pd
import processing_functions
import os

# Load AGCD precip data
precip_ds = xr.open_dataset(
    '/g/data/w97/amu561/AGCD_drought_metrics/AGCD_1900_2021/AGCD_v1_precip_total_r005_monthly_1900_2021.nc'
)
precip = precip_ds.precip

def rename_and_save_precipitation_dataarray(precip_ds):
    """
    Takes the precip datarray from precip dataset, renames it to Precipitation and saves it on it as a netcdf.

    Args:
        precip_ds (xr.Dataset): dataset of AGCD precip
    """
    precip = precip_ds.precip
    precip = precip.rename('Precipitation')
    precip = processing_functions.constrain_to_australia(precip)
    precip.to_netcdf('/g/data/w97/mg5624/RF_project/Precipitation/AGCD/AGCD_v1_precip_total_r005_monthly_1900_2021.nc')

def create_n_month_accumulated_precip_dataarray(precip, n_months):
    """
    Creates a dataarray of n-month accumulated precipitation.
    Saves resulting accumulated precip as netcdf file.

    Args:
        precip (xr.DataArray): data array of monthly precipitation
        n_months (int): number of months to accumulate the precipitation over
    """ 
    acc_precip = precip.rolling(time=n_months).sum()
    acc_precip = acc_precip.rename(f'Acc_{n_months}-Month_Precipitation')
    filepath = '/g/data/w97/mg5624/RF_project/Precipitation/AGCD/'
    filename = f'AGCD_v1_precip_total_r005_{n_months}monthly_1900_2021.nc'
    acc_precip.to_netcdf(filepath + filename)


def main():
    rename_and_save_precipitation_dataarray(precip_ds)
    n_months = [3, 6, 12, 24, 36, 48]
    for n in n_months:
        create_n_month_accumulated_precip_dataarray(precip, n)


if __name__ == "__main__":
    main()
    
# precip_3monthly = precip.rolling(time=3).sum()
# precip_12monthly = precip.rolling(time=12).sum()

# precip_3monthly = precip_3monthly.rename('acc 3-month precip')
# precip_12monthly = precip_12monthly.rename('acc 12-month precip')

# filepath = ('/g/data/w97/mg5624/RF_project/Precipitation/AGCD/')

# if not os.path.exists(filepath):
#     os.makedirs(filepath)

# filename_3monthly = 'AGCD_v1_precip_total_r005_3monthly_1900_2021.nc'
# filename_annual = 'AGCD_v1_precip_total_r005_annual_1900_2021.nc'

# precip_3monthly.to_netcdf(path=filepath + filename_3monthly)
# precip_annual.to_netcdf(path=filepath + filename_annual)
