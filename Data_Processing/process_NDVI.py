import xarray as xr
import processing_functions
import numpy as np
import pandas as pd
import os

# def find_max_monthly_values(
filepath_in = '/g/data/w97/mg5624/RF_project/NDVI/orders/98eac55928c915bdb7c8ff216de8d980/Global_Veg_Greenness_GIMMS_3G/data/'


def perform_QC_on_NDVI(filepath_in):
    """
    Performs quality control checks on the NDVI data.

    Args:
        filepath_in (str): filepath to the raw NDVI data
    """
    NDVI_data = xr.open_mfdataset(filepath_in + '/*', combine='by_coords')
    NDVI_aus = processing_functions.constrain_to_australia(NDVI_data)

    NDVI = NDVI_aus.ndvi.compute()
    print("non nan values before QC:", NDVI.count())
    
    QC = NDVI_aus_regrid.percentile.compute()
    NDVI_QC1 = xr.where(QC < 3, NDVI, np.nan)
    # print(NDVI_missing, NDVI_missing.count())
    print('non-nan values after QC1:', NDVI_QC1.count())
    
    NDVI_QC2 = xr.where(QC < 2, NDVI, np.nan)
    print('non-nan values after QC2:', NDVI_QC2.count())
    # Remove values outside of valid range [-0.3, 1] or QC = 2 or 3 (where NDVI is taken from seasonal profile or missing)
    NDVI_valid = xr.where((NDVI >= -0.3) | (NDVI <= 1), NDVI, np.nan)
    print('non-nan values after manual QC2:', NDVI_valid.compute().count())
    

    
def combine_NDVI_files_over_australia(filepath_in, save_file=False, filepath_out=None):
    """
    Takes in NDVI files for various years of data and combines them all into one file.

    Args:
        filepath_in (str): filepath to where all the files are stored
        save_file (bool): choose whether to save the combined file
        filepath_out (str): filepath to save combined file to. Optional argument, must be defined 
        if save_file=True
        
    Returns:
        NDVI (xr.DataArray): dataarray of the combined timeseires for NDVI
    """
    NDVI_data = xr.open_mfdataset(filepath_in + '/*', combine='by_coords', engine='netcdf4')
    NDVI_aus = processing_functions.constrain_to_australia(NDVI_data)
    NDVI_aus_regrid = processing_functions.regrid_to_5km_grid(NDVI_aus)

    
    if save_file:
        if filepath_out == None:
            raise ValueError(
                'Filepath for saving file unspecified, please specify fielpath when calling this function with save_file=True'
            )
        else:
            if not os.path.exists(fileapth_out):
                os.makedirs(filepath_out)
            filename = 'ndvi3g_geo_v1_1_1982-2022_bimonthly_0.05grid.nc'
            NDVI_valid.to_netcdf(filepath_out)    
    return NDVI_valid



filepath = '/g/data/w97/mg5624/RF_project/NDVI/australia/'
filename = 'ndvi3g_geo_v1_1_1982-2022_bimonthly_0.05grid.nc'
perform_QC_on_NDVI('/g/data/w97/mg5624/RF_project/NDVI/orders/98eac55928c915bdb7c8ff216de8d980/Global_Veg_Greenness_GIMMS_3G/')
# combine_NDVI_files_over_australia(filepath_in)

# NDVI = NDVI.compute()

# print(NDVI)
# NDVI = NDVI.where(NDVI != -5000, drop=True)
# print(NDVI)

# NDVI_lower_invalid = NDVI.where(NDVI < -0.3, drop=True)
# print(NDVI_lower_invalid)

# NDVI_upper_invalid = NDVI.where(NDVI > 1, drop=True)
# print(NDVI_upper_invalid)

# print(nan_values_upper)
# print(nan_values_lower)
