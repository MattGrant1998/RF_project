import xarray as xr
import processing_functions
from processing_functions import regrid_to_5km_grid
import os


def constrain_and_regrid_water_storage():
    """
    Constrains water storage to Australia and regrids it to 5km grid. Saves netcdf in my data driectory.
    """
    # Load water_storage dataset and change coordinates to lat longs
    water_storage_ds = xr.open_dataset('/g/data/w97/mg5624/RF_project/Water_Storage/01_monthly_grids_ensemble_means_allmodels/GRACE_REC_v03_JPL_MSWEP_monthly_ensemble_mean.nc')
    water_storage = water_storage_ds.rec_ensemble_mean
    print(water_storage)
    # Constrain water_storage data to cover just Australia
    water_storage_aus = water_storage.sel(lon=slice(112, 155), lat=slice(-45, -9))
    
    # regrid water_storage to precip
    interpolated_water_storage_aus = regrid_to_5km_grid(water_storage_aus)
    
    # Save new "high-res" water_storage data
    interpolated_water_storage_aus.to_netcdf(
        processing_functions.my_data_dir + 'RF_project/Water_Storage/GRACE_REC_v03_JPL_MSWEP_monthly_ensemble_mean_Australia_0.05grid.nc'
    )


def calculate_CWS():
    """
    Calculates change in water storage from the water storage data. Saves resulting netcdf to my data driectory.   
    """
    water_storage = xr.open_dataarray(
        processing_functions.my_data_dir + 'RF_project/Water_Storage/GRACE_REC_v03_JPL_MSWEP_monthly_ensemble_mean_Australia_0.05grid.nc'
    )
  
    # Calculate Change in Water Storage (CWS)
    CWS = water_storage.diff(dim='time', n=1)

    print(CWS)
    CWS.to_netcdf(
        processing_functions.my_data_dir + 'RF_project/Water_Storage/CWS_v03_JPL_MSWEP_monthly_ensemble_mean_Australia_0.05grid.nc' 
    )

def main():
    # constrain_and_regrid_water_storage()
    calculate_CWS()

if __name__ == "__main__":
    main()    
    