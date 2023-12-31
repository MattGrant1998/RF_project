import xarray as xr
import processing_functions


def process_runoff():
    """
    Takes unprocessed runoff data array and changes coords to lat/long and regrids to same grid as precip data.
    """
    runoff_ds = xr.open_dataset('/g/data/iu04/australian-water-outlook/historical/v1/AWRALv7/processed/values/month/qtot.nc')
    runoff = runoff_ds.qtot
    runoff = processing_functions.rename_coord_titles_to_lat_long(runoff)
    runoff_const = processing_functions.constrain_to_australia(runoff)
    regridded_runoff = processing_functions.regrid_to_5km_grid(runoff_const)
    renamed_runoff = regridded_runoff.rename('Runoff')
    year_month_dt_runoff = processing_functions.set_time_coord_to_year_month_datetime(renamed_runoff)    
    year_month_dt_runoff.to_netcdf('/g/data/w97/mg5624/RF_project/Runoff/AWRA/AWRAv7_Runoff_month_1911_2023.nc')

    
def main():
    process_runoff()

if __name__ == "__main__":
    main()
