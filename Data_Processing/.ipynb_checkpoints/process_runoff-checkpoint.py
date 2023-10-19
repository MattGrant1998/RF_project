import xarray as xr
import processing_functions


def process_runoff():
    """
    Takes unprocess runoff data array and changes coords to lat/long and regrids to same grid as precip data.
    """
    runoff_ds = xr.open_dataset('/g/data/iu04/australian-water-outlook/historical/v1/AWRALv7/processed/values/month/qtot.nc')
    runoff = runoff_ds.qtot
    runoff = processing_functions.rename_coord_titles_to_lat_long(runoff)
    regridded_runoff = processing_functions.regrid_to_5km_grid(runoff)
    regridded_runoff = regridded_runoff.rename('Runoff')
    # Save new "high-res" runoff data
    regridded_runoff.to_netcdf('/g/data/w97/mg5624/RF_project/Runoff/AWRA/AWRAv7_Runoff_month_1911_2023.nc')


def main():
    process_runoff()

if __name__ == "__main__":
    main()
