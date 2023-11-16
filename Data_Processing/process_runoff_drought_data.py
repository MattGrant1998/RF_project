import xarray as xr
import os
import pandas as pd
from processing_functions import constrain_to_australia, regrid_to_5km_grid


def fix_drought_metric_time(runoff_drought_data, data_with_time_from_1900_2021):
    """
    Changes the time coord of runoff_drought_data to that of data_with_correct_time.

    Args:
        runoff_drought_data (xr.DataArray): data array of drought/no drought defined by a runoff
        data_with_correct_time (xr.DataArray): data array with monthly time coord from 1900 to 2021

    Returns:
        runoff_drought_data (xr.DataArray): data array with monthly time coord from 1960 to 2020
    """
    full_time = data_with_time_from_1900_2021['time']
    runoff_time = full_time.sel(time=slice('1960', '2020'))

    runoff_drought_data['time'] = runoff_time

    return runoff_drought_data


def process_drought_metric_data(runoff_drought_data):
    """
    Constrains runoff data to defined Australia coords and regrids to 5km.
    Saves file to data directory.

    Args:
        runoff_drought_data (xr.DataArray): data array with monthly time coord from 1960 to 2020

    Returns:
        const_and_regrid_runoff (xr.DataArray): runoff data constrined to a tighter area and regridded to 5km
    """
    const_and_regrid_runoff = regrid_to_5km_grid(
        constrain_to_australia(
            runoff_drought_data
        )
    )

    return const_and_regrid_runoff


def main():
    VARS = [
        # 'runoff',
        'soil_moisture'
    ]
    var_name = {
        'runoff': 'qtot', 
        'soil_moisture': 'sm',
    }

    for var in VARS:
        drought_file = f'/g/data/w97/amu561/Steven_CABLE_runs/drought_metrics_AWRA_ref/3-month/drought_metrics_AWRA_ref_{var_name[var]}_scale_3_1960_2020.nc'
        drought_ds = xr.open_dataset(drought_file)
        drought_metric = drought_ds.timing

        precip_file = '/g/data/w97/mg5624/RF_project/Precipitation/AGCD/AGCD_v1_precip_total_r005_monthly_1900_2021.nc'
        precip = xr.open_dataarray(precip_file)

        drought_metric_processed = process_drought_metric_data(
            fix_drought_metric_time(
                drought_metric, precip
            )
        )

        drought_metric_processed = drought_metric_processed.rename('Drought')

        filepath_out = f'/g/data/w97/mg5624/RF_project/drought_metric/{var}_percentile/'

        if not os.path.exists(filepath_out):
            os.makedirs(filepath_out)

        filename = f'AWRA_{var}_percentile_drought_metric_monthly_1960-2020.nc'

        drought_metric_processed.to_netcdf(filepath_out + filename)


if __name__ == "__main__":
    main()
