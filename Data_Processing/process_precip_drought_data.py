import xarray as xr
import os


def fix_time_coord(precip_drought_data, data_with_correct_time):
    """
    Changes the time coord of precip_drought_data to that of data_with_correct_time.

    Args:
        precip_drought_data (xr.DataArray): data array of drought/no drought defined by precipitation
        data_with_correct_time (xr.DataArray): data array with monthly time coord from 1900 to 2021

    Returns:
        precip_drought_data (xr.DataArray): data array with monthly time coord from 1900 to 2021
    """
    precip_drought_data['time'] = data_with_correct_time['time']

    return precip_drought_data


def main():
    precip_drought_path = '/g/data/w97/amu561/Steven_CABLE_runs/drought_metrics_AGCD/3-month/drought_metrics_AGCD_precip_1900_2021_baseline_1970_2005_scale_3.nc'
    precip_drought_ds = xr.open_dataset(precip_drought_path)
    precip_drought = precip_drought_ds.timing
    print('Loaded drought metric')

    precip = xr.open_dataarray('/g/data/w97/mg5624/RF_project/Precipitation/AGCD/AGCD_v1_precip_total_r005_monthly_1900_2021.nc')
    precip_drought_correct_time = fix_time_coord(precip_drought, precip)
    print('corrected time coord of drought data')

    renamed_precip_drought = precip_drought_correct_time.rename('Drought')
    print('renamed drought data')
    print(renamed_precip_drought)
    filepath_out = '/g/data/w97/mg5624/RF_project/drought_metric/precip_percentile/'

    if not os.path.exists(filepath_out):
        os.makedirs(filepath_out)

    filename = 'AGCD_precip_percentile_drought_metric_monthly_1900-2021.nc'

    renamed_precip_drought.to_netcdf(filepath_out + filename)
    print('done - woo!')

if __name__ == "__main__":
    main()
