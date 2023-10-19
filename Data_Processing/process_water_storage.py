import xarray as xr
import processing_functions
from processing_functions import regrid_to_5km_grid
import numpy as np
import os
import h5py
import pandas as pd
import sys


def convert_time_to_datetime(time):
    """
    Takes in array of time in year format and converts it to datetime with year and month.

    Args:
        time (np.array): time with only the year labelled

    Returns:
        datetime (pd.Series): time in datetime format
    """
    years = [str(int(year)) for year in time]
    month_ints = list(range(1, 13)) * int((len(years) / 12))
    months = [str(month).zfill(2) for month in month_ints]
    day = ['16'] * len(years)

    time_df = pd.DataFrame({'Year': years, 'Month': months, "Day": day})
    datetime = pd.to_datetime(time_df)

    return datetime


def create_dataset_from_mat_data(mat_data, variable_name):
    """
    Converts a .mat file read using h5py.File into an xarray Dataset.

    Args:
        mat_data (h5py.FIle): mat file of data to be converted
        variable_name (str): name of variable which needs converted

    Returns:
        dataset (xr.Dataset): dataset of the data within the .mat file
    """
    arrays = {}
    for k, v in mat_data.items():
        arrays[k] = np.array(v)

    data = {}
    dims = ['Time', 'Latitude', 'Longitude']
    data_var = variable_name

    # Replace the nan values with np.nan so we can remove them later
    data_arr = arrays[data_var]
    non_numbers_mask = np.isnan(data_arr.astype(float))
    np.set_printoptions(threshold=sys.maxsize)
    print(data_arr[non_numbers_mask])
    data_arr[non_numbers_mask] = np.nan
    
    for coord in dims:
        data[coord] = arrays[coord]

    Datetime = convert_time_to_datetime(arrays["Time"][0])

    dataset = xr.Dataset({
        "time": ("time", Datetime),
        "lat": ("lat", arrays["Latitude"][0]),
        "lon": ("lon", arrays["Longitude"][0]),
        data_var: (("time", "lat", "lon"), data_arr)
    })

    return dataset


def convert_mat_to_nc(mat_filepath, save_file=True, filepath_out=None):
    """
    Converts .mat file to .nc file and can save the .nc file in specified location if required.

    Args:
        mat_filepath (str): filepath to the .mat file
        save_file (bool): option to save the .nc file or not
        filepath_out (str): 
            filepath where the nc file is to be saved, default is None, must be specified if save_file is True

    Returns:
        dataset (xr.Dataset): data from mat file in an Xarray Dataset
    """
    mat_data = h5py.File(mat_filepath)
    dataset = create_dataset_from_mat_data(mat_data, 'TWSA')

    if save_file:
        if filepath_out == None:
            raise ValueError(
                'Filepath for saving file unspecified, please specify fielpath when calling this function with save_file=True'
            )
        else:
            dataset.to_netcdf(filepath_out)
    print(dataset)
    return dataset


def constrain_and_regrid_water_storage(global_water_storage_data):
    """
    Constrains water storage to Australia and regrids it to 5km grid. Saves netcdf in my data directory.

    Args:
        global_water_storage_data (xr.DataArray): water storage data over the globe (or any area larger than Australia)

    Returns:
        interpolated_water_storage_aus (xr.DataArray): water storage data across Australia at 5km scale
    """
    # Constrain water_storage data to cover just Australia
    water_storage_aus = processing_functions.constrain_to_australia(global_water_storage_data)

    # regrid water_storage to precip
    interpolated_water_storage_aus = regrid_to_5km_grid(water_storage_aus)

    return interpolated_water_storage_aus


def calculate_CWS_and_save_file(water_storage_data, filepath_out):
    """
    Calculates change in water storage from the water storage data. Saves resulting netcdf to filepath_out.

    Args:
        water_storage_data (xr.DataArray): water storage dataset
        filepath_out (str): filepath where the change in water stroage file is to be saved
    """
    print('before regrid water \n:', water_storage_data)
    aus_water_storage = constrain_and_regrid_water_storage(water_storage_data)
    print('aus water:\n', aus_water_storage)
    CWS = aus_water_storage.diff(dim='time', n=1)
    print('CWS:\n', CWS)
    CWS.to_netcdf(filepath_out)


def main():
    GTWS_filepath = '/g/data/w97/mg5624/RF_project/Water_Storage/GTWS-MLrec/'
    water_storage_mat = GTWS_filepath + 'CSR-based_GTWS-MLrec_TWS.mat'
    water_storage_nc = GTWS_filepath + 'CSR-based_GTWS-MLrec_TWS.nc'
    CWS_filepath = GTWS_filepath + 'CSR-based_GTWS-MLrec_CWS_australia_0.05_grid.nc'
    
    WS = convert_mat_to_nc(water_storage_mat, save_file=True, filepath_out=water_storage_nc)
    calculate_CWS_and_save_file(WS, CWS_filepath)


if __name__ == "__main__":
    main()
    