import pandas as pd
import xarray as xr
import numpy as np
import os
from pathlib import Path
import processing_functions


precip_filepath = processing_functions.my_data_dir + 'RF_project/Precipitation/AGCD/'
ET_filepath = processing_functions.my_data_dir + 'RF_project/ET_products/v3_6/'
SM_filepath = processing_functions.my_data_dir + 'RF_project/Soil_Moisture/v3_8/'


MODEL_TYPES = [
    'full', 
    'long_ts',
    # 'test'
]

TIME_PERIODS = {
    'test': [1981, 1983],
    'full': [1980, 2022], 
    'long_ts': [1911, 2022]
}


AREAS = [
    'SE_australia',
    # 'test'
]


COORDS = {
    'SE_australia': {
        'lats': (-40, -29),
        'lons': (139, 155)
    },
    'test': {
        'lats': (-32, -28),
        'lons': (149, 152)
    },
}


VARS = {
    1981: ['Precipitation', 'Runoff', 'ENSO_index', 'PET'],
    1980: ['Precipitation', 'Acc_3-Month_Precipitation', 'Acc_6-Month_Precipitation', 'Acc_12-Month_Precipitation', 
           'Acc_24-Month_Precipitation', 'Runoff', 'ENSO_index', 'IOD_index', 'SAM_index', 'ET', 'PET', 'SMsurf', 'SMroot'],
    
    1911: ['Precipitation', 'Acc_3-Month_Precipitation', 'Acc_6-Month_Precipitation', 'Acc_12-Month_Precipitation', 
           'Acc_24-Month_Precipitation', 'Runoff', 'ENSO_index', 'IOD_index'],
}


FILES = {
    'Precipitation': precip_filepath + 'AGCD_v1_precip_total_r005_monthly_1900_2021.nc',
    'Acc_3-Month_Precipitation': precip_filepath + 'AGCD_v1_precip_total_r005_3monthly_1900_2021.nc',
    'Acc_6-Month_Precipitation': precip_filepath + 'AGCD_v1_precip_total_r005_6monthly_1900_2021.nc',
    'Acc_12-Month_Precipitation': precip_filepath + 'AGCD_v1_precip_total_r005_12monthly_1900_2021.nc',
    'Acc_24-Month_Precipitation': precip_filepath + 'AGCD_v1_precip_total_r005_24monthly_1900_2021.nc',
    'Runoff': processing_functions.my_data_dir + 'RF_project/Runoff/AWRA/AWRAv7_Runoff_month_1911_2023.nc', 
    'ENSO_index': processing_functions.my_data_dir + 'RF_project/ENSO/ENSO_BEST_index_sorted.csv',
    'IOD_index': processing_functions.my_data_dir + 'RF_project/IOD/IOD_DMI_index_sorted.csv',
    'SAM_index': processing_functions.my_data_dir + 'RF_project/SAM/SAM_AAO_index_sorted.csv',
    'ET': ET_filepath + 'ET/ET_1980-2021_GLEAM_v3.6a_MO_Australia_0.05grid.nc',
    'PET': ET_filepath + 'PET/PET_1980-2021_GLEAM_v3.6a_MO_Australia_0.05grid.nc',
    'SMsurf': SM_filepath + 'SMroot/SMroot_1980-2022_GLEAM_v3.8a_MO_Australia_0.05grid.nc',
    'SMroot': SM_filepath + 'SMsurf/SMsurf_1980-2022_GLEAM_v3.8a_MO_Australia_0.05grid.nc',
}


def add_year_month_coord_to_dataarray(dataarray):
    """
    Adds year, month, and year_month coordinates to the input dataarray. 
    Requires dataarray to have time cooridnate in datetime format.

    Args:
        dataarray (xr.DataArray): dataarray to add coordinates to

    Returns:
        dataarray (xr.DataArray): 
            dataarray with additonal year, month, and year_month coordinates
    """
    # Add year_month coordinates to data
    dataarray['Year'] = dataarray['time'].dt.strftime('%Y')
    dataarray['Month'] = dataarray['time'].dt.strftime('%m')
    dataarray['Year_Month'] = dataarray['Year'] + '-' + dataarray['Month']

    return dataarray


def constrain_data(data, start_year, end_year, area):
    """
    Constrains the data to the specified area.

    Args:
        data (xr.DataArray): the data to be constrained
        start_year (int): start year
        end_year (int): end year
        area (str): name of the area for data to be constrained to

    Returns:
        constrained_data (xr.DataArray): data constrained to are and time bounds
    """
    lats = COORDS[area]['lats']
    lons = COORDS[area]['lons']

    lat_min, lat_max = lats[0], lats[1]
    lon_min, lon_max = lons[0], lons[1]

    start_year = str(start_year)
    end_year = str(end_year)

    if 'lat' and 'lon' in data.coords.keys():
        constrained_data = data.sel(
            time=slice(start_year, end_year),
            lat=slice(lat_min, lat_max),
            lon=slice(lon_min, lon_max)
        )
    else:
        constrained_data = data.sel(
            time=slice(start_year, end_year)
        )

    return constrained_data


def read_csv_as_dataset(csv_file):
    """
    Opens a csv file as a xr.Dataset.

    Args:
        csv_file (str): filepath to csv file

    Returns:
        ds (xr.Dataset): csv data in a dataset
    """
    df = pd.read_csv(csv_file)
    df.dropna(axis=0)
    
    if 'Unnamed: 0' in df.columns:
        df.drop('Unnamed: 0', axis=1, inplace=True)
        
    df['time'] = df['Year_Month'] + '-01'
    df['time'] = pd.to_datetime(df['time'])

    df.drop('Year_Month', axis=1, inplace=True)
    df.sort_values(by='time', inplace=True)
    df = df.set_index(['time'])
    ds = df.to_xarray()
    return ds


def create_list_of_predictors_ds(start_year, end_year, area):
    """
    Puts all the predictors' datasets into a list. 

    Args:
        start_year (int): start year
        end_year (int): end year
        area (str): one of the areas defined in the COORDS dictionary

    Returns:
        predictors (list): list of datasets of the predictors
    """
    predictors = []
    for variable in VARS[start_year]:
        
        file = FILES[variable]
        if file[-3:] == 'csv':
            predictor_ds = read_csv_as_dataset(file)
        elif file[-2:] == 'nc':
            predictor_ds = xr.open_dataset(file)
        else:
            raise ValueError(f'File type of {new_predictor_name} not supported. '
                             f'Expected .nc or .csv file got file: {file} instead')
        
        predictor_ds = constrain_data(predictor_ds, start_year, end_year, area)

        if 'Year_Month' not in predictor_ds.coords.keys():
            predictor_ds = add_year_month_coord_to_dataarray(predictor_ds)
            
        predictors.append(predictor_ds)

    return predictors


def add_cyclical_months_to_dataset(predictors_ds):
    """
    Adds variables of sin(month) and cos(month) to the predictors dataset.

    Args:
        predictors_ds (xr.Dataset): dataset with data for RF predictors without cyclical months
    
    Returns:
        predictors_ds (xr.Dataset): Ddataset with data for RF predictors including the cyclical months
    """
    predictors_ds = predictors_ds.assign(
        Sin_month=np.sin(predictors_ds['Month'].astype(int)),
        Cos_month=np.cos(predictors_ds['Month'].astype(int))
    )
    return predictors_ds
    

def merge_datasets(predictors):
    """
    Merges the predictors dataset into one dataset.

    Args:
        predictors (list): list of the predictors datasets to be merged together

    Returns:
        predictors_ds (xr.Dataset): Dataset with all the predictors data in it
    """
    predictors_ds = xr.merge(predictors, join='outer')
    predictors_ds = add_cyclical_months_to_dataset(predictors_ds)
    return predictors_ds


def convert_dataset_to_dataframe(predictors_ds):
    """
    Converts predictors dataset into a pandas dataframe.

    Args:
        predictors_ds (xr.Dataset): data of all predictors for RF model 

    Returns:
        predictors_df (pd.DataFrame): dataframe containing all the predictors data for the RF model
    """
    predictors_df = predictors_ds.to_dataframe()
    predictors_df.reset_index(inplace=True)

    coord_rename = {
        'lon': 'Longitude',
        'lat': 'Latitude',
    }
    
    predictors_df.rename(columns=coord_rename, inplace=True)

    predictors_df['Latitude'] = predictors_df['Latitude'].astype(float).round(2)
    predictors_df['Longitude'] = predictors_df['Longitude'].astype(float).round(2)

    # column_order = ['K', 'L', 'M'] + [col for col in df.columns if col not in ['K', 'L', 'M']]
    return predictors_df
    

def main():
    for area in AREAS:
        print(area)
        for model in MODEL_TYPES:
            time_period = TIME_PERIODS[model]
            start_year = time_period[0]
            end_year = time_period[-1]

            predictors_ds = merge_datasets(
                create_list_of_predictors_ds(start_year, end_year, area)
            )        
            print(predictors_ds['time'])
            predictors_df = predictors_ds.to_dataframe()
            # Save predictors dataframe
            filepath = processing_functions.my_data_dir + f'/RF_project/predictors_data/{model}_model/'
            if not os.path.exists(filepath):
                os.makedirs(filepath)
                
            filename_nc = f'predictors_dataset_{start_year}-{end_year}_{area}.nc'
            filename_csv = f'predictors_dataframe_{start_year}-{end_year}_{area}.csv'
            
            predictors_ds.to_netcdf(filepath + filename_nc)
            predictors_df.to_csv(filepath + filename_csv)

# def main2():
#     predictors_ds = xr.open_dataset(processing_functions.my_data_dir + '/RF_project/predictors_data/predictors_dataset_1911-2022_SE_australia.nc')
#     predictors_df = predictors_ds.to_dataframe()
#     predictors_df.to_csv(processing_functions.my_data_dir + '/RF_project/predictors_data/predictors_dataframe_1911-2022_SE_australia.csv')

if __name__ == "__main__":
    main()
