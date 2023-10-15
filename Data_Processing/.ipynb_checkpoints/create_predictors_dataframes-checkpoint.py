import pandas as pd
import xarray as xr
import numpy as np
import os
from pathlib import Path
import processing_functions


precip_filepath = processing_functions.my_data_dir + 'RF_project/Precipitation/AGCD/'
ET_filepath = processing_functions.my_data_dir + 'RF_project/ET_products/v3_6/'
SM_filepath = processing_functions.my_data_dir + 'RF_project/Soil_Moisture/v3_8/'

TIME_PERIODS = [
    [1980, 2022], 
    [1911, 2022]
]


AREAS = [
    'SE_australia',
]


COORDS = {
    'SE_australia': {
        'lats': (-38, -27),
        'lons': (140, 154)
    },
}


VARS = {
    1980: ['Precipitation', 'Acc_3-Month_Precipitation', 'Acc_Annual_Precipitation', 'Runoff', 
           'ENSO_index', 'IOD_index', 'SAM_index', 'ET', 'PET', 'SMsurf', 'SMroot', 'Sin_month', 'Cos_month'],
    
    1911: ['Precipitation', 'Acc_3-Month_Precipitation', 'Acc_Annual_Precipitation', 
           'Runoff', 'ENSO_index', 'IOD_index', 'Sin_month', 'Cos_month'],
}


FILES = {
    'Precipitation': '/g/data/w97/amu561/AGCD_drought_metrics/AGCD_1900_2021/AGCD_v1_precip_total_r005_monthly_1900_2021.nc',
    'Acc_3-Month_Precipitation': precip_filepath + 'AGCD_v1_precip_total_r005_3monthly_1900_2021.nc',
    'Acc_Annual_Precipitation': precip_filepath + 'AGCD_v1_precip_total_r005_annual_1900_2021.nc',
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


def create_predictor_dataframe(start_year, end_year, area, data, data_name):
    """
    Creates a dataframe that runs from start year to end year, to be used to
    predict droughts in that time period. Predictors can then be added to 
    this dataframe using the add_predictor_to_predictors_dataframe function.
    
    Args:
        start_year (int): start year
        end_year (int): end year
        area (str): one of the areas defined in the COORDS dictionary
        data (xr.DataSet or xr.DataArray): 
            data of a predictor containing at least data covering the time period and area of interest
        data_name (str): name of the data
    
    Returns:
        constrained_df (pd.DataFrame): dataframe of data over specified area and time 
    """
    data = data.rename(data_name)
    
    lats = COORDS[area]['lats']
    lons = COORDS[area]['lons']

    lat_min, lat_max = lats[0], lats[1]
    lon_min, lon_max = lons[0], lons[1]

    start_year = str(start_year)
    end_year = str(end_year)
    
    constrained_data = data.sel(
        time=slice(start_year, end_year),
        lat=slice(lat_min, lat_max),
        lon=slice(lon_min, lon_max)
    )

    constrained_data = add_year_month_coord_to_dataarray(constrained_data)

    constrained_df = constrained_data.to_dataframe()
    constrained_df.reset_index(inplace=True)

    coord_rename = {
        'lon': 'Longitude',
        'lat': 'Latitude'
    }
    
    constrained_df.rename(columns=coord_rename, inplace=True)
   
    constrained_df['Latitude'] = constrained_df['Latitude'].astype(float).round(2)
    constrained_df['Longitude'] = constrained_df['Longitude'].astype(float).round(2)
    
    return constrained_df


def create_predictors_df_from_precip_data(start_year, end_year, area, replace=False):
    """
    Creates the initial predictors dataframe, using the precip data as a base.
    Saves the dataframe to datadir (unless replace=False and dataframe already exists)

    Args:
        start_year (int): start year
        end_year (int): end year
        area (str): one of the areas defined in the COORDS dictionary
        replace (bool): if dataframe exists in datadir and replace=True, it will overwrite
        current dataframe. It will do nothing if replace=False and dataframe exists already
    """
    precip_ds = xr.open_dataset('/g/data/w97/amu561/AGCD_drought_metrics/AGCD_1900_2021/AGCD_v1_precip_total_r005_monthly_1900_2021.nc')
    precip = precip_ds.precip
    predictors_df = create_predictor_dataframe(start_year, end_year, area, precip, 'Precipitation')
    
    filepath = processing_functions.my_data_dir + '/RF_project/predictors_data/'
    filename = f'predictors_dataframe_{start_year}-{end_year}_{area}.csv'

    if Path(filepath + filename).is_file():
        pass
    else:
        predictors_df.to_csv(filepath + filename)


def add_predictor_to_predictors_dataframe(predictors_df, new_predictor_name, replace=False):
    """
    Adds specified predictor data into the predictors dataframe.

    Args:
        predictor_df (pd.DataFrame): dataframe of the predictors
        new_predictor_data (xr.DataArray or pd.DataFrame): new predictor data to be added
        new_predictor_name (str): name of the new predictor
        replace (bool): if true, new predictor will replace any of the same name in dataframe

    Returns:
        merged_df (pd.DataFrame): predicotr_df with additional predictor in it            
    """
    # Return the original dataframe if predictor_name exists and we're not replacing it.
    if new_predictor_name in predictors_df and not replace:
        return predictors_df

    file = FILES[new_predictor_name]
    
    if file[-3:] == 'csv':
        new_predictor_df = pd.read_csv(file)
        dataframe['Month'] = dataframe['Month'].astype(int).astype(str)
        dataframe["Year_Month"] = dataframe['Year'].astype(str) + '-' + dataframe['Month'].str.zfill(2)
    elif file[-2:] == 'nc':
        new_predictor_data = xr.open_dataarray(file)
        new_predictor_data = add_year_month_coord_to_dataarray(new_predictor_data)
        new_predictor_df = new_predictor_data.to_dataframe()
        new_predictor_df.reset_index(inplace=True)
    else:
        raise ValueError(f'File type of {new_predictor_name} not supported. Expected .nc or .csv file got file: {file} instead')
        
    merged_df = pd.merge(predictors_df, new_predictor_df, on='Year_Month', how='inner')

    return merged_df


def add_cyclical_month_columns_to_predictors_df(predictors_df):
    """
    Adds two extra columns to training dataframe (sine_month and cosine_month) to proved cyclical months.
    
    Args:
        predictors_df (pd.DataFrame): Dataframe containing the predictors data
    """
    months = np.arange(1, 13)

    angles = 2 * np.pi * month_numbers / 12
    sin_month = np.sin(angles)
    cos_month = np.cos(angles)

    month_data = {'Month': months, 'Sin_month': sin_month, 'Cos_month': cos_month}
    month_df = pd.DataFrame(month_data)
    
    merged_df = pd.merge(predictors_df, month_df, on='Month', how='inner')

    return merged_df
    

def main():
    for area in AREAS:
        for time_period in TIME_PERIODS:
            start_year = time_period[0]
            end_year = time_period[-1]

            # Create initial predictors dataframe with just precip data
            precip_ds = xr.open_dataset('/g/data/w97/amu561/AGCD_drought_metrics/AGCD_1900_2021/AGCD_v1_precip_total_r005_monthly_1900_2021.nc')
            precip = precip_ds.precip
            predictors_df = create_predictor_dataframe(start_year, end_year, area, precip, 'Precipitation')

            # Add all other variables to predictors dataframe
            for var in VARS[start_year]:
                predictors_df = add_predictor_to_predictors_dataframe(predictors_df, var)

            # Save predictors dataframe
            filepath = processing_functions.my_data_dir + '/RF_project/predictors_data/'
            filename = f'predictors_dataframe_{start_year}-{end_year}_{area}.csv'
            
            predictors_df.to_csv(filepath + filename)


if __name__ == "__main__":
    main()
