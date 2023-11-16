import xarray as xr
import os
import pandas as pd
from create_predictors_datasets import constrain_data, convert_dataset_to_dataframe

datadir = '/g/data/w97/mg5624/RF_project/'

VARS = [
    'precip', 
    # 'runoff', 
    # 'soil_moisture'
]

metric_filepath = {
    'precip': f'{datadir}/drought_metric/precip_percentile/AGCD_precip_percentile_drought_metric_monthly_1900-2021.nc',
    'runoff': f'{datadir}/drought_metric/runoff_percentile/AWRA_runoff_percentile_drought_metric_monthly_1960-2020.nc',
    'soil_moisture': f'{datadir}/drought_metric/soil_moisture_percentile/AWRA_soil_moisture_percentile_drought_metric_monthly_1960-2020.nc',
}


def load_and_constrain_drought_metric(metric_var, start_year, end_year):
    """
    Loads the dataarray for the rpecip based drought event data and consstrains to time defined and SE australia.

    Args:
        metric_var (str): the variable the drought metric is based on (e.g. precip, runoff, etc.)
        start_year (int): start year to constrain data to
        end_year (int): end year to constrain data to
    """
    path = metric_filepath[metric_var]
    drought_metric = xr.open_dataarray(path)

    const_drought_metric = constrain_data(drought_metric, start_year, end_year, 'SE_australia')

    return const_drought_metric


def add_drought_metric_to_predictors_ds(predictors_ds, drought_metric):
    """
    Adds new variable (from the list of n-month meaned variables) to our predictors dataset.

    Args:
        predictors_ds (xr.Dataset): predicotrs dataset
        drought_metric (xr.DataArray): precip based binary drought event data (constrained to required time and area)

    Returns:
        predictors_drought_metric_ds (xr.Dataset): dataset with predictors data plus precip based drought events data
    """
    predictors_drought_metric_ds = xr.merge([predictors_ds, drought_metric], join='left')

    return predictors_drought_metric_ds


def save_augmented_predictors_ds_and_df_as_test_training_ds_and_df(metric_var, predictor_and_drought_metric_ds):
    """
    Saves the augmented predictors ds as a test training ds.

    Args:
        predictors_drought_metric_ds (xr.Dataset): dataset with predictors data plus precip based drought events data
    """
    filepath = datadir + f'/training_data/test_training/{metric_var}'
    filename_nc = f'test_training_data_{metric_var}_def.nc'
    predictor_and_drought_metric_ds.to_netcdf(filepath + filename_nc)

    predictor_and_drought_metric_df = convert_dataset_to_dataframe(predictor_and_drought_metric_ds)
    filename_csv = f'test_training_data_{metric_var}_def.csv'
    predictor_and_drought_metric_df.to_csv(filepath + filename_csv)


def main():
    predictors_ds_path = datadir + 'predictors_data/1980_model/new_predictors_dataset_1980-2022_SE_australia.nc'
    predictors_ds = xr.open_dataset(predictors_ds_path)
    
    for var in VARS:
        save_augmented_predictors_ds_and_df_as_test_training_ds_and_df(var,
            add_drought_metric_to_predictors_ds(
                predictors_ds, 
                load_and_constrain_drought_metric(
                    var, 1980, 2022
                ),
            ),
        )

    # predictors_test_ds = xr.open_dataset('/g/data/w97/mg5624/RF_project/training_data/test_training/test_training_data_precip_def.nc')
    # predictor_and_drought_metric_df = convert_dataset_to_dataframe(predictors_test_ds)
    # filepath = datadir + '/training_data/test_training/'
    # filename_csv = 'test_training_data_precip_def.csv'
    # predictor_and_drought_metric_df.to_csv(filepath + filename_csv)


if __name__ == "__main__":
    main()
    