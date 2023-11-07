import xarray as xr
import os
import pandas as pd
from create_predictors_datasets import constrain_data, add_year_month_coord_to_dataarray

datadir = '/g/data/w97/mg5624/RF_project/'

def load_and_constrain_precip_drought(start_year, end_year):
    """
    Loads the dataarray for the rpecip based drought event data and consstrains to time defined and SE australia.

    Args:
        start_year (int): start year to constrain data to
        end_year (int): end year to constrain data to
    """
    path = f'{datadir}/drought_metric/precip_percentile/AGCD_precip_percentile_drought_metric_monthly_1900-2021.nc'
    precip_drought = xr.open_dataarray(path)

    const_precip_drought = constrain_data(precip_drought, start_year, end_year, 'SE_australia')

    return const_precip_drought


def add_precip_drought_to_predictors_ds(predictors_ds, precip_drought):
    """
    Adds new variable (from the list of n-month meaned variables) to our predictors dataset.

    Args:
        predictors_ds (xr.Dataset): predicotrs dataset
        precip_drought (xr.DataArray): precip based binary drought event data (constrained to required time and area)

    Returns:
        predictors_precip_drought_ds (xr.Dataset): dataset with predictors data plus precip based drought events data
    """
    predictors_precip_drought_ds = xr.merge([predictors_ds, precip_drought], join='left')

    return predictors_precip_drought_ds


def save_augmented_predictors_ds_as_test_training_ds(predictor_and_precip_drought_ds):
    """
    Saves the augmented predictors ds as a test training ds.

    Args:
        predictors_precip_drought_ds (xr.Dataset): dataset with predictors data plus precip based drought events data
    """
    filepath = datadir + '/training_data/'
    filename = 'test_training_data_precip_def.nc'
    predictor_and_precip_drought_ds.to_netcdf(filepath + filename)


def main():
    predictors_ds_path = datadir + 'predictors_data/1980_model/new_predictors_dataset_1980-2022_SE_australia.nc'
    predictors_ds = xr.open_dataset(predictors_ds_path)
    save_augmented_predictors_ds_as_test_training_ds(
        add_precip_drought_to_predictors_ds(
            predictors_ds, 
            load_and_constrain_precip_drought(
                1980, 2022
            ),
        ),
    )


if __name__ == "__main__":
    main()
    