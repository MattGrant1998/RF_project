import xarray as xr
import pandas as pd
from create_predictors_datasets import convert_dataset_to_dataframe
import sys


datdir = '/g/data/w97/mg5624/RF_project/'
ET_path = datadir + 'ET_products/v3_6/'
SM_path = datadir + 'Soil_Moisture/v3_8/'
Runoff_path = datadir + 'Runoff/AWRA/'

predictors_ds = xr.open_dataset('/g/data/w97/mg5624/RF_project/predictors_data/1980_model/predictors_dataset_1980-2022_SE_australia.nc')

# def filepath(var_name, n_month):

VARS = ['Runoff', 'ET', 'PET', 'SMsurf', 'SMroot']
N_MONTHS = [3, 6, 12, 24]

FILES = {
    'Runoff': Runoff_path + f'AWRAv7_Runoff_{n_months}_month_mean_1911_2023.nc',
    'ET': ET_path + f'ET/ET_1980-2021_GLEAM_v3.6a_{n_months}_month_mean_MO_Australia_0.05grid.nc',
    'PET': ET_path + f'PET/PET_1980-2021_GLEAM_v3.6a_{n_months}_month_mean_MO_Australia_0.05grid.nc',
    'SMsurf': SM_path + f'SMroot/SMroot_1980-2022_GLEAM_v3.8a_{n_months}_month_mean_MO_Australia_0.05grid.nc',
    'SMroot': SM_path + f'SMsurf/SMsurf_1980-2022_GLEAM_v3.8a_{n_months}_month_mean_MO_Australia_0.05grid.nc',
}

    # return filepath[var_name]
# new_var = 

def add_new_var_to_predictors_ds(predictors_ds, var_name, n_months):
    """
    Adds new variable (from the list of n-month meaned variables) to our predictors dataset.

    Args:
        predictors_ds (xr.Dataset): predicotrs dataset
        var_name (str): name of variable
        n_months (int): the number of months the variables is meaned over
    """
    new_var = xr.open_dataset(FILES[var_name])
    predictors_ds_with_new_var = xr.merge([predictors_ds, new_var], join='outer')
    print(predictors_ds_with_new_var)

    return predictors_ds_with_new_var

merged_ds = add_new_var_to_predictors_ds(predictors_ds, 'Runoff', 3)
merged_df = convert_dataset_to_dataframe(merged_ds)

print(merged_df)


