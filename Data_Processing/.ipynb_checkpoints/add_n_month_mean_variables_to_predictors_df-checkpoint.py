import xarray as xr
import pandas as pd
from create_predictors_datasets import convert_dataset_to_dataframe, constrain_data, add_year_month_coord_to_dataarray
import sys


datadir = '/g/data/w97/mg5624/RF_project/'
ET_path = datadir + 'ET_products/v3_6/'
SM_path = datadir + 'Soil_Moisture/v3_8/'
Runoff_path = datadir + 'Runoff/AWRA/'

VARS = ['Runoff', 
        # 'ET', 
        # 'PET',
        # 'SMsurf', 
        # 'SMroot'
       ]
N_MONTHS = [3, 
            # 6, 
            # 12, 
            24]


def var_filepath(var_name, n_months):
    """
    Defines the filepath to the variable depending on the n-months.

    Args:
        var_name (str): name of variable
        n_months (int): the number of months the variable is meaned over
    """
    filepath = {
        'Runoff': Runoff_path + f'AWRAv7_Runoff_{n_months}_month_mean_1911_2023.nc',
        'ET': ET_path + f'ET/ET_1980-2021_GLEAM_v3.6a_{n_months}_month_mean_MO_Australia_0.05grid.nc',
        'PET': ET_path + f'PET/PET_1980-2021_GLEAM_v3.6a_{n_months}_month_mean_MO_Australia_0.05grid.nc',
        'SMsurf': SM_path + f'SMroot/SMroot_1980-2022_GLEAM_v3.8a_{n_months}_month_mean_MO_Australia_0.05grid.nc',
        'SMroot': SM_path + f'SMsurf/SMsurf_1980-2022_GLEAM_v3.8a_{n_months}_month_mean_MO_Australia_0.05grid.nc',
    }

    return filepath[var_name]


def add_new_var_to_predictors_ds(predictors_ds, var_name, n_months):
    """
    Adds new variable (from the list of n-month meaned variables) to our predictors dataset.

    Args:
        predictors_ds (xr.Dataset): predicotrs dataset
        var_name (str): name of variable
        n_months (int): the number of months the variable is meaned over
    """
    file = var_filepath(var_name, n_months)
    new_var = xr.open_dataset(file)
    new_var_constr = constrain_data(new_var, 1980, 2022, 'SE_australia')

    if 'Year_Month' not in new_var.coords.keys():
            new_var = add_year_month_coord_to_dataarray(new_var)
        
    predictors_ds_with_new_var = xr.merge([predictors_ds, new_var], join='outer')

    return predictors_ds_with_new_var


def create_list_of_vars(vars_list):
    """
    Saves all the input vars as a list of xr.datasets.

    Args:
        vars_list (list of str): list of the names of variables

    Returns:
        variable_datsets (list of xr.Dataset): list of xr.Datasets of the variables named in vars_list
    """
    variable_datasets = []
    for var in vars_list:
        for n in N_MONTHS:
            file = var_filepath(var, n)
            var_ds = xr.open_dataset(file)
            variable_datasets.append(var_ds)
    print(variable_datasets)
    return variable_datasets


def save_predictors_ds_and_df(predictors_ds):
    """
    Saves predictors ds with the new variables in it to our datadir. Also converts the ds to a df and saves that too.

    Args:
        predictors_ds (xr.Dataset): predictors ds with the new vraibles merged into it
    """
    filepath = datadir + 'predictors_data/1980_model/'
    predictors_ds.to_netcdf(filepath + 'predictors_dataset_1980-2022_SE_australia1.nc')
    print('ds saved')
    
    predictors_df = convert_dataset_to_dataframe(predictors_ds)
    print('ds to df complete')
    predictors_df.to_csv(filepath + 'predictors_dataframe_1980-2022_SE_australia1.csv')
    print('df saved: ', predictors_df.columns)


# def save_ds_as_df(ds):
#     filepath = datadir + 'predictors_data/1980_model/'
#     predictors_df = convert_dataset_to_dataframe(ds)
#     print('ds to df complete')
#     print(predictors_df)
#     predictors_df.to_csv(filepath + 'predictors_dataframe_1980-2022_SE_australia1.csv')
#     print('df saved')
    
        
def main():
    predictors_ds = xr.open_dataset(
        datadir + 'predictors_data/1980_model/predictors_dataset_1980-2022_SE_australia.nc'
    )
    var_datasets = create_list_of_vars(VARS)
    
    var_datasets.append(predictors_ds)
    print('ds list made')
    new_predictors_ds = xr.merge(var_datasets)
    print('ds merged')
    save_predictors_ds_and_df(predictors_ds)
    print('job done :)')
    
    
    # filepath = datadir + 'predictors_data/1980_model/'
    # ds = xr.open_dataset(filepath + 'predictors_dataset_1980-2022_SE_australia1.nc')
    # print(ds)
    # save_ds_as_df(ds)

if __name__ == "__main__":
    main()


