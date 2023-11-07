import xarray as xr
import create_predictors_datasets
import processing_functions


VARS = ['ET', 'PET', 'SMsurf', 'SMroot', 'Runoff']
N_MONTHS = [3, 6, 12, 24]
FILES = {
    'Runoff': {
        'filepath': processing_functions.my_data_dir + 'RF_project/Runoff/AWRA/',
        'file_in': 'AWRAv7_Runoff_month_1911_2023.nc',
    },
    'ET': {
        'filepath': create_predictors_datasets.ET_filepath + 'ET/',
        'file_in': 'ET_1980-2021_GLEAM_v3.6a_MO_Australia_0.05grid.nc',
    },
    'PET': {
        'filepath': create_predictors_datasets.ET_filepath + 'PET/',
        'file_in': 'PET_1980-2021_GLEAM_v3.6a_MO_Australia_0.05grid.nc',
    },
    'SMsurf': {
        'filepath': create_predictors_datasets.SM_filepath + 'SMroot/',
        'file_in': 'SMroot_1980-2022_GLEAM_v3.8a_MO_Australia_0.05grid.nc',
    },
    'SMroot': {
        'filepath': create_predictors_datasets.SM_filepath + 'SMsurf/',
        'file_in': 'SMsurf_1980-2022_GLEAM_v3.8a_MO_Australia_0.05grid.nc',
    }
}   


def create_n_month_mean(var_name, n_months):
    """
    Creates n-month mean data of input monthly data.

    Args:
        data (xr.DataArray): monthly data to take find n-month means of
        n_months (int): number of months to aggreagte over
    """
    filepath = FILES[var_name]['filepath']
    filename_in = FILES[var_name]['file_in']
    data = xr.open_dataarray(filepath + filename_in)
    n_month_mean = data.rolling(time=n_months).mean()

    FILES_OUT = {
        'Runoff': f'AWRAv7_Runoff_{n_months}_month_mean_1911_2023.nc',
        'ET': f'ET_1980-2021_GLEAM_v3.6a_{n_months}_month_mean_MO_Australia_0.05grid.nc',
        'PET': f'PET_1980-2021_GLEAM_v3.6a_{n_months}_month_mean_MO_Australia_0.05grid.nc',
        'SMsurf': f'SMroot_1980-2022_GLEAM_v3.8a_{n_months}_month_mean_MO_Australia_0.05grid.nc',
        'SMroot': f'SMsurf_1980-2022_GLEAM_v3.8a_{n_months}_month_mean_MO_Australia_0.05grid.nc',
    }

    new_name = f'Mean_{n_months}-Month_{var_name}'
    n_month_mean = n_month_mean.rename(new_name)
    
    filename_out = FILES_OUT[var_name]

    n_month_mean.to_netcdf(filepath + filename_out)


def main():
    for var in VARS:
        for n in N_MONTHS:
            create_n_month_mean(var, n)


if __name__ == "__main__":
    main()
    