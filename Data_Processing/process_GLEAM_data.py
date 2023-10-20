import processing_functions
import xarray as xr
import os
import itertools


GLEAM_versions = [
    'v3_8',
    'v3_6',
]

final_year = {
    'v3_8': '2022',
    'v3_6': '2021',
}

ET_vars = [
    'E',
    'Et',
    'Ep',  
]

ET_renaming = {
    'E': 'E',
    'Et': 'T',
    'Ep': 'PET',
}

SM_vars = ['SMroot', 'SMsurf']


def define_file_path(GLEAM_version):
    """
    Defines the GLEAM file path based on the GLEAM version required.
    Args:
    GLEAM_version (str): the GLEAM version required (e.g. v3_8)
    """
    if GLEAM_version == 'v3_6':
        GLEAM_data_path = '/g/data/ua8/GLEAM_v3-5/v3-6a/monthly/'
    else:
        GLEAM_data_path = processing_functions.shared_data_dir + f'/Global_ET_products/GLEAM_{GLEAM_version}/{GLEAM_version}a/monthly/'

    return GLEAM_data_path


def constrain_and_regrid_dataset(dataarray):
    """
    Takes in a filepath to a netcdf file. Then constrains nc file to Australia and regrids to 0.05 degree grid
    Args:
    dataset (xr.DataArray): xarray DataArray of spatial data covering Australia (and more)
    """
    dataarray_aus = processing_functions.constrain_to_australia(dataarray)
    dataarray_5km = processing_functions.regrid_to_5km_grid(dataarray_aus)

    return dataarray_5km


def process_ET_products(var, GLEAM_version):
    """
    Constrains E, T, and PET products to Australia and regrids to 0.05 degree grid.
    Args:
    var (str): shorthand name of variable as in original GLEAM file ('E', 'Et', or 'Ep')
    GLEAM_version (str): the GLEAM version required (e.g. v3_8)
    """
    GLEAM_data_path = define_file_path(GLEAM_version)
    v_number = GLEAM_version[-1]
    dataarray = xr.open_dataarray(GLEAM_data_path + f'{var}_1980-{final_year[GLEAM_version]}_GLEAM_v3.{v_number}a_MO.nc')

    new_var_name = ET_renaming[var]
    dataarray.name = new_var_name
    processed_dataarray = constrain_and_regrid_dataset(dataarray)
    year_month_dt_dataarray = processing_functions.set_time_coord_to_year_month_datetime(processed_dataarray)
    filepath = processing_functions.my_data_dir + f'RF_project/ET_products/{GLEAM_version}/{new_var_name}/'
    
    if not os.path.exists(filepath):
        os.makedirs(filepath)
        
    filename = f'{new_var_name}_1980-{final_year[GLEAM_version]}_GLEAM_v3.{v_number}a_MO_Australia_0.05grid.nc'
    processed_dataarray.to_netcdf(filepath + filename)


def process_SM_products(var, GLEAM_version):
    """
    Constrains E, T, and PET products to Australia and regrids to 0.05 degree grid.
    Args:
    var (str): shorthand name of variable as in original GLEAM file ('SMsurf', or 'SMroot')
    GLEAM_version (str): the GLEAM version required (e.g. v3_8)
    """
    GLEAM_data_path = define_file_path(GLEAM_version)
    v_number = GLEAM_version[-1]
    dataarray = xr.open_dataarray(GLEAM_data_path + f'{var}_1980-{final_year[GLEAM_version]}_GLEAM_v3.{v_number}a_MO.nc')
    processed_dataarray = constrain_and_regrid_dataset(dataarray)
    year_month_dt_dataarray = processing_functions.set_time_coord_to_year_month_datetime(processed_dataarray)
    filepath = processing_functions.my_data_dir + f'RF_project/Soil_Moisture/{GLEAM_version}/{var}/'

    if not os.path.exists(filepath):
        os.makedirs(filepath)

    filename = f'{var}_1980-{final_year[GLEAM_version]}_GLEAM_v3.{v_number}a_MO_Australia_0.05grid.nc'
    processed_dataarray.to_netcdf(filepath + filename)


def create_ET_data(GLEAM_version):
    """
    Calculates evapotranspiration from evaporation and transpiration.
    Args:
    GLEAM_version (str): the GLEAM version required (e.g. v3_8)
    """
    data_path = processing_functions.my_data_dir + f'RF_project/ET_products/{GLEAM_version}/'
    E = xr.open_dataarray(data_path + 'E/E_1980-2021_GLEAM_v3.6a_MO_Australia_0.05grid.nc')
    T = xr.open_dataarray(data_path + 'T/T_1980-2021_GLEAM_v3.6a_MO_Australia_0.05grid.nc')
    ET = E + T
    ET.rename('ET')
    ET_filepath = data_path + '/ET/'
    if not os.path.exists(ET_filepath):
        os.makedirs(ET_filepath)

    filename = 'ET_1980-2021_GLEAM_v3.6a_MO_Australia_0.05grid.nc'
    ET.to_netcdf(ET_filepath + filename)


def main():
    # combinations = itertools.product(
    #     ET_vars,
    #     GLEAM_versions
    # )
    # for var, v in combinations:
    #     process_ET_products(var, v)

    combinations = itertools.product(
        SM_vars,
        GLEAM_versions
    )
    for var, v in combinations:
        process_SM_products(var, v)

    # for v in GLEAM_versions:
    #     create_ET_data(v)

if __name__ == "__main__":
    main()

# def save_required_GLEAM_datsets_over_Australia_on_5km_grid():
#     for var in variables:
#         if var == 'ET':
#             # Load evaporation and transpiration files
#             evap_ds = xr.open_dataset(GLEAM_data_path + '/E_1980-2022_GLEAM_v3.8a_MO.nc')
#             evap = evap_ds.E
            
#             transp_ds = xr.open_dataset(GLEAM_data_path + 'Et_1980-2022_GLEAM_v3.8a_MO.nc')
#             transp = transp_ds.Et
            
#             # Calculate Evapotranspiration
#             dataset = evap + transp
        
#         elif var == 'PET':
#             dataset = xr.open_dataset(GLEAM_data_path + 'Ep_1980-2022_GLEAM_v3.8a_MO.nc')
#         else:
#             dataset = xr.open_dataset(GLEAM_data_path + f'{var}_1980-2022_GLEAM_v3.8a_MO.nc')
            
#         dataset_5km = constrain_and_regrid_dataset(dataset)
        
#         if var == 'ET' or 'PET':
#             filepath_save = processing_functions.my_data_dir + f'RF_project/ET_products/{var}/'
#         else:
#             filepath_save = processing_functions.my_data_dir + f'RF_project/SM_products/{var}/'
    
        
#         filename = f'{var}_1980-2022_GLEAM_v3.8a_MO_Australia_0.05grid.nc'
#         if not os.path.exists(filepath_save):
#             os.makedirs(filepath_save)
    
#         dataset_5km.to_netcdf(filepath_save + filename)

# def rename_data_vars(dataset, new_data_var_name):
#     for name in dataset.data_vars:
#         old_name = name

#     dataset.rename({str(old_name):new_data_var_name})

#     return dataset
