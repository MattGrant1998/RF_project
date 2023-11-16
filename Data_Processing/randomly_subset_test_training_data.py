import xarray as xr
import pandas as pd 
import numpy as np
import os
import sys
import multiprocessing


n_iterations = int(sys.argv[1])

def create_df_for_n_randomly_selected_drought_or_no_drought_points(test_training_ds, drought_bin, start_year=1980, n_random_points=500):
    """
    Creates a dataframe of the variables in test_training_ds, but for n randomly selected points of drought or no drought.

    Args:
        test_training_ds (xr.Dataset): dataset of test training with percentile based drought definition
        drought_bin (0 or 1): 0 when selecting points of no drought, 1 when selecting drought points
        start_year (int): earliest year we want the data to go to (default=1980)
        n_random_points (int): number of random points to be selected from dataset (deafult=500)
    
    Returns:
        random_points_df (pd.DataFrame): dataframe with n randomly selected points of either dorught or no drought
    """
    test_training_ds = test_training_ds.sel(time=slice(str(start_year), None))
    test_training_flattened = test_training_ds.stack(all_dims=('time', 'lat', 'lon'))
    test_training_flattened = test_training_flattened.dropna(dim='all_dims')

    drought_da = test_training_flattened.Drought

    mask = drought_da == drought_bin

    drought_masked = drought_da.where(mask)
    non_nan_drought = drought_masked.dropna(dim='all_dims')

    selected_indices = np.random.choice(non_nan_drought.size, n_random_points, replace=False)
    selected_drought_values = non_nan_drought.isel(all_dims=selected_indices)

    time_values = selected_drought_values['time'].values
    lat_values = selected_drought_values['lat'].values
    lon_values = selected_drought_values['lon'].values

    all_dims_values = list(zip(time_values, lat_values, lon_values))

    selected_values = test_training_flattened.sel(all_dims=all_dims_values)

    random_points_df = selected_values.to_dataframe()
    random_points_df.drop(['lon','lat','time'], axis=1, inplace=True)
    random_points_df.reset_index(inplace=True)

    return random_points_df


def create_randomly_selected_df_of_drought_and_no_drought(test_training_ds, start_year=1980, n_random_points=500):
    """
    Finds dfs for n random points of both drought and no drought, and merges them together.

    Args:
        test_training_ds (xr.Dataset): dataset of test training with percentile based drought definition
        start_year (int): earliest year we want the data to go to (default=1980)
        n_random_points (int): number of random points to be selected from dataset (deafult=500)

    Returns:
        merged_df (pd.DataFrame): training dataframe of n drought and no drought points
    """
    no_drought_df = create_df_for_n_randomly_selected_drought_or_no_drought_points(test_training_ds, 0, start_year, n_random_points)
    drought_df = create_df_for_n_randomly_selected_drought_or_no_drought_points(test_training_ds, 1, start_year, n_random_points)
    merged_df = no_drought_df.merge(drought_df, how='outer')
    merged_df.sort_values(by='time', inplace=True)

    return merged_df


# test_training_ds = xr.open_dataset(
#     f'/g/data/w97/mg5624/RF_project/training_data/test_training/precip/test_training_data_precip_def.nc'
# )
# df = create_randomly_selected_df_of_drought_and_no_drought(test_training_ds, n_random_points=3)
# print(df)


def save_randomly_selected_df(drought_no_drought_df, n, variable, start_year=1980):
    """
    Saves the dataframe of randomly selected drought and no drought points to datadir.

    Args:
        drought_no_drought_df (pd.DataFrame): randomly selected drought and no drought training df
        n (int): relating to the configuration of random points
        variable (str): name of the variable that the drought definition is based on
        start_year (int): earliest year we want the data to go to (default=1980)
    """
    number_points = len(drought_no_drought_df)
    filepath = f'/g/data/w97/mg5624/RF_project/training_data/test_training/{variable}/{start_year}+_random_subsets/'
    filename = f'test_training_data_{variable}_{number_points}_points_{start_year}+_random_selection{n}.csv'

    if not os.path.exists(filepath):
        os.makedirs(filepath)

    drought_no_drought_df.to_csv(filepath + filename)


def main():
    VARS = [
        'precip',
        'runoff',
        'soil_moisture',
    ]

    YEARS = [
        1980,
        2014,
    ]
    
    for var in VARS:
        for year in YEARS:
            test_training_ds = xr.open_dataset(
                f'/g/data/w97/mg5624/RF_project/training_data/test_training/{var}/test_training_data_{var}_def.nc'
            )

            for n in range(n_iterations):
                save_randomly_selected_df(
                    create_randomly_selected_df_of_drought_and_no_drought(
                        test_training_ds, start_year=year,
                    ),
                    n, var, start_year=year
                )


# # Try running for parrellelisation of code
# def process_iteration(var, n):
#     test_training_ds = xr.open_dataset(
#         f'/g/data/w97/mg5624/RF_project/training_data/test_training/{var}/test_training_data_{var}_def.nc'
#     )

#     save_randomly_selected_df(
#         create_randomly_selected_df_of_drought_and_no_drought(test_training_ds),
#         n+1
#     )


# def main():
#     VARS = ['precip']

#     # Create a multiprocessing pool
#     pool = multiprocessing.Pool(processes=multiprocessing.cpu_count())

#     for var in VARS:
#         # Use a list comprehension to submit jobs to the pool
#         results = [pool.apply_async(process_iteration, (var, n)) for n in range(n_iterations)]

#         # Wait for all jobs to complete
#         for result in results:
#             result.get()

#     # Close the pool
#     pool.close()
#     pool.join()


if __name__ == "__main__":
    main()
