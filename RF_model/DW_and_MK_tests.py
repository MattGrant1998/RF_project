import numpy as np
import pandas as pd
import xarray as xr
import os
from statsmodels.formula.api import ols
from statsmodels.stats.stattools import durbin_watson
import pymannkendall as mk
from dask_jobqueue import PBSCluster
from dask.distributed import Client
from numba import njit


datadir = '/g/data/w97/mg5624/RF_project/'
plotdir = '/g/data/w97/mg5624/plots/RF_project/results_analysis/'


def sum_drought_events_per_time_period(drought_events_data, number_of_years):
    """
    Finds the number of drought events per specified number of year for each grid cell.

    Args:
        drought_events_data (xr.DataArray): drought event data (monthly timescale)
        number_of_years (int): number of years to sum the number of droughts over

    Returns:
        drought_events_per_year (xr.DataArray): number of events per year
    """
    drought_events_per_time_period = drought_events_data.resample(time=f'{number_of_years}Y').sum(dim='time').isel(time=slice(1, -1))
    # drought_events_per_time_period = drought_events_per_time_period.rename({'year': 'time'})

    return drought_events_per_time_period


def calculate_DW_score(data, DW_upper_bound=2.5, DW_lower_bound=1.5):
    """
    Calculates the DW score for each grid point of the input data.

    Args:
        data (xr.DataArray): spatial and temporal dataarray
        DW_upper_bound (float): upper bound for DW null hypothesis to be satisfied (default=2.5)
        DW_lower_bound (float): lower bound for DW null hypothesis to be satisfied (default=1.5)

    Returns:
        DW_array (pd.DataFrame): dataframe containing DW score and indication if null hypothesis 
                                 satisfied or not (1=+ve corr, 0=no corr, -1=-ve corr) 
    """
    print("DW")
    data_name = data.name
    DW_df = pd.DataFrame()

    # Flatten the data array to simplify operations
    data_df = data.to_dataframe().reset_index()
    data_df['latlon'] = data_df.apply(lambda row: (row['lat'], row['lon']), axis=1)
    
    # Group by latlon and perform the DW calculations
    group_data = data_df.groupby("latlon")
    DW_scores = group_data.apply(lambda group: calculate_DW_group(group, data_name, DW_upper_bound, DW_lower_bound))
    
    DW_df = DW_scores.reset_index()

    DW_df.drop('latlon', axis=1, inplace=True)
    return DW_df


def calculate_DW_group(group, data_name, DW_upper_bound, DW_lower_bound):
    lat, lon = group["lat"].iloc[0], group["lon"].iloc[0]

    # if all values are nan, then skip this grid point
    if group[data_name].isnull().all():
        return pd.Series({"lat": lat, "lon": lon, "DW_score": np.nan, "DW_hyp": np.nan})

    model = ols(f'{data_name} ~ time', data=group).fit()
    DW_score = durbin_watson(model.resid)

    if DW_score < DW_lower_bound:
        DW_hyp = 1
    elif DW_score > DW_upper_bound:
        DW_hyp = -1
    else:
        DW_hyp = 0

    return pd.Series({"lat": lat, "lon": lon, "DW_score": DW_score, "DW_hyp": DW_hyp})


def find_MK_trendtest(data, test_type, year_from, year_to):
    """
    Computes the MK trendtest for each grid point in the data.

    Args:
        data (xr.DataArray): spatial and temporal data
        test_type (str): MK trend test type being performed
        year_from (int): year from whcih the trend goes from
        year_to (int): year to which the trend goes to

    Returns:
        MK_data (pd.DataFrame): trend test result at each grid point
    """
    data = data.sel(time=slice(year_from, year_to))
    data_name = data.name
    MK_df = pd.DataFrame()
    # numpy_arayd = ata.values()

    for i in data['lat'].values:
        for j in data['lon'].values:
            data_ij = data.sel(lat=i, lon=j)
            data_ij_df = data_ij.to_dataframe()
            data_ij_df.reset_index(inplace=True)

            # if all values are nan, then skip this grid point
            if data_ij_df[data_name].isnull().values.all():
                MK_dict = {'lat': i, 'lon': j, 'MK_trend': np.nan, 'MK_slope': np.nan}
                # break
            else:
                if test_type == 'original':
                    MK_result = mk.original_test(data_ij_df[data_name].values)
                elif test_type == 'hamed_rao':
                    MK_result = mk.hamed_rao_modification_test(data_ij_df[data_name].values)
                elif test_type == 'yue_wang':
                    MK_result = mk.yue_wang_modification_test(data_ij_df[data_name].values)
                elif test_type == 'seasonal_sens_slope':
                    MK_result = mk.seasonal_sens_slope(data_ij_df[data_name].values)

                if test_type == 'seasonal_sens_slope':
                    MK_dict = {'lat': i, 'lon': j, 'MK_slope': MK_result.slope}
                else:
                    MK_dict = {'lat': i, 'lon': j, 'MK_trend': MK_result.trend, 'MK_slope': MK_result.slope}
                
            MK_df_ij = pd.DataFrame([MK_dict])
            MK_df = pd.concat((MK_df, MK_df_ij))

    return MK_df


# import dask
# from dask import delayed
# import dask.dataframe as dd

# @dask.delayed
# def compute_mk_trend(i, j, data_ij_df, test_type, data_name):
#     if data_ij_df[data_name].isnull().values.all():
#         MK_dict = {'lat': i, 'lon': j, 'MK_trend': np.nan, 'MK_slope': np.nan}
#     else:
#         if test_type == 'original':
#             MK_result = mk.original_test(data_ij_df[data_name].values)
#         elif test_type == 'hamed_rao':
#             MK_result = mk.hamed_rao_modification_test(data_ij_df[data_name].values)

#         MK_dict = {'lat': i, 'lon': j, 'MK_trend': MK_result.trend, 'MK_slope': MK_result.slope}

#     MK_df_ij = pd.DataFrame([MK_dict])
#     return MK_df_ij


# def find_MK_trendtest(data, test_type):
#     print('MK')
#     data_name = data.name
#     MK_dask_list = []

#     for i in data['lat'].values:
#         for j in data['lon'].values:
#             data_ij = data.sel(lat=i, lon=j)
#             data_ij_df = data_ij.to_dataframe()
#             data_ij_df.reset_index(inplace=True)

#             MK_dask = compute_mk_trend(i, j, data_ij_df, test_type, data_name)
#             MK_dask_list.append(MK_dask)

#     MK_dask_results = dask.compute(*MK_dask_list)
#     MK_df = pd.concat(MK_dask_results, ignore_index=True)

#     return MK_df


def save_stat_test_df(data, stat_test, model_type, measure, year_from, year_to, season='None', test_type='original'):
    """
    Saves stats test data as csv file.

    Args:
        data (xr.DataArray): data to perform stats test on
        stat_test (str): statistical test which the data is for ('MK' or 'DW')
        model_type (str): the model type which the data is for ('1980' or '1911')
        measure (str): the drought measure ('events' or 'proba')
        season (str): if requiring seasonal data then specify season (e.g. 'DJF'), else 'None'
        test_type (str): matters for stat_test = 'MK' only, type of MK test that was conducted
                         ('original' (default) or 'hamed_rao')
    """
    filepath = datadir + f'{stat_test}_test/drought_{measure}/{model_type}_model/{test_type}/'

    if not os.path.exists(filepath):
        os.makedirs(filepath)

    if stat_test == 'MK':
        df = find_MK_trendtest(data, test_type, year_from, year_to)
        filename = f'{year_from}-{year_to}_{test_type}_{stat_test}_test_drought_{measure}_{model_type}_model.csv'
        print(season)
        if season != 'None':
            print(season)
            filename = f'{season}_{filename}'
        print(filepath + filename)
    elif stat_test == 'DW':
        df = calculate_DW_score(data)
        filename = f'{year_from}-{year_to}_{stat_test}_test_drought_{measure}_{model_type}_model.csv'
    else:
        raise ValueError(f'{stat_test} is not a valid statistical test for this function.')
    
    df.to_csv(filepath + filename)


def find_season_data(data, season, aggregate_by):
    """
    Creates an dataarray of all the points in the specified season.

    Args:
        data (xr.DataArray): data to create seasonal data of
        season (str): season of interest to pull from data
        aggregate_by (str): how to aggregate the season data ('sum' or 'mean')

    Returns:
        data_seas (xr.DataArray): data but only for the specified season
    """
    season_dict = {
        'DJF': 12,
        'MAM': 3,
        'JJA': 6,
        'SON': 9
    }

    if aggregate_by == 'sum':
        data_allseas = data.resample(time='QS-DEC').sum(dim='time').isel(time=slice(1, -1))
    elif aggregate_by == 'mean':
        data_allseas = data.resample(time='QS-DEC').mean(dim='time').isel(time=slice(1, -1))

    data_seas = data_allseas.sel(time=data_allseas['time.month'] == season_dict[season])
    print(data_seas)
    return data_seas


def load_drought_data(model, measure, season='None'):
    """"
    Loads drought data.

    Args:
        model (str): the model of required data ('1980' or '1911')
        measure (str): the drought measure to load ('events' or 'proba')
        season (str): if requiring seasonal data then specify season (e.g. 'DJF'), else 'None'

    Returns:
        data (xr.DataArray): data array of the required model and drought measure
    """
    file = datadir + f'drought_prediction/{model}_model/drought_prediction_dataset_{model}_model.nc'
    drought_ds = xr.open_dataset(file)

    aggregate_dict = {
        'events': 'sum',
        'proba': 'mean',
    }

    measure_dict = {
        'events': drought_ds.drought,
        'proba': drought_ds.drought_proba,
    }

    if season == 'None':
        data = measure_dict[measure]
    else:
        data = find_season_data(measure_dict[measure], season, aggregate_dict[measure])

    return data
    

MODELS = [
    '1911', 
    '1980',
    # 'test',
]

YEARS = {
    '1911': [
        ['1911', '2021'], 
        ['1950', '2021'], 
        ['1980', '2021'],
        ['1950', '1980'],
    ],

    '1980': [
        ['1980', '2021'],
    ],

    'test': [
        ['1981', '1983']
    ]
}

# MEASURES = [
#     # 'proba', 
#     'events'
# ]

SEASONS = [
    'None',
    'DJF',
    'MAM',
    'JJA',
    'SON',
]

test_type = [
    'original',
    'hamed_rao',
    'yue_wang',
    'seasonal_sens_slope',
]


def main():
    save_stat_test_df(
        load_drought_data(
            '1911', 'proba', season='DJF'
        ), 
        'MK', '1911', 'proba', '1911', '1950', test_type='hamed_rao'
    )
    # for model in MODELS:
    #     for years in YEARS[model]:
    #         # save_stat_test_df(
    #         #     calculate_DW_score(
    #         #         load_drought_data(model, measure)
    #         #     ), 'DW', model, measure
    #         # )

    #         start_year = years[0]
    #         end_year = years[-1]
    #         for type in test_type:
    #             if type == 'seasonal_sens_slope':
    #                 save_stat_test_df(
    #                     load_drought_data(
    #                         model, 'proba', season=season
    #                     ), 
    #                     'MK', model, 'proba', start_year, end_year, test_type=type
    #                 )
    #             else:
    #                 for season in SEASONS:
    #                     save_stat_test_df(
    #                         load_drought_data(
    #                             model, 'proba', season=season
    #                         ), 
    #                         'MK', model, 'proba', start_year, end_year, season=season, test_type=type
    #                     )
            

    #             save_stat_test_df(
    #                 sum_drought_events_per_time_period(
    #                     load_drought_data(
    #                         model, 'events'
    #                     ), 
    #                     5
    #                 ), 
    #                 'MK', model, 'events', start_year, end_year, test_type=type
    #             )
                


if __name__ == "__main__":
    main()