import numpy as np
import pandas as pd
import xarray as xr
import os
from statsmodels.formula.api import ols
from statsmodels.stats.stattools import durbin_watson
import pymannkendall as mk


datadir = '/g/data/w97/mg5624/RF_project/'
plotdir = '/g/data/w97/mg5624/plots/RF_project/results_analysis/'


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


def find_MK_trendtest(data, test_type):
    """
    Computes the MK trendtest for each grid point in the data.

    Args:
        data (xr.DataArray): spatial and temporal data

    Returns:
        MK_data (pd.DataFrame): trend test result at each grid point
    """
    print('MK')
    data_name = data.name
    MK_df = pd.DataFrame()
    for i in data['lat'].values:
        for j in data['lon'].values:
            data_ij = data.sel(lat=i, lon=j)
            data_ij_df = data_ij.to_dataframe()
            data_ij_df.reset_index(inplace=True)

            # if all values are nan, then skip this grid point
            if data_ij_df[data_name].isnull().values.all():
                break

            if test_type == 'original':
                MK_result = mk.original_test(data_ij_df[data_name].values)
            elif test_type == 'hamed_rao':
                MK_result = mk.hamed_rao_modification_test(data_ij_df[data_name].values)

            MK_dict = {'lat': i, 'lon': j, 'MK_trend': MK_result.trend, 'MK_slope': MK_result.slope}
            MK_df_ij = pd.DataFrame([MK_dict])
            MK_df = pd.concat((MK_df, MK_df_ij))

    return MK_df


def save_stat_test_df(df, stat_test, model, measure, test_type='original'):
    """
    Saves df
    """
    filepath = datadir + f'{stat_test}_test/drought_{measure}/{model}_model/'

    if not os.path.exists(filepath):
        os.makedirs(filepath)

    if stat_test == 'MK':
        filename = f'{test_type}_{stat_test}_test_drought_prediction_{model}_model.csv'
    else:
        filename = f'{stat_test}_test_drought_{measure}_{model}_model.csv'

    df.to_csv(filepath + filename)


def load_drought_data(model, measure):
    """"
    Loads drought data
    """
    file = datadir + f'drought_prediction/{model}_model/drought_prediction_dataset_{model}_model.nc'
    drought_ds = xr.open_dataset(file)
    
    measure_dict = {
        'events': drought_ds.drought,
        'proba': drought_ds.drought_proba,
    }

    data = measure_dict[measure]

    return data


MODELS = ['1911', '1980']
MEASURES = ['proba']
test_type = [
    # 'original',
    'hamed_rao'
]


def main():
    for measure in MEASURES:
        for model in MODELS:
            save_stat_test_df(
                calculate_DW_score(
                    load_drought_data(model, measure)
                ), 'DW', model, measure
            )

            # for type in test_type:
            #     save_stat_test_df(
            #         find_MK_trendtest(
            #             load_drought_data(model, measure), type
            #         ), 'MK', model, measure, test_type=type
            #     )


if __name__ == "__main__":
    main()



