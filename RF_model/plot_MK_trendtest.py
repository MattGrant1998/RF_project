import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
import numpy as np
import os
import matplotlib as mpl
import DW_and_MK_tests as MK_script


datadir = '/g/data/w97/mg5624/RF_project/'
plotdir = '/g/data/w97/mg5624/plots/RF_project/results_analysis/'
scratch = '/scratch/w97/mg5624/plots/RF_project/results_analysis/'


MK_df = pd.read_csv('/g/data/w97/mg5624/RF_project/MK_test/drought_proba/1980_model/hamed_rao_MK_test_drought_proba_1980_model.csv')

def load_MK_df(year_from, year_to, measure, model, test_type, season='None'):
    """
    Load the MK trendtest dataframe for the specified years, measure, model and group by.

    Args:
        year_from (str): year that the trend is from
        year_to (str): year the trend is until
        measure (str): measure for drought ('events' or 'proba')
        model (str): which model is the trend for ('1980' or '1911')
        season (str): default season = 'None' means data not split into seasons, otherwise specify season (e.g. 'DJF', etc.)

    Return:
        MK_df (pd.DataFrame): dataframe with MK trend and slope
    """
    filepath = datadir + f'MK_test/drought_{measure}/{model}_model/{test_type}/'
    filename = f'{year_from}-{year_to}_{test_type}_MK_test_drought_{measure}_{model}_model.csv'

    if season != 'None':
        filename = f'{season}_{filename}'

    MK_df = pd.read_csv(filepath + filename)

    return MK_df


def plot_MK_trendtest_results(year_from, year_to, measure, model, test_type, season='None'):
    """
    Plots results from the MK trendtest. Hatching is overlayed to indicate where the trends are significant.

    Args:
        MK_df (pd.DataFrame): dataframe containing the MK slope and trend of drought probability
        year_from (str): year that the trend is from
        year_to (str): year the trend is until
        measure (str): measure for drought ('events' or 'proba')
        model (str): which model is the trend for ('1980' or '1911')
        season (str): default season = 'None' means data not split into seasons, otherwise specify season (e.g. 'DJF', etc.)
    """
    MK_df = load_MK_df(year_from, year_to, measure, model, test_type, season='None')
    MK_df['MK_trend'].replace({'no trend': 0, 'decreasing': -1, 'increasing': 1}, inplace=True)
    MK_df.set_index(['lat', 'lon'], inplace=True)
    MK_da = MK_df.to_xarray()
    trend = MK_da.MK_trend
    slope = MK_da.MK_slope
    significant = trend != 0
    trend_masked = trend.where(significant)
    plt.figure()
    slope.plot()
    trend_masked.plot.contourf(hatches=[5*'/'], colors='none', levels=(0.5, 1), add_colorbar=False)
    mpl.rcParams['hatch.linewidth'] = 0.35
    figpath = scratch + f'MK_trends/{model}/{test_type}/{measure}/{year_from}-{year_to}/'
    figname = f'{model}_model_{year_from}-{year_to}_{measure}_{test_type}_MK_trend.png'

    if season != 'None':
        figpath = f'{figpath}/{season}/'
        figname = f'{season}_{figname}'

    if not os.path.exists(figpath):
        os.makedirs(figpath)

    
    # print(figname + figpath)
    plt.savefig(figpath + figname)
    plt.close()



def main():
    for model in MK_script.MODELS:
        for years in MK_script.YEARS[model]:
            start_year = years[0]
            end_year = years[-1]
            for type in MK_script.TEST_TYPE:
                # if type == 'seasonal_sens_slope':
                #     plot_MK_trendtest_results(
                #         start_year, end_year, 'proba', model, type
                #     )

                # else:
                #     for season in MK_script.SEASONS:
                #         plot_MK_trendtest_results(
                #             start_year, end_year, 'proba', model, type, season=season
                #         )

                plot_MK_trendtest_results(
                    start_year, end_year, 'events', model, type
                )



if __name__ == "__main__":
    main()
