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

def load_MK_df(year_from, year_to, measure, model, grouped_by='not_grouped'):
    """
    Load the MK trendtest dataframe for the specified years, measure, model and group by.

    Args:
        year_from (str): year that the trend is from
        year_to (str): year the trend is until
        measure (str): measure for drought ('events' or 'proba')
        model (str): which model is the trend for ('1980' or '1911')
        grouped_by (str): if the trends are grouped by e.g. season, month, etc. indicate it here (default: not_grouped)

    Return:
        MK_df (pd.DataFrame): dataframe with MK trend and slope
    """
    filepath = datadir + f'MK_test/drought_{measure}/{model}_model/'
    filename = f'{year_from}-{year_to}_hamed_rao_MK_test_drought_{measure}_{model}_model.csv'

    MK_df = pd.read_csv(filepath + filename)

    return MK_df


def plot_MK_trendtest_results(MK_df, year_from, year_to, measure, model, grouped_by='not_grouped'):
    """
    Plots results from the MK trendtest. Hatching is overlayed to indicate where the trends are significant.

    Args:
        MK_df (pd.DataFrame): dataframe containing the MK slope and trend of drought probability
        year_from (str): year that the trend is from
        year_to (str): year the trend is until
        measure (str): measure for drought ('events' or 'proba')
        model (str): which model is the trend for ('1980' or '1911')
        grouped_by (str): if the trends are grouped by e.g. season, month, etc. indicate it here (default: not_grouped)
    """
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
    figpath = scratch + f'MK_trends/{model}/{measure}/{year_from}-{year_to}/{grouped_by}/'

    if not os.path.exists(figpath):
        os.makedirs(figpath)

    if grouped_by == 'not_grouped':
        group_by_label = ''
    else:
        group_by_label = f'{grouped_by}_'

    figname = f'{model}_model_{year_from}-{year_to}_{measure}_{group_by_label}MK_trend.png'
    plt.savefig(figpath + figname)
    plt.close()


def main():
    for model in MK_script.MODELS:
        for years in MK_script.YEARS[model]:
            for measure in MK_script.MEASURES:
                start_year = years[0]
                end_year = years[-1]

                plot_MK_trendtest_results(
                    load_MK_df(
                        start_year, end_year, measure, model
                    ), start_year, end_year, measure, model
                )


if __name__ == "__main__":
    main()
