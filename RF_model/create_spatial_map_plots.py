import math
import cartopy.crs as ccrs
import cartopy
import cartopy.feature as cfeature
import xarray as xr
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns

datadir = '/g/data/w97/mg5624/RF_project/'
plotdir = '/g/data/w97/mg5624/plots/RF_project/results_analysis/'
scratch = '/scratch/w97/mg5624/plots/RF_project/results_analysis/'

drought_ds_full = xr.open_dataset(datadir + '/drought_prediction/full_model/drought_prediction_dataset_full_model.nc')
drought_ds_long_ts = xr.open_dataset(datadir + '/drought_prediction/long_ts_model/drought_prediction_dataset_long_ts_model.nc')

drought_events_full = drought_ds_full.drought
drought_events_long_ts = drought_ds_long_ts.drought

drought_proba_full = drought_ds_full.drought_proba
drought_proba_long_ts = drought_ds_long_ts.drought_proba


start_years = [1980, 1990, 2000, 2010]
end_years = [1989, 1999, 2009, 2022]
METRICS = ['mean', 'max', 'min']

# Number of drought events per year
def create_drought_events_per_year_plots(data, start_year, end_year, model_type):
    """
    Creates figure of spatial maps of the droughts per year for each year from start_year to end_year (inclusive).

    Args:
        data (xr.DataArray): dataarray of the drought events, inclusive of the years from start_year to end_year
        start_year (int or str): first year of plots
        end_year (int or str): last year to plot
        model_type (str or int): the model that is being plotted (either '1980' or '1911')
    """
    drought_events_per_year = drought_events_full.groupby('time.year').sum('time')
    drought_events_per_year_const = drought_events_per_year.sel(year=slice(str(start_year), str(end_year)))
    num_plots = len(drought_events_per_year_const['year'].values)
    num_cols = 4
    num_rows = math.ceil(num_plots / num_cols)
    
    plt.figure(figsize=(16, 12), dpi=300)
    plt.suptitle('Number of Droughts per Year', fontsize=25)
    for i in range(num_rows):
        for j in range(num_cols):
            index = (i * num_cols) + j
            if index < num_plots:
                year_drought_map = drought_events_per_year_const.isel(year=index)      
                ax = plt.subplot2grid((num_rows, num_cols), (i, j), projection=ccrs.PlateCarree())
                year = year_drought_map['year'].values
                cbar_sets = {'fraction': 0.04, 'pad': 0.04, 'label': 'No. of Droughts'}
                plot = year_drought_map.plot.pcolormesh(cmap='OrRd', vmin=0, vmax=12, cbar_kwargs=cbar_sets)
                ax.coastlines(resolution='50m')
                plt.title(year, fontsize=18)
                # plt.tight_layout()
    
    figpath = scratch + f'/spatial_maps/drought_events/{model_type}_model/'
    if not os.path.exists(figpath):
        os.makedirs(figpath)
    
    figname = f'{model_type}_model_droughts_per_year_from_{start_year}_to_{end_year}.png'
    plt.savefig(figpath + figname)


def create_yearly_drought_proba_plots(data, start_year, end_year, model_type, metric):
    """
    Creates figure of spatial maps of the droughts per year for each year from start_year to end_year (inclusive).

    Args:
        data (xr.DataArray): dataarray of the drought events, inclusive of the years from start_year to end_year
        start_year (int or str): first year of plots
        end_year (int or str): last year to plot
        model_type (str or int): the model that is being plotted (either '1980' or '1911')
        metric (str): metric of the drought probability (e.g. 'mean', 'max')
    """
    if metric == 'mean':
        yearly_drought_proba = drought_events_full.groupby('time.year').mean('time')
    elif metric == 'max':
        yearly_drought_proba = drought_events_full.groupby('time.year').max('time')
    elif metric == 'min':
        yearly_drought_proba = drought_events_full.groupby('time.year').min('time')
        
    drought_proba_const = yearly_drought_proba.sel(year=slice(str(start_year), str(end_year)))
    num_plots = len(drought_proba_const['year'].values)
    num_cols = 4
    num_rows = math.ceil(num_plots / num_cols)
    
    plt.figure(figsize=(16, 12), dpi=300)
    plt.suptitle(f'{metric[0].upper() + metric[1:]} Drought Probability', fontsize=25)
    for i in range(num_rows):
        for j in range(num_cols):
            index = (i * num_cols) + j
            if index < num_plots:
                year_drought_map = drought_proba_const.isel(year=index)      
                ax = plt.subplot2grid((num_rows, num_cols), (i, j), projection=ccrs.PlateCarree())
                year = year_drought_map['year'].values
                cbar_sets = {'fraction': 0.04, 'pad': 0.04, 'label': 'Probability of Drought'}
                plot = year_drought_map.plot.pcolormesh(cmap='Reds', vmin=0, vmax=1, cbar_kwargs=cbar_sets)
                ax.coastlines(resolution='50m')
                plt.title(year, fontsize=18)
                # plt.tight_layout()
    
    figpath = scratch + f'/spatial_maps/drought_proba/{model_type}_model/{metric}/'
    if not os.path.exists(figpath):
        os.makedirs(figpath)
    
    figname = f'{model_type}_{metric}_drought_proba_from_{start_year}_to_{end_year}.png'
    plt.savefig(figpath + figname)


def main():
    for i, start_year in enumerate(start_years):
        end_year = end_years[i]
        create_drought_events_per_year_plots(drought_events_full, start_year, end_year, '1980')
        create_drought_events_per_year_plots(drought_events_long_ts, start_year, end_year, '1911')

        for metric in METRICS:
            end_year = end_years[i]
            create_yearly_drought_proba_plots(drought_events_full, start_year, end_year, '1980', metric)
            create_yearly_drought_proba_plots(drought_events_long_ts, start_year, end_year, '1911', metric)
            

if __name__ == "__main__":
    main()