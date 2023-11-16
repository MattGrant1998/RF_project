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
import matplotlib

datadir = '/g/data/w97/mg5624/RF_project/'
plotdir = '/g/data/w97/mg5624/plots/RF_project/results_analysis/'
scratch = '/scratch/w97/mg5624/plots/RF_project/results_analysis/'

drought_ds_1980 = xr.open_dataset(datadir + '/drought_prediction/1980_model/drought_prediction_dataset_1980_model.nc')
drought_ds_1911 = xr.open_dataset(datadir + '/drought_prediction/1911_model/drought_prediction_dataset_1911_model.nc')

drought_events_1980 = drought_ds_1980.drought
drought_events_1911 = drought_ds_1911.drought

drought_proba_1980 = drought_ds_1980.drought_proba
drought_proba_1911 = drought_ds_1911.drought_proba

MODELS = [
    '1980', 
    # '1911'
]

MEASURES = ['Probability Anomaly', 'Event']

# DATA = {
#     'Event': {'1980': drought_events_1980, '1911': drought_events_1911},
#     'Probability Anomaly': {'1980': anom_proba_1980, '1911': anom_proba_1911},
# }

DATA_PROBA = {
    '1980': drought_proba_1980,
    '1911': drought_proba_1911
}

DATA_EVENT = {
    '1980': drought_events_1980,
    '1911': drought_events_1911,
}

TIME_PERIODS = [
    # [1980, 2022],
    [1980, 1989],
    [1990, 1999],
    [2000, 2009],
    [2010, 2022],
    # [2015, 2022],
]

known_droughts = {
    '1980': [
        # '1982-83', 
        'millenium', 
        # 'tinderbox'
    ],
    '1911': [
        '1914-15', 
        'WWII', 
        '1965-68', 
        '1982-83', 
        'millenium', 
        'tinderbox'
    ],
}

known_droughts_years = {
    # '1914-15': list(range(1912, 1917)),
    # 'WWII': list(range(1935, 1948)),
    # '1965-68': list(range(1963, 1970)),
    # '1982-83': list(range(1981, 1985)),
    'millenium': list(range(1998, 2011)),
    # 'tinderbox': list(range(2015, 2021)),
}

METRICS = ['mean', 'max', 'min']


def create_anomaly_from_mean_proba(drought_proba_data):
    """
    Creates dataarray of the anomaly from the mean probability of drought over the dataset.

    Args:
        drought_proba_data (xr.DataArray): drought probability data over the whole timeseries
    
    Returns:
        anom_drought_proba (xr.DataArray): anomaly from mean probability of drought data
    """
    mean_drought_proba = drought_proba_data.mean(dim='time')
    anom_drought_proba = drought_proba_data - mean_drought_proba

    return anom_drought_proba


def create_anomaly_from_yearly_drought_events(drought_events_data):
    """
    Creates dataarray of the anomaly from mean of drought events per year.

    Args:
        drought_events_data (xr.DataArray): drought events data

    Returns:
        anom_drought_events (xr.DataArray): anomaly from the mean drought events per year
    """
    drought_events_per_year = drought_events_data.groupby('time.year').sum(dim='time')
    mean_drought_events_per_year = drought_events_per_year.mean('year')
    anom_drought_events_per_year = drought_events_per_year - mean_drought_events_per_year

    return anom_drought_events_per_year


def create_drought_events_per_year_plots(data, start_year, end_year, model_type, anom=False):
    """
    Creates figure of spatial maps of the droughts per year for each year from start_year to end_year (inclusive).

    Args:
        data (xr.DataArray): dataarray of the drought events, inclusive of the years from start_year to end_year
                             if anom=True this will be in the format of anomaly number of drought events per year
        start_year (int or str): first year of plots
        end_year (int or str): last year to plot
        model_type (str or int): the model that is being plotted (either '1980' or '1911')
    """
    if not anom:
        drought_events_per_year = data.groupby('time.year').sum('time')
    else:
        drought_events_per_year = data
        
    drought_events_per_year_const = drought_events_per_year.sel(year=slice(str(start_year), str(end_year)))
        
    num_plots = len(drought_events_per_year_const['year'].values)
    num_cols = 4
    num_rows = math.ceil(num_plots / num_cols)
    
    plt.figure(figsize=(16, 12), dpi=300)
    
    if anom:
        plt.suptitle('Anomaly of Number of Droughts per Year', fontsize=25)
        cmap = 'BrBG_r'
        vmax = 12
        vmin = -12
        figpath = scratch + f'/spatial_maps/drought_events/{model_type}_model/yearly/anom/'
        figname = f'anom_{model_type}_model_droughts_per_year_from_{start_year}_to_{end_year}.png'
    else:
        plt.suptitle('Number of Droughts per Year', fontsize=25)
        cmap = 'OrRd'
        vmax = 12
        vmin = 0
        figpath = scratch + f'/spatial_maps/drought_events/{model_type}_model/yearly/'
        figname = f'{model_type}_model_droughts_per_year_from_{start_year}_to_{end_year}.png'
        
    for i in range(num_rows):
        for j in range(num_cols):
            index = (i * num_cols) + j
            if index < num_plots:
                year_drought_map = drought_events_per_year_const.isel(year=index)      
                ax = plt.subplot2grid((num_rows, num_cols), (i, j), projection=ccrs.PlateCarree())
                year = year_drought_map['year'].values
                cbar_sets = {'fraction': 0.04, 'pad': 0.04, 'label': 'No. of Droughts'}
                plot = year_drought_map.plot.pcolormesh(cmap=cmap, vmin=vmin, vmax=vmax, cbar_kwargs=cbar_sets)
                ax.coastlines(resolution='50m')
                plt.title(year, fontsize=18)
    plt.tight_layout()
      
    if not os.path.exists(figpath):
        os.makedirs(figpath)
    
    plt.savefig(figpath + figname)
    plt.close()


def create_indiviual_year_of_drought_proba_plot(data, year, model_type):
    """
    Creates a spatial plot of mean drought probability anomaly for an individual year.

    Args:
        data (xr.DataArray): dataarray of the drought probability, inclusive of the years from start_year to end_year
        year (int or str): year to plot
        model_type (str or int): the model that is being plotted (either '1980' or '1911')
        metric (str): metric of the drought probability (e.g. 'mean', 'max')
    """
    yearly_drought_proba = data.groupby('time.year').mean('time')
    drought_proba_const = yearly_drought_proba.sel(year=year)
    plt.figure(figsize=(16, 12), dpi=300)

    plt.title(year, fontsize=44)
    cmap = 'BrBG_r'
    vmax = 0.5
    vmin=-0.5
    extend = 'both'
    figpath = scratch + f'/spatial_maps/drought_proba/{model_type}_model/poster_plots/'
    figname = f'anom_{model_type}_mean_drought_proba_for_{year}.png'
      
    ax = plt.subplot(projection=ccrs.PlateCarree())
    cbar_sets = {'fraction': 0.04, 'pad': 0.04, 'label': 'Probability of Drought', 'extend': extend}
    drought_proba_const.plot.pcolormesh(cmap=cmap, vmin=vmin, vmax=vmax, cbar_kwargs=cbar_sets)
    ax.coastlines(resolution='50m')

    if not os.path.exists(figpath):
        os.makedirs(figpath)
    
    plt.savefig(figpath + figname)
    plt.close()


def create_yearly_drought_proba_plots(data, start_year, end_year, model_type, metric, anom=False):
    """
    Creates figure of spatial maps of the average droughts probability for each year from start_year to end_year (inclusive).

    Args:
        data (xr.DataArray): dataarray of the drought probability, inclusive of the years from start_year to end_year
        start_year (int or str): first year of plots
        end_year (int or str): last year to plot
        model_type (str or int): the model that is being plotted (either '1980' or '1911')
        metric (str): metric of the drought probability (e.g. 'mean', 'max')
    """
    if metric == 'mean':
        yearly_drought_proba = data.groupby('time.year').mean('time')
    elif metric == 'max':
        yearly_drought_proba = data.groupby('time.year').max('time')
    elif metric == 'min':
        yearly_drought_proba = data.groupby('time.year').min('time')
        
    drought_proba_const = yearly_drought_proba.sel(year=slice(str(start_year), str(end_year)))
    num_plots = len(drought_proba_const['year'].values)
    num_cols = 4
    num_rows = math.ceil(num_plots / num_cols)
    
    plt.figure(figsize=(16, 12), dpi=300)
    

    if anom:
        plt.suptitle(f'{metric[0].upper() + metric[1:]} Drought Probability Anomaly for the {model_type} model', fontsize=25)
        cmap = 'BrBG_r'
        vmax = 0.5
        vmin=-0.5
        extend = 'both'
        figpath = scratch + f'/spatial_maps/drought_proba/{model_type}_model/yearly/{metric}/anom/'
        figname = f'anom_{model_type}_{metric}_drought_proba_from_{start_year}_to_{end_year}.png'
    else:
        plt.suptitle(f'{metric[0].upper() + metric[1:]} Drought Probability', fontsize=25)
        cmap = 'Reds'
        vmax=1
        vmin=0
        extend = 'neither'
        figpath = scratch + f'/spatial_maps/drought_proba/{model_type}_model/yearly/{metric}/'
        figname = f'{model_type}_{metric}_drought_proba_from_{start_year}_to_{end_year}.png'
        
    for i in range(num_rows):
        for j in range(num_cols):
            index = (i * num_cols) + j
            if index < num_plots:
                year_drought_map = drought_proba_const.isel(year=index)      
                ax = plt.subplot2grid((num_rows, num_cols), (i, j), projection=ccrs.PlateCarree())
                year = year_drought_map['year'].values
                cbar_sets = {'fraction': 0.04, 'pad': 0.04, 'label': 'Probability of Drought', 'extend': extend}
                plot = year_drought_map.plot.pcolormesh(cmap=cmap, vmin=vmin, vmax=vmax, cbar_kwargs=cbar_sets)
                ax.coastlines(resolution='50m')
                plt.title(year, fontsize=18)
    plt.tight_layout()
    
    
    if not os.path.exists(figpath):
        os.makedirs(figpath)
    
    plt.savefig(figpath + figname)
    plt.close()


def create_monthly_drought_proba_or_event_maps_for_specified_year(data, year, model_type, drought_measure):
    """
    Creates a spatial map of drought probability anomaly or drought events from mean for each month in the specified year.

    Args:
        data (xr.DataArray): drought probability anomaly from mean
        year (int): year of interest
        model_type (str): the model to plot ('1911' or '1980' model)
        drought_measure (str): which measure of drought to plot: 'Probability Anomaly' or 'Events'
        known_drought (str): name of the known drought comapring to (if not comparing to known drought, put 'None')
    """
    year_data = data.sel(time=slice(str(year) + '-01', str(year) + '-12'))
    num_rows = 4
    num_cols = 3

    plt.figure(figsize=(16, 16), dpi=300)
    plot_dict = {
        'Probability Anomaly': {
            'vmin': -0.7, 
            'vmax': 0.7,
            'cmap': 'BrBG_r',
            'extend': 'both'
        },
        'Event': {
            'vmin': 0, 
            'vmax': 1,
            'cmap': matplotlib.colors.ListedColormap(['white', 'red']),
            'extend': 'neither'
        }
    }
    vmin = plot_dict[drought_measure]['vmin']
    vmax = plot_dict[drought_measure]['vmax']
    cmap = plot_dict[drought_measure]['cmap']
    extend = plot_dict[drought_measure]['extend']
    
    months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'June', 'July', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    plt.suptitle(f'Drought {drought_measure} for the {model_type} Model', fontsize=25)
    for i in range(num_rows):
        for j in range(num_cols):
            index = (i * num_cols) + j
            month_drought_map = year_data.isel(time=index)
            ax = plt.subplot2grid((num_rows, num_cols), (i, j), projection=ccrs.PlateCarree())
            year = month_drought_map['time.year'].values
            month = months[index]
            cbar_sets = {'fraction': 0.04, 'pad': 0.04, 'label': f'Drought {drought_measure}', 'extend': extend}
            plot = month_drought_map.plot.pcolormesh(cmap=cmap, vmin=vmin, vmax=vmax, cbar_kwargs=cbar_sets)
            ax.coastlines(resolution='50m')
            plt.title(f'{month} {year}', fontsize=18)
    plt.tight_layout()

    for category, year_range in known_droughts_years.items():
        if year in year_range:
            drought_event = category
            break
        else:
            drought_event = 'other_years'

    measure_filepath = {
        'Probability Anomaly': 'drought_proba',
        'Event': 'drought_events'
    }
    
    figpath = scratch + f'/spatial_maps/{measure_filepath[drought_measure]}/{model_type}_model/monthly/{drought_event}/'
    figname = f'{model_type}_{measure_filepath[drought_measure]}_for_{year}.png'

    if not os.path.exists(figpath):
        os.makedirs(figpath)

    plt.savefig(figpath + figname)
    plt.close()
    

def main():
    # for period in TIME_PERIODS:
    #     start_year = period[0]
    #     end_year = period[-1]

    #     for model in MODELS:
    #         create_drought_events_per_year_plots(DATA_EVENT[model], start_year, end_year, model)
    #         anom_drought_event = create_anomaly_from_yearly_drought_events(DATA_EVENT[model])
    #         create_drought_events_per_year_plots(anom_drought_event, start_year, end_year, model, anom=True)
        
    #         create_yearly_drought_proba_plots(DATA_PROBA[model], start_year, end_year, model, 'mean')
    #         anom_drought_proba = create_anomaly_from_mean_proba(DATA_PROBA[model])
    #         create_yearly_drought_proba_plots(anom_drought_proba, start_year, end_year, model, 'mean', anom=True)
    
    for model in MODELS:
        for drought in known_droughts[model]:
            for year in known_droughts_years[drought]:
                
                # create_monthly_drought_proba_or_event_maps_for_specified_year(DATA_EVENT[model], year, model, 'Event')

                anom_drought_proba = create_anomaly_from_mean_proba(DATA_PROBA[model])
                create_monthly_drought_proba_or_event_maps_for_specified_year(anom_drought_proba, year, model, 'Probability Anomaly')
                # create_indiviual_year_of_drought_proba_plot(anom_drought_proba, year, model)
                # create_monthly_drought_proba_or_event_maps_for_specified_year()


if __name__ == "__main__":
    main()
    