import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning) 

# Data

import numpy as np
import pandas as pd
import xarray as xr
import bottleneck as bn
import iris
import os

# Plotting

import matplotlib.pyplot as plt
import seaborn as sns
import geopandas as gpd
from shapely.geometry import Point

# Scikit-learn

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, balanced_accuracy_score

# Save filepaths
plotdir = '/g/data/w97/mg5624/plots/RF_project/'
datadir = '/g/data/w97/mg5624/RF_project/'


MODEL_TYPES = [
    'full',
    'long_ts',
    # 'test',
]

AREAS = [
    # 'test',
    'SE_australia',
]

PREDICTORS = {
    'test': ['Precipitation', 'Runoff', 'ENSO_index', 'PET'],
    'full': ['Precipitation', 'Acc_3-Month_Precipitation', 'Acc_6-Month_Precipitation', 'Acc_12-Month_Precipitation', 
             'Acc_24-Month_Precipitation', 'Runoff', 'ENSO_index', 'IOD_index', 'SAM_index', 'ET', 'PET', 'SMsurf', 
             'SMroot', 'Sin_month', 'Cos_month'],
    'long_ts': ['Precipitation', 'Acc_3-Month_Precipitation', 'Acc_6-Month_Precipitation', 'Acc_12-Month_Precipitation', 
                'Acc_24-Month_Precipitation', 'Acc_12-Month_Precipitation', 'Runoff', 'ENSO_index', 'IOD_index', 
                'Sin_month', 'Cos_month'],
}

TIME_PERIODS = {
    'test': [1981, 1983],
    'full': [1980, 2022], 
    'long_ts': [1911, 2022]
}


def train_RF_model(data, predictors, target, random_seed, n_estimators):
    """
    Trains a random forest model.
    Args:
        data (pd.DataFrame): Contains all training data
        predictors (list): list of strings of the names of the predictor variables
        target (str): name of the target variable ('Drought')
        random_seed (int): random seed value for the RF model
        n_estimators (int): number of estimators to builid RF model with

    Returns:
        clf (sklearn.RandomForestClassifier): trained random forest classifier
    """        
    X = data[predictors]
    y = data[target]
        
    # Create and train the Random Forest model
    clf = RandomForestClassifier(n_estimators=n_estimators, random_state=random_seed)
    clf.fit(X, y)

    return clf


def predict_droughts(RF_model, predictors_df, predictors, model_type):
    """
    Uses the RF model to predict droughts for timeframe and area covered by predictors_df.

    Args:
        RF_model (sklearn.RandomForestClassifier): the trained RF model
        predictors_df (pd.Dataframe): data of all the precitors used to train the RF over the desired area and timeframe
        predictors (list): list of all the predictors used to train the RF model
        model_type (str): the type of model we are training (either 'full' or 'long_ts')

    Returns:
        drought_prediction_ds (xr.Dataset): data of 'Drought / No Drought' and 'Drought Probability'
    """
    # Predict droughts
    predictors_df.dropna(axis=0, inplace=True)
    print(predictors_df['time'])
    print(predictors_df.loc[predictors_df['time']=='1981-02-01'])
    predictors_data = predictors_df[predictors]
    predicted_drought_bin = RF_model.predict(predictors_data)
    predicted_drought_proba = RF_model.predict_proba(predictors_data)

    # Create drought prediction df
    drought_prediction_df = predictors_df[['time', 'lat', 'lon']].copy()
    drought_prediction_df['drought'] = predicted_drought_bin

    drought_proba_arr = predicted_drought_proba[:, 1]
    no_drought_proba_arr = predicted_drought_proba[:, 0]
    drought_prediction_df['drought_proba'] = drought_proba_arr
    drought_prediction_df['no_drought_proba'] = no_drought_proba_arr

    # Convert dataframe to dataset and save
    drought_prediction_df['time'] = pd.to_datetime(drought_prediction_df['time'])
    drought_prediction_df = drought_prediction_df.set_index(['time', 'lat', 'lon'])
    drought_prediction_ds = drought_prediction_df.to_xarray()

    filepath = datadir + f'/drought_prediction/{model_type}_model/'
    if not os.path.exists(filepath):
        os.makedirs(filepath)
    print(drought_prediction_ds['time'])
    filename = f'drought_prediction_dataset_{model_type}_model.nc'
    drought_prediction_ds.to_netcdf(filepath + filename)


def main():
    training_data = pd.read_csv(datadir + '/training_data/training_data.csv')
    training_data.dropna(axis=0, inplace=True)
    for area in AREAS:
        for model in MODEL_TYPES:
            time_period = TIME_PERIODS[model]
            start_year = time_period[0]
            end_year = time_period[-1]
            predictors_data_filepath = datadir + f'/predictors_data/{model}_model/'
            predictors_data_filename = f'predictors_dataframe_{start_year}-{end_year}_{area}.csv'
            predictors_data = pd.read_csv(predictors_data_filepath + predictors_data_filename)
            
            predict_droughts(
                train_RF_model(
                    training_data, PREDICTORS[model], 'Drought', 42, 500
                ),
                predictors_data, 
                PREDICTORS[model], 
                model
            )


if __name__ == "__main__":
    main()
    