import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning) 

import os
import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, balanced_accuracy_score, confusion_matrix

plotdir = '/g/data/w97/mg5624/plots/RF_project/model_analytics/'
datadir = '/g/data/w97/mg5624/RF_project/'

training_data = pd.read_csv(datadir + 'training_data/training_data.csv')
training_data.dropna(axis=0, inplace=True)

predictors_new = [
    'Precipitation', 'Acc_3-Month_Precipitation', 'Acc_6-Month_Precipitation', 'Acc_12-Month_Precipitation', 'Acc_24-Month_Precipitation', 
    'Runoff', 'Mean_3-Month_Runoff', 'Mean_6-Month_Runoff', 'Mean_12-Month_Runoff', 'Mean_24-Month_Runoff', 'ENSO_index', 'IOD_index', 
    'SAM_index', 'ET', 'Mean_3-Month_ET', 'Mean_6-Month_ET', 'Mean_12-Month_ET', 'Mean_24-Month_ET', 'PET', 'Mean_3-Month_PET',
    'Mean_6-Month_PET', 'Mean_12-Month_PET', 'Mean_24-Month_PET', 'SMsurf', 'Mean_3-Month_SMsurf', 'Mean_6-Month_SMsurf', 
    'Mean_12-Month_SMsurf', 'Mean_24-Month_SMsurf', 'SMroot', 'Mean_3-Month_SMroot', 'Mean_6-Month_SMroot', 'Mean_12-Month_SMroot', 
    'Mean_24-Month_SMroot', 'Sin_month', 'Cos_month'
]

predictors_new_simplified = [
    'Acc_12-Month_Precipitation', 'Mean_12-Month_Runoff', 'ENSO_index', 'IOD_index', 
    'SAM_index', 'Mean_12-Month_ET', 'Mean_12-Month_PET', 'Mean_12-Month_SMsurf', 'Mean_12-Month_SMroot', 
    'Sin_month', 'Cos_month'
]

# predictors with all variables
predictors_1980 = [
    'Precipitation', 'Acc_3-Month_Precipitation', 'Acc_6-Month_Precipitation', 
    'Acc_12-Month_Precipitation', 'Acc_24-Month_Precipitation', 'Runoff', 
    'ENSO_index', 'IOD_index', 'SAM_index', 'ET', 'PET', 'SMsurf', 
    'SMroot', 'Sin_month', 'Cos_month'
]


# predictors of variables with timeseries back to 1950 or ealier (these go back to at least 1911)
predictors_1911 = [
    'Precipitation', 'Acc_3-Month_Precipitation', 'Acc_6-Month_Precipitation', 
    'Acc_12-Month_Precipitation', 'Acc_24-Month_Precipitation', 
    'Runoff', 'ENSO_index', 'IOD_index', 'Sin_month', 'Cos_month'
]


predictors_1911_simp = [
    'Acc_12-Month_Precipitation', 'Mean_12-Month_Runoff', 'ENSO_index', 'IOD_index', 'Sin_month', 'Cos_month'
]

variable_to_label = {
    'Precipitation': 'Precipitation',
    'Acc_3-Month_Precipitation': '3-Month Acc Precipitation',
    'Acc_6-Month_Precipitation': '6-Month Acc Precipitation',
    'Acc_12-Month_Precipitation': '12-Month Acc Precipitation',
    'Acc_24-Month_Precipitation': '24-Month Acc Precipitation',
    'Runoff': 'Runoff',
    'Mean_3-Month_Runoff': '3-Month Mean Runoff', 
    'Mean_6-Month_Runoff': '6-Month Mean Runoff', 
    'Mean_12-Month_Runoff': '12-Month Mean Runoff', 
    'Mean_24-Month_Runoff': '24-Month Mean Runoff',
    'ENSO_index': 'ENSO',
    'IOD_index': 'IOD',
    'SAM_index': 'SAM',
    'ET': 'ET',
    'Mean_3-Month_ET': '3-Month Mean ET',
    'Mean_6-Month_ET': '6-Month Mean ET',
    'Mean_12-Month_ET': '12-Month Mean ET',
    'Mean_24-Month_ET': '24-Month Mean ET',
    'PET': 'PET',
    'Mean_3-Month_PET': '3-Month Mean PET',
    'Mean_6-Month_PET': '6-Month Mean PET',
    'Mean_12-Month_PET': '12-Month Mean PET',
    'Mean_24-Month_PET': '24-Month Mean PET',
    'SMsurf': 'Surface SM',
    'Mean_3-Month_SMsurf': '3-Month Mean Surface SM',
    'Mean_6-Month_SMsurf': '6-Month Mean Surface SM',
    'Mean_12-Month_SMsurf': '12-Month Mean Surface SM',
    'Mean_24-Month_SMsurf': '24-Month Mean Surface SM',
    'SMroot': 'Root SM',
    'Mean_3-Month_SMroot': '3-Month Mean Root SM',
    'Mean_6-Month_SMroot': '6-Month Mean Root SM',
    'Mean_12-Month_SMroot': '12-Month Mean Root SM',
    'Mean_24-Month_SMroot': '24-Month Mean Root SM',
    'Sin_month': 'Sine of Month',
    'Cos_month': 'Cosine of Month',
}

target = 'Drought'

model_types = [
    # '1980',
    '1911',
    '1911_simp',
    # 'new',
    # 'new_simp',
]

model_title = {
        "1980": "1980",
        "1911": "1911",
        "1911_simp": "1911 (simplified)"
        "new": "New",
        "new_simp": "New (simplified)"
    }

predictors_dict = {
    '1980': predictors_1980,
    '1911': predictors_1911,
    '1911_simp': predictors_1911_simp
    'new': predictors_new,
    'new_simp': predictors_new_simplified,
}

y = training_data['Drought']


def calculate_performance_metrics(y_test, y_pred):
    """
    Calculates performance metrics from the RF model
    Args:
    y_test: Data held back from data split for testing
    y_pred: Prediction made by RF Classifier model
    """
    # Calculate performance metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    balanced_accuracy = balanced_accuracy_score(y_test, y_pred)
    confusion_matrix = sklearn.metrics.confusion_matrix(y_test, y_pred, normalize='all')
    false_alarm = confusion_matrix[0, 1]
    
    # Save results in DataFrame
    performance_data = {
        'Accuracy': [accuracy],
        'Precision': [precision],
        'Recall': [recall],
        'F1-Score': [f1],
        'Balance Accuracy': [balanced_accuracy],
        'False Alarm': [false_alarm]
    }

    performance_df = pd.DataFrame(performance_data)
    
    return performance_df


def create_performance_metric_bar_chart(performance_df, random_seed, model_type):
    """
    Creates a bar chart plot of the performance metrics for the model type specified.

    Args:
        performance_df (pd.DataFrame): performance metrics stored in a dataframe
        random_seed (int or str):  if performance metrics from one random_seed then integer value of that seed,
        if performance metrics are averaged from many random_seeds then "average_score"
        modle_type (str): the model type the performance metrics are for, either '1980', '1911', or 'new'
    """
    performance_df = performance_df.T
    ax = performance_df.plot(kind='bar', figsize=(12, 6), legend=False)
    ax.figure.subplots_adjust(bottom=0.22)
    if isinstance(random_seed, int):
        random_seed_title = f'Random Seed {random_seed}'
    else:
        random_seed_title = f'Average Scores of Multiple Iterations'
    plt.title(f'{model_type} Model Performance Metric for {random_seed_title}')
    plt.xlabel('Performance Metric')
    plt.ylabel('Scores')
    plt.xticks(ha='right', rotation=45)

    # Add labels on top of each bar
    for i in ax.patches:
        ax.text(i.get_x() + i.get_width() / 2, i.get_height() + 0.01, round(i.get_height(), 3), ha='center')

    # Save figure
    figpath = plotdir + f'/performance_metrics/{model_type}/'

    if not os.path.exists(figpath):
        os.makedirs(figpath)
        
    if isinstance(random_seed, int):
        figname = f'{model_type}_model_performance_metrics_for_seed{random_seed}.png'
    else:
        figname = f'{model_type}_model_performance_metrics_{random_seed}.png'

    plt.savefig(figpath + figname)
    plt.close()


def create_bar_chart_to_compare_two_performance_metrics(
    performance_df_model1, performance_df_model2, model1_name, model2_name, random_seed
):
    """
    Creates a bar plot to compare the performance metrics of the 1980 model and
    long timeseries model.
    
    Args:
        performance_df_1980 (pd.DataFrame): the performance metrics for model 1
        performance_df_1911 (pd.DataFrame): the performance metrics for the model 2
        model1_name (str): name of model 1
        model2_name (str): name of model 2
        random_seed (int or str):  if performance metrics from one random_seed then integer value of that seed,
        if performance metrics are averaged from many random_seeds then "average_score"
    """
    # Create a barplot comparing the scores of the two models
    performance_frames = [performance_df_model1, performance_df_model2]
    concat_performance_df = pd.concat(performance_frames)
    
    model1_title, model2_title = model_title[model1_name], model_title[model2_name]
    concat_performance_df.index = [model1_title, model2_title]
    concat_performance_df = concat_performance_df.T
    
    ax = concat_performance_df.plot(kind='bar', figsize=(12, 6), 
                                    color=['coral', 'lightskyblue'])
    ax.figure.subplots_adjust(bottom=0.22)
    if isinstance(random_seed, int):
        random_seed_title = f'Random Seed {random_seed}'
    else:
        random_seed_title = f'Average Scores of Multiple Iterations'
    plt.title(f'Model Performance Metrics for {random_seed_title}')
    plt.xlabel('Performance Metric')
    plt.ylabel('Scores')
    plt.xticks(ha='right', rotation=45)
    plt.legend(title='Models', loc='upper left', bbox_to_anchor=(1.0, 1.0))

    # Add labels on top of each bar
    for i in ax.patches:
        ax.text(i.get_x() + i.get_width() / 2, i.get_height() + 0.01, round(i.get_height(), 2), ha='center')

    # Save figure
    figpath = plotdir + '/performance_metrics/comparisons/'

    if not os.path.exists(figpath):
        os.makedirs(figpath)
        
    if isinstance(random_seed, int):
        figname = f'{model1_name}_vs_{model2_name}_model_performance_metrics_for_seed{random_seed}.png'
    else:
        figname = f'{model1_name}_vs_{model2_name}_model_performance_metrics_{random_seed}.png'

    plt.savefig(figpath + figname)
    plt.close()

def create_bar_chart_1980_vs_1911_performance_metrics(
    performance_df_1980, performance_df_1911, random_seed
):
    """
    Creates a bar plot to compare the performance metrics of the 1980 model and
    long timeseries model.
    
    Args:
        performance_df_1980 (pd.DataFrame): the performance metrics for the 1980 model
        performance_df_1911 (pd.DataFrame): the performance metrics for the long timeseries model
        random_seed (int or str):  if performance metrics from one random_seed then integer value of that seed,
        if performance metrics are averaged from many random_seeds then "average_score"
    """
    # Create a barplot comparing the scores of the two models
    performance_frames = [performance_df_1980, performance_df_1911]
    concat_performance_df = pd.concat(performance_frames)
    
    concat_performance_df.index = ['1980', '1911']
    concat_performance_df = concat_performance_df.T
    
    ax = concat_performance_df.plot(kind='bar', figsize=(12, 6), 
                                    color=['coral', 'lightskyblue'])
    ax.figure.subplots_adjust(bottom=0.22)
    if isinstance(random_seed, int):
        random_seed_title = f'Random Seed {random_seed}'
    else:
        random_seed_title = f'Average Scores of Multiple Iterations'
    plt.title(f'Model Performance Metrics for {random_seed_title}')
    plt.xlabel('Performance Metric')
    plt.ylabel('Scores')
    plt.xticks(ha='right', rotation=45)
    plt.legend(title='Models', loc='upper left', bbox_to_anchor=(1.0, 1.0))

    # Add labels on top of each bar
    for i in ax.patches:
        ax.text(i.get_x() + i.get_width() / 2, i.get_height() + 0.01, round(i.get_height(), 2), ha='center')

    # Save figure
    figpath = plotdir + '/performance_metrics/1980_vs_1911_comparison/'

    if not os.path.exists(figpath):
        os.makedirs(figpath)
        
    if isinstance(random_seed, int):
        figname = f'1980_vs_1911_model_performance_metrics_for_seed{random_seed}.png'
    else:
        figname = f'1980_vs_1911_model_performance_metrics_{random_seed}.png'

    plt.savefig(figpath + figname)
    plt.close()


def create_bar_chart_1980_vs_1911_vs_new_performance_metrics(
    performance_df_1980, performance_df_1911, performance_df_new, random_seed
):
    """
    Creates a bar plot to compare the performance metrics of the 1980 model and
    long timeseries model.
    
    Args:
        performance_df_1980 (pd.DataFrame): the performance metrics for the 1980 model
        performance_df_1911 (pd.DataFrame): the performance metrics for the long timeseries model
        random_seed (int or str):  if performance metrics from one random_seed then integer value of that seed,
        if performance metrics are averaged from many random_seeds then "average_score"
    """
    # Create a barplot comparing the scores of the two models
    performance_frames = [performance_df_1980, performance_df_1911, performance_df_new]
    concat_performance_df = pd.concat(performance_frames)
    
    concat_performance_df.index = ['1980', '1911', 'New']
    concat_performance_df = concat_performance_df.T
    
    ax = concat_performance_df.plot(kind='bar', figsize=(12, 6))
    ax.figure.subplots_adjust(bottom=0.22)
    if isinstance(random_seed, int):
        random_seed_title = f'Random Seed {random_seed}'
    else:
        random_seed_title = f'Average Scores of Multiple Iterations'
    plt.title(f'Model Performance Metrics for {random_seed_title}')
    plt.xlabel('Performance Metric')
    plt.ylabel('Scores')
    plt.xticks(ha='right', rotation=45)
    plt.legend(title='Models', loc='upper left', bbox_to_anchor=(1.0, 1.0))

    # Add labels on top of each bar
    for i in ax.patches:
        ax.text(i.get_x() + i.get_width() / 2, i.get_height() + 0.01, round(i.get_height(), 2), ha='center')

    # Save figure
    figpath = plotdir + '/performance_metrics/1980_vs_1911_vs_new_comparison/'

    if not os.path.exists(figpath):
        os.makedirs(figpath)
        
    if isinstance(random_seed, int):
        figname = f'1980_vs_1911_vs_new_model_performance_metrics_for_seed{random_seed}.png'
    else:
        figname = f'1980_vs_1911_vs_new_model_performance_metrics_{random_seed}.png'

    plt.savefig(figpath + figname)
    plt.close()
    

def create_variable_importance_barchart(variable_importance, X, model_type, random_seed):
    """
    Plots barchart indicating each variables importances in the model.
    
    Args:
        variable_importance (np.array): importance of each variable
        X: predictor variables
        model_type (str): either "1980" or "1911"
        random_seed (int): random state of the RF model ('average_score' if average of many random seeds)
    """
    # Create a bar graph of variable importances
    ax = plt.figure(figsize=(12, 6))
    ax.figure.subplots_adjust(bottom=0.3)
    x_labels = [variable_to_label[var_name] for var_name in X.columns]
    plt.bar(x_labels, variable_importance)
    plt.xticks(ha='right', rotation=45)
    plt.xlabel('Predictors')
    plt.ylabel('Importance')
    if isinstance(random_seed, int):
        random_seed_title = f'Random Seed {random_seed}'
        figname = f'variable_importance_{model_type}_var_model_seed{random_seed}.png'
    else:
        random_seed_title = f'the Average of Multiple Iterations'
        figname = f'variable_importance_{model_type}_{random_seed}.png'
        
    plt.title(f'Variable Importance in the {model_type} Model for {random_seed_title}')
    figpath = plotdir + f'/variable_importance/{model_type}/'

    if not os.path.exists(figpath):
        os.makedirs(figpath)
    
    plt.savefig(figpath + figname)
    plt.close()


def performance_and_variable_importance_from_n_iterated_RF_model_seeds(X, y, test_size, n_iterations=5):
    """
    Trains the Random Forest with different seeds to assess stability and generalisability of the model.
    Args:
    X: predictor variables data
    y: target variables data
    test_size (float): proportion of data to be held back for testing
    n_iterations (int): number of iterations of model (default=100)
    """
    seeds = np.arange(n_iterations)

    # Initialize arrays to store variable importance
    variable_importance = np.zeros((len(X.columns), n_iterations))

    # Initialize arrays to store performance metrics
    accuracy_scores = np.zeros(n_iterations)
    precision_scores = np.zeros(n_iterations)
    recall_scores = np.zeros(n_iterations)
    f1_scores = np.zeros(n_iterations)
    balanced_accuracy_scores = np.zeros(n_iterations)
    false_alarm_scores = np.zeros(n_iterations)

    # Train the model and calculate performance metrics for each iteration
    for i, seed in enumerate(seeds):
        # Split the data into training and testing sets for each iteration
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=seed)
    
        # Train the Random Forest model with a different seed
        clf = RandomForestClassifier(n_estimators=500, random_state=seed)
        clf.fit(X_train, y_train)

        # Calculate variable importance
        variable_importance[:, i] = clf.feature_importances_
        
        # Predict on test data
        y_pred = clf.predict(X_test)
    
        # Calculate performance metrics
        accuracy_scores[i] = accuracy_score(y_test, y_pred)
        precision_scores[i] = precision_score(y_test, y_pred)
        recall_scores[i] = recall_score(y_test, y_pred)
        f1_scores[i] = f1_score(y_test, y_pred)
        balanced_accuracy_scores[i] = balanced_accuracy_score(y_test, y_pred)
        confusion_matrix = sklearn.metrics.confusion_matrix(y_test, y_pred, normalize='all')
        false_alarm_scores[i] = confusion_matrix[0, 1]
        
    # Create a DataFrame to store the performance metrics for each iteration
    performance_df = pd.DataFrame({
        'Accuracy': accuracy_scores,
        'Precision': precision_scores,
        'Recall': recall_scores,
        'F1-score': f1_scores,
        'Balanced Accuracy': balanced_accuracy_scores,
        'False Alarm': false_alarm_scores
    })

    variable_importance_df = pd.DataFrame(variable_importance.T, columns=X.columns)
    
    return performance_df, variable_importance_df


def find_mean_performance_metrics_and_var_importance(performance_df, variable_importance_df):
    """
    Find the mean of each performance metric.
    Args:
    performance_df (pd.DataFrame): 
        dataframe of the performance metrics from each iteration of the RF model
    """
    mean_performance_metric = performance_df.mean(axis=0)
    mean_performance_df = mean_performance_metric.to_frame().T

    mean_variable_importance = variable_importance_df.mean(axis=0)
    mean_variable_importance_df = mean_variable_importance.to_frame().T
    
    return mean_performance_df, mean_variable_importance_df
    

def create_performance_metrics_boxplot(performance_df, model_type):
    """
    Plots a boxplot of the performance metric scores
    Args:
        performance_df (pd.DataFrame): dataframe of the performance metrics from each iteration of the RF model
        model_type (str): describing the type of RF model either "1980", "1911", or "new"
    """
    # Draw a  box and whisker plot to display the results of each performance metric across 30 iterations
    plt.figure(figsize=(12, 6))
    sns.boxplot(data=performance_df, palette="pastel")
    plt.ylabel('Score')
    plt.title(f'Box and Whisker Plot for Performance Metrics of the {model_type} Model Across 100 Iterations')

    # Save plot
    figpath = plotdir + f'/performance_metrics/{model_type}/'

    if not os.path.exists(figpath):
        os.makedirs(figpath)
    
    figname = f'performance_metric_boxplot_{model_type}_var_model.png'
    plt.savefig(figpath + figname)
    
    plt.close()
    

def create_variable_importance_boxplot(variable_importance_df, model_type):
    """
    Plots a boxplot of the variable importance across n_iterated models.
    Args:
        variable_importance_df (pd.DataFrame): dataframe of the variable importance from each iteration of the RF model
        model_type (str): describing the type of RF model either "1980", or "1911"
    """       
    # Draw a  box and whisker plot to display the results of each performance metric across 30 iterations
    plt.figure(figsize=(12, 6))
    ax = sns.boxplot(data=variable_importance_df, palette="pastel")
    ax.figure.subplots_adjust(bottom=0.3)
    plt.ylabel('Score')
    plt.xticks(ha='right', rotation=45)
    plt.title(f'Box and Whisker Plot for Variable Importance of the {model_type} Model Across 100 Iterations')

    # Save plot
    figpath = plotdir + f'/variable_importance/{model_type}/'

    if not os.path.exists(figpath):
        os.makedirs(figpath)
    
    figname = f'variable_importance_boxplot_{model_type}_var_model.png'
    plt.savefig(figpath + figname)
    
    plt.close()


def combine_all_iterative_functions(data, predictors, target, test_size, model_type, n_iterations=100):
    """
    Trains n_iterations RF models with different random states and creates plots of mean performance metrics and
    variable importance. Also creates boxplots to show the spread of the performance metrics over the iterations.

    Args:
        data (pd.DataFrame): training data
        predictors (list): list of all the predictors for the model
        target (str): target variable for the model
        test_size (float): proportion of training data to hold back for testing
        model_type (str): describing the type of RF model either "1980", "1911", or "new"
        n_iterations (int): number of RF models to train (default=100)
    """
    X = data[predictors]
    y = data[target]

    performance_df, variable_importance_df = performance_and_variable_importance_from_n_iterated_RF_model_seeds(X, y, test_size, n_iterations)
    mean_performance_df, mean_variable_importance_df = find_mean_performance_metrics_and_var_importance(performance_df, variable_importance_df)
    mean_variable_importance = mean_variable_importance_df.values[0]
    print(mean_variable_importance_df, mean_variable_importance)
    
    create_performance_metrics_boxplot(performance_df, model_type)
    # create_variable_importance_boxplot(variable_importance_df, model_type)
    
    create_performance_metric_bar_chart(mean_performance_df, 'average_score', model_type)
    create_variable_importance_barchart(mean_variable_importance, X, model_type, 'average_score')

    return performance_df, mean_performance_df


def main():
    mean_performance_df_dict = {}
    for model in model_types:
        print('Model: ', model)
        predictors = predictors_dict[model]
        performance_df_from_n_seeds, mean_performance_df = combine_all_iterative_functions(training_data, predictors, 'Drought', 0.3, model)

        mean_performance_df_dict[model] = mean_performance_df
    create_bar_chart_to_compare_two_performance_metrics(mean_performance_df_dict['1911'], mean_performance_df_dict['1911_simp'], '1911', '1911_simp', 'average_score')
    # create_bar_chart_1980_vs_1911_vs_new_performance_metrics(
    #     mean_performance_df_dict['1980'], mean_performance_df_dict['1911'], mean_performance_df_dict['new'], 'average_score'
    # )

    # create_bar_chart_1980_vs_1911_performance_metrics(
    #     mean_performance_df_dict['1980'], mean_performance_df_dict['1911'], 'average_score'
    # )


if __name__ == "__main__":
    main()
    