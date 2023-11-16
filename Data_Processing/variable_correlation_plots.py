import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Load training data to test on
training_data = pd.read_csv('/g/data/w97/mg5624/RF_project/training_data/training_data.csv')

variables = ['Acc_12-Month_Precipitation', 'Mean_12-Month_Runoff', 'ENSO_index', 'IOD_index', 'SAM_index', 'Mean_12-Month_ET', 
             'Mean_12-Month_PET', 'Mean_12-Month_SMsurf', 'Mean_12-Month_SMroot', 'Sin_month', 'Cos_month']

def create_variable_correlation_plot(dataframe, variables):
    """
    Creates a heatmap of the variable correlations of the variables in the dataframe specified.

    Args:
        dataframe (pd.DataFrame): dataframe inclusive of the varibales wanting to check corelation of
        variables (list of str): list of the varibale names to check correlation of
    """
    training_data_vars_only = dataframe[variables]
    
    corr_df = training_data_vars_only.corr()
    
    plt.figure(figsize=(12, 12))
    sns.heatmap(corr_df, vmin=-1, vmax=1, cmap='RdBu')

    plt.subplots_adjust(left=0.2, right=0.97, bottom=0.15, top=0.95)
    plt.xticks(rotation=45, ha='right')
    filepath = '/g/data/w97/mg5624/plots/RF_project/variable_correlation/new/'

    if not os.path.exists(filepath):
        os.makedirs(filepath)
    filename = 'training_data_variable_correlation.png'

    plt.savefig(filepath + filename)
    plt.close()


def main():
    create_variable_correlation_plot(training_data, variables)


if __name__ == "__main__":
    main()
