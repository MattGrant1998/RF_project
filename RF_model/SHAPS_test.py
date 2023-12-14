import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

import shap
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
import time

datadir = '/g/data/w97/mg5624/RF_project/'

predictors_data_filepath = {
    '1980': datadir + 'predictors_data/1980_model/predictors_dataframe_1980-2022_SE_australia.csv',
    '1911': datadir + 'predictors_data/1911_model/predictors_dataframe_1911-2022_SE_australia.csv',
    'test': datadir + 'predictors_data/test_model/predictors_dataframe_1981-1983_test.csv',
}


predictors_data = pd.read_csv(predictors_data_filepath['1980'])
print(time.time, ': predictors dataframe loaded')

# print(predictors_data)
training_data = pd.read_csv(datadir + '/training_data/training_data.csv')
training_data.dropna(axis=0, inplace=True)


predictor_names = {
    '1980': [
        'Acc_12-Month_Precipitation', 'Mean_12-Month_Runoff', 'ENSO_index', 'IOD_index', 
        'SAM_index', 'Mean_12-Month_ET', 'Mean_12-Month_PET', 'Mean_12-Month_SMsurf', 'Mean_12-Month_SMroot', 
        'Sin_month', 'Cos_month'
    ],

    '1911': [
        'Acc_12-Month_Precipitation', 'Mean_12-Month_Runoff', 'ENSO_index', 'IOD_index', 'Sin_month', 'Cos_month'
    ],

    'test': [
        'Precipitation', 'Runoff', 'ENSO_index', 'PET', 'Sin_month', 'Cos_month'
    ],
}

# predictors_test = ['Precipitation', 'Runoff', 'ENSO_index', 'PET', 'Sin_month', 'Cos_month']
target = 'Drought'

X_train = training_data[predictor_names['1980']]
y = training_data[target]


# Create and train the Random Forest model
clf = RandomForestClassifier(n_estimators=500, random_state=42)
clf.fit(X_train, y)
print(time.time, ': model trained')

year_of_interest = 2008

predictors_data_at_time = predictors_data.loc[predictors_data['Year'] == 2008]
X_pred = predictors_data_at_time[predictor_names['1980']]

explainer = shap.TreeExplainer(clf)
print(time.time, ': explainer defined')
shap_values = explainer.shap_values(X_pred)
print(time.time, ': shap_values defined')

plt.figure()
shap.summary_plot(shap_values, X_pred)
plt.savefig('/home/561/mg5624/RF_project/shaps_test_2008.png')
plt.close()
print(time.time, ': figure made')




