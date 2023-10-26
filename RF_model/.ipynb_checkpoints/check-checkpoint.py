import pandas as pd

pd.set_option('display.max_rows', None)
print('Full dataframe:\n')
full_df = pd.read_csv('/g/data/w97/mg5624/RF_project/predictors_data/full_model/predictors_dataframe_1980-2022_SE_australia.csv')
print(full_df.loc[full_df['Year_Month']=='1981-02'].loc[full_df['lon']==149.0].loc[full_df['lat']==-32.00])

print('\n\nLong ts dataframe:\n')
long_df = pd.read_csv('/g/data/w97/mg5624/RF_project/predictors_data/long_ts_model/predictors_dataframe_1911-2022_SE_australia.csv')
print(long_df.loc[long_df['Year_Month']=='1981-02'].loc[long_df['lon']==149.0].loc[long_df['lat']==-32.00])