def create_predictor_dataframe(start_year, end_year, area, data, data_name):
    """
    Creates a dataframe that runs from start year to end year, to be used to
    predict droughts in that time period. Predictors can then be added to 
    this dataframe using the add_predictor_to_predictors_dataframe function.
    
    Args:
        start_year (int): start year
        end_year (int): end year
        area (str): one of the areas defined in the COORDS dictionary
        data (xr.DataSet or xr.DataArray): 
            data of a predictor containing at least data covering the time period and area of interest
        data_name (str): name of the data
    
    Returns:
        constrained_df (pd.DataFrame): dataframe of data over specified area and time 
    """
    data = data.rename(data_name)
    
    constrained_data = constrain_data(data, area, start_year, end_year) 

    constrained_data = add_year_month_coord_to_dataarray(constrained_data)

    constrained_df = constrained_data.to_dataframe()
    constrained_df.reset_index(inplace=True)

    coord_rename = {
        'lon': 'Longitude',
        'lat': 'Latitude'
    }
    
    constrained_df.rename(columns=coord_rename, inplace=True)
   
    constrained_df['Latitude'] = constrained_df['Latitude'].astype(float).round(2)
    constrained_df['Longitude'] = constrained_df['Longitude'].astype(float).round(2)
    
    return constrained_df


def add_predictor_to_predictors_dataframe(predictors_df, new_predictor_name, area, start_year, end_year, replace=False):
    """
    Adds specified predictor data into the predictors dataframe.

    Args:
        predictors_df (pd.DataFrame): dataframe of the predictors
        new_predictor_name (str): name of the new predictor
        start_year (int): start year
        end_year (int): end year
        area (str): one of the areas defined in the COORDS dictionary
        replace (bool): if true, new predictor will replace any of the same name in dataframe

    Returns:
        merged_df (pd.DataFrame): predictors_df with additional predictor in it
    """
    # Return the original dataframe if predictor_name exists, and we're not replacing it.
    if new_predictor_name in predictors_df and not replace:
        print('Skipped ', new_predictor_name)
        return predictors_df
    
    file = FILES[new_predictor_name]
    
    if file[-3:] == 'csv':
        new_predictor_df = pd.read_csv(file)
        new_predictor_df['Month'] = new_predictor_df['Month'].astype(int).astype(str)
        new_predictor_df["Year_Month"] = (new_predictor_df['Year'].astype(str) + '-'
                                          + new_predictor_df['Month'].str.zfill(2))
    elif file[-2:] == 'nc':
        new_predictor_data = xr.open_dataarray(file)
        new_predictor_data_constrained = constrain_data(new_predictor_data, area, start_year, end_year)
        new_predictor_data_constrained = add_year_month_coord_to_dataarray(new_predictor_data_constrained)
        new_predictor_df = new_predictor_data_constrained.to_dataframe()
        new_predictor_df.reset_index(inplace=True)
    else:
        raise ValueError(f'File type of {new_predictor_name} not supported. '
                         f'Expected .nc or .csv file got file: {file} instead')
        
    new_predictor_df.dropna(axis=0, inplace=True)
    
    if 'lon' in new_predictor_df.columns:
        coord_rename = {
            'lon': 'Longitude',
            'lat': 'Latitude'
        }
        new_predictor_df.rename(columns=coord_rename, inplace=True)    
    elif 'Longitude' in new_predictor_df.columns:
        pass
    else:
        raise ValueError(f'Cannot find lon or Longitude in dataframe columns. Column names are {new_predictor_df.columns}.'
                         f' If other coordinate name used for dataframe, please change to lon or Longitude before proceeding '
                         f'(same with Y coord but to lat or Latitude).')
    
    new_predictor_df['Latitude'] = new_predictor_df['Latitude'].astype(float).round(2)
    new_predictor_df['Longitude'] = new_predictor_df['Longitude'].astype(float).round(2)
    print(new_predictor_df)
    merged_df = pd.merge(predictors_df, new_predictor_df, on='Year_Month', how='inner')
    
    return merged_df