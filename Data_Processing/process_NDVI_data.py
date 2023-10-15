import xarray as xr
 
NDVI_data = xr.open_mfdataset(
    '/g/data/w97/mg5624/RF_project/NDVI/orders/98eac55928c915bdb7c8ff216de8d980/Global_Veg_Greenness_GIMMS_3G/data/*'
)
NDVI = NDVI_data.ndvi

nan_values_upper = NDVI.values > 1
nan_values_lower = NDVI.values < -0.3

print(nan_values_upper)
print(nan_values_lower)
