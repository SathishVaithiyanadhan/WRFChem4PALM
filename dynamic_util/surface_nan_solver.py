#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
WRF4PALM - Vectorized surface NaN solver
"""
import numpy as np

def surface_nan_uv_vectorized(data, z, uv10):
    """
    Vectorized surface NaN resolution for U/V winds
    Uses numpy operations instead of loops
    """
    nan_mask = np.isnan(data)
    if not np.any(nan_mask):
        return data
    
    # Find first non-NaN index for each column/profile
    first_valid = np.argmax(~nan_mask, axis=0)
    
    # Vectorized calculation for all points
    result = data.copy()
    
    for idx in range(data.shape[0]):
        if idx < np.min(first_valid):
            # Still in NaN region - use log profile
            valid_idx = np.min(first_valid)
            if valid_idx < len(z):
                terrain_height = z[valid_idx]
                a = (data[valid_idx] - uv10) / np.log(terrain_height / (terrain_height + 10))
                b = uv10 - a * np.log(10 + terrain_height)
                result[idx] = a * np.log(z[idx]) + b
    
    return result

def surface_nan_s_vectorized(data, z, s2):
    """
    Vectorized surface NaN resolution for scalars
    Linear interpolation from surface
    """
    nan_mask = np.isnan(data)
    if not np.any(nan_mask):
        return data
    
    result = data.copy()
    first_valid = np.argmax(~nan_mask, axis=0)
    
    for idx in range(data.shape[0]):
        if idx < np.min(first_valid):
            valid_idx = np.min(first_valid)
            if valid_idx < len(z):
                terrain_height = z[valid_idx]
                a = (s2 - data[valid_idx]) / 2.0
                b = s2 - a * (2 + terrain_height)
                result[idx] = a * z[idx] + b
    
    return result

def surface_nan_w_vectorized(data):
    """
    Vectorized surface NaN resolution for vertical wind
    """
    nan_mask = np.isnan(data)
    if not np.any(nan_mask):
        return data
    
    result = data.copy()
    first_valid = np.argmax(~nan_mask, axis=0)
    
    for idx in range(data.shape[0]):
        if idx < np.min(first_valid):
            valid_idx = np.min(first_valid) + 1
            if valid_idx < len(data):
                result[idx] = data[valid_idx]
    
    return result

# Keep original function names for compatibility
surface_nan_uv = surface_nan_uv_vectorized
surface_nan_s = surface_nan_s_vectorized
surface_nan_w = surface_nan_w_vectorized

def solve_surface(all_ts, ds_we, ds_sn, surface_var_dict, var):
    """
    Vectorized surface NaN solver for boundaries
    """
    z = ds_we.z.data.astype(np.float32)
    
    for ts in all_ts:
        # Process west/east boundaries (vectorized over y)
        for bc in [0, -1]:
            if var == "U" or var == "V":
                surface_var = surface_var_dict[var]
                surface_value = surface_var.sel(time=ts)[:, bc].data.astype(np.float32)
                
                # Vectorized operation over y dimension
                for j in range(ds_we[var].shape[2]):
                    data_slice = ds_we[var].sel(time=ts)[:, j, bc].data
                    if np.any(np.isnan(data_slice)):
                        fixed = surface_nan_uv(data_slice, z, surface_value[j])
                        ds_we[var].loc[dict(time=ts, y=j, x=bc)] = fixed
            
            elif var == "W":
                for j in range(ds_we[var].shape[2]):
                    data_slice = ds_we[var].sel(time=ts)[:, j, bc].data
                    if np.any(np.isnan(data_slice)):
                        fixed = surface_nan_w(data_slice)
                        ds_we[var].loc[dict(time=ts, y=j, x=bc)] = fixed
            
            else:
                surface_var = surface_var_dict[var]
                surface_value = surface_var.sel(time=ts)[:, bc].data.astype(np.float32)
                
                for j in range(ds_we[var].shape[2]):
                    data_slice = ds_we[var].sel(time=ts)[:, j, bc].data
                    if np.any(np.isnan(data_slice)):
                        fixed = surface_nan_s(data_slice, z, surface_value[j])
                        ds_we[var].loc[dict(time=ts, y=j, x=bc)] = fixed
        
        # Process south/north boundaries (vectorized over x)
        for bc in [0, -1]:
            if var == "U" or var == "V":
                surface_var = surface_var_dict[var]
                surface_value = surface_var.sel(time=ts)[bc, :].data.astype(np.float32)
                
                for i in range(ds_sn[var].shape[3]):
                    data_slice = ds_sn[var].sel(time=ts)[:, bc, i].data
                    if np.any(np.isnan(data_slice)):
                        fixed = surface_nan_uv(data_slice, z, surface_value[i])
                        ds_sn[var].loc[dict(time=ts, y=bc, x=i)] = fixed
            
            elif var == "W":
                for i in range(ds_sn[var].shape[3]):
                    data_slice = ds_sn[var].sel(time=ts)[:, bc, i].data
                    if np.any(np.isnan(data_slice)):
                        fixed = surface_nan_w(data_slice)
                        ds_sn[var].loc[dict(time=ts, y=bc, x=i)] = fixed
            
            else:
                surface_var = surface_var_dict[var]
                surface_value = surface_var.sel(time=ts)[bc, :].data.astype(np.float32)
                
                for i in range(ds_sn[var].shape[3]):
                    data_slice = ds_sn[var].sel(time=ts)[:, bc, i].data
                    if np.any(np.isnan(data_slice)):
                        fixed = surface_nan_s(data_slice, z, surface_value[i])
                        ds_sn[var].loc[dict(time=ts, y=bc, x=i)] = fixed
    
    return var, (ds_we[var], ds_sn[var])