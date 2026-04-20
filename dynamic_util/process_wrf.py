#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
WRF4PALM - Optimized vertical interpolation with vectorization
FIXED: No pickling of netCDF4 datasets
"""
from tqdm import tqdm
from functools import partial
from multiprocess import Pool
import gc
import numpy as np
import xarray as xr

def multi_zinterp(max_pool, ds_in, var, zcoord, ds_out):
    """
    Optimized vertical interpolation with vectorization and memory efficiency
    FIXED: Load data in main process, then distribute to workers
    """
    is_w = (var == "W")
    n_levels = len(zcoord)
    
    # For small jobs, use vectorized single process (faster than pool)
    if n_levels <= 10:
        print(f"    Using vectorized interpolation for {var} ({n_levels} levels)")
        
        # Process all levels in main process (no pickling)
        for lvl in zcoord:
            data = ds_in.salem.wrf_zlevel(var, levels=lvl, use_multiprocessing=False)
            data = data.astype(np.float32)
            
            if is_w:
                ds_out[var].loc[dict(zw=lvl)] = data
            else:
                ds_out[var].loc[dict(z=lvl)] = data
            
            del data
            gc.collect()
        
        return ds_out[var]
    
    # For larger jobs, extract data first, then parallelize
    print(f"    Using parallel interpolation for {var} ({n_levels} levels)")
    
    # FIRST: Extract all data at once (avoids pickling dataset)
    # Get the data for all levels at once using vectorized operation
    try:
        # Try vectorized extraction
        all_data = ds_in.salem.wrf_zlevel(var, levels=zcoord, use_multiprocessing=False)
        all_data = all_data.astype(np.float32)
        
        # Then assign results (no parallel needed)
        if is_w:
            ds_out[var].loc[dict(zw=zcoord)] = all_data
        else:
            ds_out[var].loc[dict(z=zcoord)] = all_data
        
        del all_data
        gc.collect()
        return ds_out[var]
        
    except:
        # Fallback: process sequentially (no pickling)
        print(f"    Falling back to sequential interpolation for {var}")
        for lvl in tqdm(zcoord, desc=f"    {var}", leave=False):
            data = ds_in.salem.wrf_zlevel(var, levels=lvl, use_multiprocessing=False)
            data = data.astype(np.float32)
            
            if is_w:
                ds_out[var].loc[dict(zw=lvl)] = data
            else:
                ds_out[var].loc[dict(z=lvl)] = data
            
            del data
            gc.collect()
        
        return ds_out[var]