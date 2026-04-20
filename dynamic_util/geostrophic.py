#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
WRF4PALM - Vectorized geostrophic wind calculations
"""
import numpy as np

def coriolis(lat):  
    """Compute the Coriolis parameter - vectorized"""
    omega = 7.2921159e-05  # angular velocity of the Earth [rad/s]
    return (2 * omega * np.sin(np.radians(lat)))

def rho(T, p):
    """Calculate air density - vectorized"""
    Rd = 287.0
    return p / (Rd * T)

def calc_geostrophic_wind_plevels(array_2d_press, array_2d_temp, array_1d_lat, array_1d_lon, dy, dx):
    """
    Calculate Geostrophic wind profile - FULLY VECTORIZED
    No loops over i,j - uses numpy vectorization
    """
    # Vectorized Coriolis
    f_lat = coriolis(array_1d_lat)
    fx = np.nanmean(f_lat) * np.mean(dx)
    fy = np.nanmean(f_lat) * np.mean(dy)
    
    # Vectorized density calculation
    rho_tmp = np.nanmean(rho(array_2d_temp, array_2d_press))
    
    # Vectorized pressure gradients (no loops!)
    gradx = np.diff(array_2d_press, axis=1)  # Gradient along longitude
    grady = np.diff(array_2d_press, axis=0)  # Gradient along latitude
    
    # Vectorized midpoint interpolation
    gradx_mid = (gradx[:, :-1] + gradx[:, 1:]) / 2.0 if gradx.shape[1] > 1 else gradx
    grady_mid = (grady[:-1, :] + grady[1:, :]) / 2.0 if grady.shape[0] > 1 else grady
    
    # Vectorized wind calculations
    ug_tmp = np.nanmean((-1 / (rho_tmp * fy)) * grady_mid)
    vg_tmp = np.nanmean((1 / (rho_tmp * fx)) * gradx_mid)
    
    return np.array([ug_tmp, vg_tmp], dtype=np.float32)

def calc_geostrophic_wind_zlevels(gph, latitude, dy, dx):
    """
    Use geopotential height to calculate geostrophic wind - VECTORIZED
    """
    # Vectorized Coriolis
    f = np.nanmean(coriolis(latitude))
    
    # Vectorized gradients (no loops!)
    grady = np.diff(gph, axis=0)  # Gradient along latitude
    gradx = -np.diff(gph, axis=1)  # Gradient along longitude (negative sign)
    
    # Vectorized wind calculations
    ug = -np.nanmean(grady / dy * 9.8 / f)
    vg = np.nanmean(gradx / dx * 9.8 / f)
    
    return (ug.astype(np.float32), vg.astype(np.float32))