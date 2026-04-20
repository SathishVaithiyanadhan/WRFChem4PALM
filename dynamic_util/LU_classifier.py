#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Street Type Classifier for WRF4PALM
Classifies street types pixel-by-pixel using WRF LU_INDEX data.
"""

import numpy as np


LU_NAMES = {
    1: "Evergreen Needleleaf Forest", 2: "Evergreen Broadleaf Forest",
    3: "Deciduous Needleleaf Forest", 4: "Deciduous Broadleaf Forest",
    5: "Mixed Forests", 6: "Closed Shrublands", 7: "Open Shrublands",
    8: "Woody Savannas", 9: "Savannas", 10: "Grasslands",
    11: "Permanent Wetlands", 12: "Croplands", 13: "Urban and Built-up",
    14: "Cropland/Natural Vegetation Mosaic", 15: "Snow and Ice",
    16: "Barren or Sparsely Vegetated", 17: "Water", 18: "Wooded Tundra",
    19: "Mixed Tundra", 20: "Barren Tundra", 21: "Lake"
}


def get_lu_category_name(lu_value):
    """Get land use category name from LU_INDEX value."""
    return LU_NAMES.get(int(lu_value), f"Unknown ({lu_value})")


def get_street_type_from_lu_index(lu_index_array, force_urban=False):
    """
    Convert WRF LU_INDEX to street type classification.
    
    Parameters:
    -----------
    lu_index_array : numpy.ndarray
        Land use index array from WRF-Chem (integer values)
    force_urban : bool
        If True, force entire domain to 'urban' regardless of LU_INDEX
    
    Returns:
    --------
    numpy.ndarray
        Street type array with values: 'urban' or 'rural'
    """
    if force_urban:
        print("  FORCING urban classification (force_urban=True)")
        return np.full_like(lu_index_array, 'urban', dtype=object)
    
    street_type = np.full_like(lu_index_array, 'rural', dtype=object)
    urban_mask = (lu_index_array == 13)
    street_type[urban_mask] = 'urban'
    
    # Water bodies - treat as rural
    water_mask = np.isin(lu_index_array, [17, 21])
    street_type[water_mask] = 'rural'
    
    return street_type


def print_street_classification_stats(street_type, lu_index=None):
    """Print statistics about the street type classification."""
    total_pixels = street_type.size
    
    if total_pixels == 0:
        print("\nStreet Type Classification: (empty)")
        return
    
    urban_pixels = np.sum(street_type == 'urban')
    rural_pixels = np.sum(street_type == 'rural')
    
    urban_pct = urban_pixels / total_pixels * 100
    rural_pct = rural_pixels / total_pixels * 100
    
    print(f"\nPixel-by-Pixel Street Type Classification:")
    print(f"  Urban pixels: {urban_pixels} ({urban_pct:.1f}%)")
    print(f"  Rural pixels: {rural_pixels} ({rural_pct:.1f}%)")
    
    if lu_index is not None:
        unique_lu = np.unique(lu_index)
        print(f"\nLU_INDEX values present in THIS domain: {unique_lu.tolist()}")
        print("Categories found in THIS domain:")
        for val in unique_lu:
            name = get_lu_category_name(val)
            count = np.sum(lu_index == val)
            pct = count / total_pixels * 100
            print(f"  {int(val):2d} ({name:<35}): {pct:5.1f}% ({count} pixels)")


def classify_street_types_pixel_by_pixel(ds_interp, nx, ny, force_urban=False):
    """
    Classify street types pixel-by-pixel using interpolated LU_INDEX.
    
    Parameters:
    -----------
    ds_interp : xarray.Dataset
        Horizontally interpolated WRF dataset (already cropped to PALM domain!)
    nx, ny : int
        PALM domain dimensions from config
    force_urban : bool
        If True, force entire domain to urban
    
    Returns:
    --------
    dict
        Street type arrays and statistics
    """
    print("\n" + "="*60)
    print("PIXEL-BY-PIXEL STREET TYPE CLASSIFICATION")
    print("="*60)
    print(f"  Domain size: {nx} × {ny} = {nx*ny} pixels")
    print(f"  Force urban: {force_urban}")
    
    if 'LU_INDEX' not in ds_interp.data_vars:
        print("\nERROR: LU_INDEX not found in interpolated dataset!")
        if force_urban:
            print("  FORCING urban classification for entire domain")
            street_type_surface = np.full((ny, nx), 'urban', dtype=object)
        else:
            print("  Falling back to uniform rural classification")
            street_type_surface = np.full((ny, nx), 'rural', dtype=object)
        
        street_type_we = np.full((ny, 2), street_type_surface[0, 0], dtype=object)
        street_type_sn = np.full((2, nx), street_type_surface[0, 0], dtype=object)
        
        return {
            'street_type_surface': street_type_surface,
            'street_type_we': street_type_we,
            'street_type_sn': street_type_sn,
            'lu_index_surface': None,
            'stats': {'urban_pixels': ny*nx if force_urban else 0, 
                     'rural_pixels': 0 if force_urban else ny*nx}
        }
    
    # Get interpolated LU_INDEX - THIS IS FROM YOUR CROPPED DOMAIN!
    lu_index_surface = ds_interp['LU_INDEX'].isel(time=0).values.astype(np.int32)
    
    print(f"\nInterpolated LU_INDEX shape: {lu_index_surface.shape}")
    print(f"This is YOUR domain extracted from WRF!")
    print(f"Unique values in YOUR domain: {np.unique(lu_index_surface)}")
    
    # Classify pixel-by-pixel
    street_type_surface = get_street_type_from_lu_index(lu_index_surface, force_urban)
    
    # Print statistics
    print_street_classification_stats(street_type_surface, lu_index_surface)
    
    # Create boundary street type maps
    street_type_we = np.full((ny, 2), 'rural', dtype=object)
    for i, x_idx in enumerate([0, -1]):
        if x_idx < nx:
            street_type_we[:, i] = street_type_surface[:, x_idx]
    
    street_type_sn = np.full((2, nx), 'rural', dtype=object)
    for i, y_idx in enumerate([0, -1]):
        if y_idx < ny:
            street_type_sn[i, :] = street_type_surface[y_idx, :]
    
    urban_pixels = np.sum(street_type_surface == 'urban')
    rural_pixels = np.sum(street_type_surface == 'rural')
    
    stats = {
        'urban_pixels': int(urban_pixels),
        'rural_pixels': int(rural_pixels),
        'urban_pct': float(urban_pixels / street_type_surface.size * 100),
        'rural_pct': float(rural_pixels / street_type_surface.size * 100),
        'unique_lu': np.unique(lu_index_surface).tolist()
    }
    
    return {
        'street_type_surface': street_type_surface,
        'street_type_we': street_type_we,
        'street_type_sn': street_type_sn,
        'lu_index_surface': lu_index_surface,
        'stats': stats
    }