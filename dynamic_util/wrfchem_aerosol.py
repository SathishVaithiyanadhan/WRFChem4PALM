#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#------------------------------------------------------------------------------#
# Script for processing of WRF-CHEM files to PALM dynamic driver.
#------------------------------------------------------------------------------#
import numpy as np

# Define translation table as a dictionary for easy access
AEROSOL_TRANSLATION = {
    "SO4": ["so4"],
    "NO": ["no3"],
    "NH": ["nh4"],
    "BC": ["bc"],
    "OC": ["oc", "asoaX", "asoa1", "asoa2", "asoa3", "asoa4", 
          "bsoaX", "bsoa1", "bsoa2", "bsoa3", "bsoa4"],
    "SS": ["cl", "na"],
    "DU": ["co3", "ca", "oin"]
}

# WRF-Chem bin suffixes (4 bins in standard WRF-Chem aerosol scheme)
WRFCHEM_BIN_SUFFIXES = ['_a01', '_a02', '_a03', '_a04']

def get_wrfchem_variables_for_species(palm_species):
    """
    Get list of actual WRF-Chem variable names for a given PALM aerosol species.
    
    Parameters:
    -----------
    palm_species : str
        PALM aerosol species name (e.g., 'SO4', 'OC', 'BC', 'SS', 'NH', 'NO', 'DU')
    
    Returns:
    --------
    list
        List of WRF-Chem variable base names (without bin suffix)
        Returns empty list if species not found
    """
    if palm_species in AEROSOL_TRANSLATION:
        return AEROSOL_TRANSLATION[palm_species]
    else:
        # Fallback: use lowercase species name
        return [palm_species.lower()]

def get_all_wrfchem_variables(palm_species_list):
    """
    Get all WRF-Chem variable names (including bin suffixes) for a list of PALM species.
    
    Parameters:
    -----------
    palm_species_list : list
        List of PALM aerosol species names
    
    Returns:
    --------
    list
        List of all WRF-Chem variable names with bin suffixes
    """
    all_vars = []
    for species in palm_species_list:
        base_names = get_wrfchem_variables_for_species(species)
        for base_name in base_names:
            for suffix in WRFCHEM_BIN_SUFFIXES:
                all_vars.append(f'{base_name}{suffix}')
    return all_vars

def translate_aerosol_species(name):
    """
    Translate aerosol species names from PALM format to WRF-Chem format.
    Returns comma-separated string of WRF-Chem variable names.
    """
    if name in AEROSOL_TRANSLATION:
        return ','.join(AEROSOL_TRANSLATION[name])
    else:
        return None

def define_bins(nbin, reglim):
    """
    Define aerosol bins based on nbin and reglim
    Returns:
        dmid: geometric mean diameters for each bin
        bin_limits: bin boundary diameters
    """
    nbins = np.sum(nbin)  # total number of bins
    vlolim = np.zeros(nbins)
    vhilim = np.zeros(nbins)
    dmid   = np.zeros(nbins)
    bin_limits = np.zeros(nbins)
    
    # Sectional bin limits for first subrange
    ratio_d = reglim[1] / reglim[0]
    for b in range(nbin[0]):
        vlolim[b] = np.pi / 6.0 * (reglim[0] * ratio_d ** (float(b) / nbin[0])) ** 3
        vhilim[b] = np.pi / 6.0 * (reglim[0] * ratio_d ** (float(b+1) / nbin[0])) ** 3
        dmid[b] = np.sqrt((6.0 * vhilim[b] / np.pi) ** 0.33333333 * (6.0 * vlolim[b] / np.pi) ** 0.33333333)
    
    # Sectional bin limits for second subrange
    ratio_d = reglim[2] / reglim[1]
    for b in np.arange(nbin[0], np.sum(nbin), 1):
        c = b - nbin[0]
        vlolim[b] = np.pi / 6.0 * (reglim[1] * ratio_d ** (float(c) / nbin[1])) ** 3
        vhilim[b] = np.pi / 6.0 * (reglim[1] * ratio_d ** (float(c+1) / nbin[1])) ** 3
        dmid[b] = np.sqrt((6.0 * vhilim[b] / np.pi) ** 0.33333333 * (6.0 * vlolim[b] / np.pi) ** 0.33333333)
    
    bin_limits = (6.0 * vlolim / np.pi) ** 0.33333333
    bin_limits = np.append(bin_limits, reglim[-1])
    
    return dmid, bin_limits

def range_overlap(range1, range2):
    """Check if two ranges overlap"""
    x1, x2 = range1.start, range1.stop
    y1, y2 = range2.start, range2.stop
    return x1 <= y2 and y1 <= x2

def aerosol_binoverlap(palm_binlim, wrfchem_binlim):
    """
    Calculate overlap between PALM aerosol bins and WRF-Chem bins
    Returns:
        aerobin_open: list of open bin names
        overlap_ratio: overlap ratio matrix
    """
    overlap_ratio = np.zeros((len(palm_binlim)-1, len(wrfchem_binlim)-1))
    aerobin_open = []
    
    for pbin in range(0, len(palm_binlim)-1):
        palm_range = range(int(palm_binlim[pbin] * 1e+9), int(palm_binlim[pbin+1] * 1e+9))
        for wbin in range(0, len(wrfchem_binlim)-1):
            wrfchem_range = range(int(wrfchem_binlim[wbin] * 1e+9) + 1, int(wrfchem_binlim[wbin+1] * 1e+9))
            
            if range_overlap(palm_range, wrfchem_range):
                aerobin_open.append('_a0' + str(wbin+1))
                overlap = len(set(palm_range) & set(wrfchem_range))
                overlap_ratio[pbin, wbin] = overlap / len(wrfchem_range)
    
    return aerobin_open, overlap_ratio

def upwind_location(zlev, u, v):
    """
    Determine upwind location for initial aerosol profiles
    Based on wind direction
    """
    u_wnd = u[0, zlev, :, :]
    v_wnd = v[0, zlev, :, :]
    
    wnd_dir = np.mod(180 + np.rad2deg(np.arctan2(u_wnd, v_wnd)), 360)
    wnd_avg = np.mean(wnd_dir)
    
    if 0 < wnd_avg <= 45 or 315 < wnd_avg <= 360:
        prf_y = 0
        prf_x = round(wnd_dir.shape[1] / 2)
    elif 45 < wnd_avg <= 135:
        prf_y = round(wnd_dir.shape[0] / 2)
        prf_x = round(wnd_dir.shape[1] - 1)
    elif 135 < wnd_avg <= 225:
        prf_y = wnd_dir.shape[0] - 1
        prf_x = round(wnd_dir.shape[1] / 2)
    else:
        prf_y = round(wnd_dir.shape[0] / 2)
        prf_x = 0
    
    return prf_x, prf_y