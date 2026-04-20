#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#------------------------------------------------------------------------------#
# Script for processing of WRF-CHEM files to PALM dynamic driver.
# FULLY VECTORIZED VERSION - Optimized for memory efficiency and speed
# EXTENDED: Trace metals (Pb, Hg, Ni, Cd, As) with dynamic PM2.5 scaling
#------------------------------------------------------------------------------#
import numpy as np

# ===== AEROSOL SPECIES TRANSLATION =====
AEROSOL_TRANSLATION = {
    "SO4": ["so4"],
    "NO": ["no3"],
    "NH": ["nh4"],
    "BC": ["bc"],
    "OC": ["oc", "asoaX", "asoa1", "asoa2", "asoa3", "asoa4", 
          "bsoaX", "bsoa1", "bsoa2", "bsoa3", "bsoa4"],
    "SS": ["cl", "na"],
    "DU": ["co3", "ca", "oin"],
    "PB": ["__TRACE_PB__"],
    "HG": ["__TRACE_HG__"],
    "NI": ["__TRACE_NI__"],
    "CD": ["__TRACE_CD__"],
    "AS": ["__TRACE_AS__"]
}

WRFCHEM_BIN_SUFFIXES = ['_a01', '_a02', '_a03', '_a04']

# ===== TRACE METAL LITERATURE VALUES =====
# From Zereini et al. (2005) - Frankfurt am Main, Germany
# Concentrations in ng/m³ measured at ~15 μg/m³ PM2.5

LITERATURE_CONC_NG_M3 = {
    'urban': {'PB': 32.6, 'NI': 7.3, 'AS': 1.0, 'CD': 0.3, 'HG': 0.05},
    'rural': {'PB': 11.6, 'NI': 2.6, 'AS': 0.6, 'CD': 0.2, 'HG': 0.02}
}

# Reference PM2.5 from literature (used ONLY to derive mass fractions)
REF_PM25_NG_M3 = 6000.0  # 6 μg/m³ = 6,000 ng/m³

# Mass fractions - CONSTANT
TRACE_METAL_MASS_FRACTIONS = {
    'urban': {m: LITERATURE_CONC_NG_M3['urban'][m] / REF_PM25_NG_M3 
              for m in ['PB', 'NI', 'AS', 'CD', 'HG']},
    'rural': {m: LITERATURE_CONC_NG_M3['rural'][m] / REF_PM25_NG_M3 
              for m in ['PB', 'NI', 'AS', 'CD', 'HG']}
}

DEFAULT_TRACE_METAL_FRACTIONS = TRACE_METAL_MASS_FRACTIONS['rural'].copy()

# Direct concentrations 
TRACE_METAL_CONCENTRATIONS = LITERATURE_CONC_NG_M3.copy()
DEFAULT_TRACE_METAL_CONCENTRATIONS = LITERATURE_CONC_NG_M3['rural'].copy()

# ===== SPECIES CLASSIFICATION =====
INSOLUBLE_SPECIES = ['BC', 'DU']
SOLUBLE_SPECIES = ['SO4', 'OC', 'NH', 'NO', 'SS', 'PB', 'HG', 'NI', 'CD', 'AS']


#===============================================================================
# Trace Metal Functions
#===============================================================================

def is_trace_metal(species):
    """Check if a species is a trace metal."""
    return species in ['PB', 'HG', 'NI', 'CD', 'AS']


def get_trace_metal_mass_fraction(metal, street_type=None):
    """Get CONSTANT mass fraction for a given metal and street type."""
    if street_type is None:
        return DEFAULT_TRACE_METAL_FRACTIONS.get(metal, 0.0)
    
    if isinstance(street_type, str):
        return TRACE_METAL_MASS_FRACTIONS.get(street_type, DEFAULT_TRACE_METAL_FRACTIONS).get(metal, 0.0)
    
    result = np.full_like(street_type, DEFAULT_TRACE_METAL_FRACTIONS.get(metal, 0.0), dtype=np.float32)
    for stype, fracs in TRACE_METAL_MASS_FRACTIONS.items():
        mask = (street_type == stype)
        if np.any(mask):
            result[mask] = fracs.get(metal, 0.0)
    return result


def get_trace_metal_concentration(metal, street_type=None):
    """Get trace metal concentration in ng/m³ (constant literature values)."""
    if street_type is None:
        return DEFAULT_TRACE_METAL_CONCENTRATIONS.get(metal, 0.0)
    
    if isinstance(street_type, str):
        return TRACE_METAL_CONCENTRATIONS.get(street_type, DEFAULT_TRACE_METAL_CONCENTRATIONS).get(metal, 0.0)
    
    result = np.full_like(street_type, DEFAULT_TRACE_METAL_CONCENTRATIONS.get(metal, 0.0), dtype=np.float32)
    for stype, concs in TRACE_METAL_CONCENTRATIONS.items():
        mask = (street_type == stype)
        if np.any(mask):
            result[mask] = concs.get(metal, 0.0)
    return result


def calculate_trace_metal_from_pm25(pm25_ug_m3, metal, street_type=None):
    """
    Calculate trace metal dynamically from PM2.5.
    trace_metal (μg/m³) = PM2.5 (μg/m³) × mass_fraction
    """
    fraction = get_trace_metal_mass_fraction(metal, street_type)
    return pm25_ug_m3 * fraction


def create_trace_metal_array(pm25_data, metal, street_type_map):
    """Create trace metal array from PM2.5 data."""
    if street_type_map is None:
        fraction = DEFAULT_TRACE_METAL_FRACTIONS.get(metal, 0.0)
        return pm25_data * fraction
    
    if pm25_data.ndim == 1:
        fraction = np.mean(get_trace_metal_mass_fraction(metal, street_type_map))
        return pm25_data * fraction
    elif pm25_data.ndim == 2:
        fraction = get_trace_metal_mass_fraction(metal, street_type_map)
        return pm25_data * fraction
    elif pm25_data.ndim == 3:
        fraction = get_trace_metal_mass_fraction(metal, street_type_map)
        return pm25_data * fraction[np.newaxis, :, :]
    elif pm25_data.ndim == 4:
        fraction = get_trace_metal_mass_fraction(metal, street_type_map)
        return pm25_data * fraction[np.newaxis, np.newaxis, :, :]
    else:
        fraction = DEFAULT_TRACE_METAL_FRACTIONS.get(metal, 0.0)
        return pm25_data * fraction


def create_trace_metal_concentration_array(metal, street_type_map, vertical_levels=1):
    """
    Compatibility function - creates concentration array from street type only.
    Returns constant values (no PM2.5 scaling).
    """
    conc_ng_m3 = get_trace_metal_concentration(metal, street_type_map)
    conc_ug_m3 = conc_ng_m3 / 1000.0
    
    if vertical_levels > 1:
        if isinstance(conc_ug_m3, np.ndarray):
            if conc_ug_m3.ndim == 2:
                return np.tile(conc_ug_m3[np.newaxis, :, :], (vertical_levels, 1, 1))
            else:
                return np.full((vertical_levels, 1, 1), conc_ug_m3, dtype=np.float32)
        else:
            return np.full((vertical_levels, 1, 1), conc_ug_m3, dtype=np.float32)
    return conc_ug_m3


#===============================================================================
# Basic Translation Functions
#===============================================================================

def get_wrfchem_variables_for_species(palm_species):
    if palm_species in AEROSOL_TRANSLATION:
        return AEROSOL_TRANSLATION[palm_species]
    return [palm_species.lower()]


def get_all_wrfchem_variables(palm_species_list):
    all_vars = []
    for species in palm_species_list:
        if is_trace_metal(species):
            continue
        for base_name in get_wrfchem_variables_for_species(species):
            for suffix in WRFCHEM_BIN_SUFFIXES:
                all_vars.append(f'{base_name}{suffix}')
    return all_vars


#===============================================================================
# Bin Definition Functions
#===============================================================================

def define_bins(nbin, reglim):
    nbins = np.sum(nbin)
    vlolim = np.zeros(nbins, dtype=np.float32)
    vhilim = np.zeros(nbins, dtype=np.float32)
    dmid   = np.zeros(nbins, dtype=np.float32)
    bin_limits = np.zeros(nbins, dtype=np.float32)
    
    ratio_d = reglim[1] / reglim[0]
    for b in range(nbin[0]):
        vlolim[b] = np.pi / 6.0 * (reglim[0] * ratio_d ** (float(b) / nbin[0])) ** 3
        vhilim[b] = np.pi / 6.0 * (reglim[0] * ratio_d ** (float(b+1) / nbin[0])) ** 3
        dmid[b] = np.sqrt((6.0 * vhilim[b] / np.pi) ** 0.33333333 * (6.0 * vlolim[b] / np.pi) ** 0.33333333)
    
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
    x1, x2 = range1.start, range1.stop
    y1, y2 = range2.start, range2.stop
    return x1 <= y2 and y1 <= x2


def aerosol_binoverlap(palm_binlim, wrfchem_binlim):
    overlap_ratio = np.zeros((len(palm_binlim)-1, len(wrfchem_binlim)-1), dtype=np.float32)
    aerobin_open = []
    
    for pbin in range(len(palm_binlim)-1):
        palm_range = range(int(palm_binlim[pbin] * 1e+9), int(palm_binlim[pbin+1] * 1e+9))
        for wbin in range(len(wrfchem_binlim)-1):
            wrfchem_range = range(int(wrfchem_binlim[wbin] * 1e+9) + 1, int(wrfchem_binlim[wbin+1] * 1e+9))
            if range_overlap(palm_range, wrfchem_range):
                aerobin_open.append('_a0' + str(wbin+1))
                overlap = len(set(palm_range) & set(wrfchem_range))
                overlap_ratio[pbin, wbin] = overlap / len(wrfchem_range)
    
    return aerobin_open, overlap_ratio


#===============================================================================
# Vectorized Functions
#===============================================================================

def vectorized_aerosol_mapping(wrf_num, overlap_ratio):
    return np.dot(overlap_ratio, wrf_num).astype(np.float32)


def vectorized_mass_fraction_batch(mass_matrix):
    total = np.sum(mass_matrix, axis=-1, keepdims=True)
    total = np.where(total == 0, 1.0, total)
    return (mass_matrix / total).astype(np.float32)


def vectorized_batch_aerosol_mapping(wrf_num_matrix, overlap_ratio):
    return np.dot(wrf_num_matrix, overlap_ratio.T).astype(np.float32)


def vectorized_bin_limits_to_centers(bin_limits):
    return np.sqrt(bin_limits[:-1] * bin_limits[1:]).astype(np.float32)


def map_wrfchem_to_palm_bins(palm_binlim, wrfchem_binlim, method='overlap'):
    nbins_palm = len(palm_binlim) - 1
    nbins_wrf = len(wrfchem_binlim) - 1
    mapping = np.zeros((nbins_palm, nbins_wrf), dtype=np.float32)
    
    if method == 'simplified':
        mapping = np.ones((nbins_palm, nbins_wrf), dtype=np.float32) / nbins_wrf
    elif method == 'overlap':
        for pbin in range(nbins_palm):
            p_low, p_high = palm_binlim[pbin], palm_binlim[pbin+1]
            total = 0
            for wbin in range(nbins_wrf):
                w_low, w_high = wrfchem_binlim[wbin], wrfchem_binlim[wbin+1]
                overlap_low, overlap_high = max(p_low, w_low), min(p_high, w_high)
                if overlap_low < overlap_high:
                    weight = np.log(overlap_high / overlap_low) / np.log(w_high / w_low)
                    mapping[pbin, wbin] = weight
                    total += weight
            if total > 0:
                mapping[pbin, :] /= total
            else:
                mapping[pbin, :] = 1.0 / nbins_wrf
    return mapping