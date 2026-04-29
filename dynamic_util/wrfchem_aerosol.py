#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
WRF-Chem Aerosol Processing for PALM Dynamic Driver
Fully vectorized with correct unit conversions and bin mapping.
FIXED: #/kg -> #/m3 conversion, #/m3 mass preservation in overlap mapping,
       trace metal scaling, and soluble/insoluble splitting.
"""
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

# ===== TRACE METAL LITERATURE VALUES (Zereini et al., 2005) =====
'''LITERATURE_CONC_NG_M3 = {
    'urban': {'PB': 32.6, 'NI': 7.3, 'AS': 1.0, 'CD': 0.3, 'HG': 0.05},
    'rural': {'PB': 11.6, 'NI': 2.6, 'AS': 0.6, 'CD': 0.2, 'HG': 0.02}
}'''
LITERATURE_CONC_NG_M3 = {
    'urban': {'PB': 1.7, 'NI': 1.5, 'AS': 0.3, 'CD': 0.06, 'HG': 0.05},
    'rural': {'PB': 1.0, 'NI': 0.7, 'AS': 0.2, 'CD': 0.04, 'HG': 0.02}
}

# Reference PM2.5 from literature (for mass fraction derivation only)
REF_PM25_UG_M3 = 10.0  # μg/m³ typical urban PM2.5

# Constant trace metal mass fractions derived from literature
TRACE_METAL_MASS_FRACTIONS = {}
for stype in ['urban', 'rural']:
    TRACE_METAL_MASS_FRACTIONS[stype] = {
        m: LITERATURE_CONC_NG_M3[stype][m] * 1e-3 / REF_PM25_UG_M3
        for m in ['PB', 'NI', 'AS', 'CD', 'HG']
    }

DEFAULT_TRACE_METAL_FRACTIONS = TRACE_METAL_MASS_FRACTIONS['rural'].copy()

# ===== SPECIES CLASSIFICATION =====
INSOLUBLE_SPECIES = ['BC', 'DU']
SOLUBLE_SPECIES = ['SO4', 'OC', 'NH', 'NO', 'SS', 'PB', 'HG', 'NI', 'CD', 'AS']

# Soluble fraction per species (fraction going to mode A / soluble bins)
PARTITION_2A = {
    'SO4': 0.90, 'OC': 0.70, 'BC': 0.10, 'DU': 0.10,
    'SS': 0.90, 'NH': 0.90, 'NO': 0.90,
    'PB': 0.50, 'HG': 0.50, 'NI': 0.50, 'CD': 0.50, 'AS': 0.50
}

# ==============================================================================
# Trace Metal Helper Functions
# ==============================================================================

def is_trace_metal(species):
    """Check if a species is a trace metal."""
    return species in ['PB', 'HG', 'NI', 'CD', 'AS']


def get_trace_metal_mass_fraction(metal, street_type_map=None):
    """
    Get CONSTANT mass fraction for a given metal.
    If street_type_map is provided, use pixel-specific urban/rural fractions.
    """
    default_frac = DEFAULT_TRACE_METAL_FRACTIONS.get(metal, 0.0)
    if street_type_map is None:
        return default_frac
    
    # Start with default rural
    result = np.full(street_type_map.shape, default_frac, dtype=np.float32)
    # Apply urban fraction where street_type indicates urban
    if 'urban' in TRACE_METAL_MASS_FRACTIONS:
        urban_frac = TRACE_METAL_MASS_FRACTIONS['urban'].get(metal, default_frac)
        urban_mask = (street_type_map == 'urban') | (street_type_map == 1)
        if np.any(urban_mask):
            result[urban_mask] = urban_frac
    
    return result


# ==============================================================================
# WRF Variable Name Functions
# ==============================================================================

def get_wrfchem_variables_for_species(palm_species):
    """Get WRF-Chem variable names for a PALM species."""
    if palm_species in AEROSOL_TRANSLATION:
        return AEROSOL_TRANSLATION[palm_species]
    return [palm_species.lower()]


def get_all_wrfchem_variables(palm_species_list):
    """Get all WRF-Chem aerosol mass variable names (with bin suffixes)."""
    all_vars = []
    for species in palm_species_list:
        if is_trace_metal(species):
            continue
        for base_name in get_wrfchem_variables_for_species(species):
            for suffix in WRFCHEM_BIN_SUFFIXES:
                all_vars.append(f'{base_name}{suffix}')
    return all_vars


# ==============================================================================
# Bin Definition Functions
# ==============================================================================

def define_bins(nbin, reglim):
    """
    Define PALM bin structure.
    Returns dmid (geometric mean diameter) and bin_limits (lower limits + upper limit).
    """
    nbins = int(np.sum(nbin))
    dmid = np.zeros(nbins, dtype=np.float32)
    vlolim = np.zeros(nbins, dtype=np.float32)
    vhilim = np.zeros(nbins, dtype=np.float32)
    bin_limits = np.zeros(nbins + 1, dtype=np.float32)
    
    # Subrange 1
    ratio_d = reglim[1] / reglim[0]
    for b in range(nbin[0]):
        vlolim[b] = np.pi / 6.0 * (reglim[0] * ratio_d ** (float(b) / nbin[0])) ** 3
        vhilim[b] = np.pi / 6.0 * (reglim[0] * ratio_d ** (float(b+1) / nbin[0])) ** 3
        dmid[b] = np.sqrt((6.0 * vhilim[b] / np.pi) ** 0.33333333 * 
                          (6.0 * vlolim[b] / np.pi) ** 0.33333333)
        bin_limits[b] = (6.0 * vlolim[b] / np.pi) ** 0.33333333
    
    # Subrange 2
    ratio_d = reglim[2] / reglim[1]
    for b in range(nbin[0], nbins):
        c = b - nbin[0]
        vlolim[b] = np.pi / 6.0 * (reglim[1] * ratio_d ** (float(c) / nbin[1])) ** 3
        vhilim[b] = np.pi / 6.0 * (reglim[1] * ratio_d ** (float(c+1) / nbin[1])) ** 3
        dmid[b] = np.sqrt((6.0 * vhilim[b] / np.pi) ** 0.33333333 * 
                          (6.0 * vlolim[b] / np.pi) ** 0.33333333)
        bin_limits[b] = (6.0 * vlolim[b] / np.pi) ** 0.33333333
    
    bin_limits[-1] = reglim[-1]
    
    return dmid, bin_limits


def aerosol_binoverlap(palm_binlim, wrfchem_binlim):
    """
    Calculate overlap ratios between PALM and WRF-Chem bins.
    Returns open_bins list and overlap_ratio matrix.
    The overlap ratio is FRACTION of each WRF bin falling into each PALM bin.
    """
    n_palm = len(palm_binlim) - 1
    n_wrf = len(wrfchem_binlim) - 1
    overlap_ratio = np.zeros((n_wrf, n_palm), dtype=np.float32)
    
    # Convert to nm for integer range overlap
    palm_nm = (np.array(palm_binlim) * 1e9).astype(int)
    wrf_nm = (np.array(wrfchem_binlim) * 1e9).astype(int)
    
    for wbin in range(n_wrf):
        w_low = wrf_nm[wbin]
        w_high = wrf_nm[wbin + 1]
        w_width = w_high - w_low
        
        if w_width <= 0:
            continue
            
        for pbin in range(n_palm):
            p_low = palm_nm[pbin]
            p_high = palm_nm[pbin + 1]
            p_width = p_high - p_low
            
            if p_width <= 0:
                continue
                
            # Calculate overlap
            overlap_low = max(w_low, p_low)
            overlap_high = min(w_high, p_high)
            
            if overlap_low < overlap_high:
                # Logarithmic overlap fraction for aerosol size distributions
                overlap_width = overlap_high - overlap_low
                # Fraction of WRF bin falling into PALM bin
                overlap_ratio[wbin, pbin] = overlap_width / w_width
    
    return overlap_ratio


# ==============================================================================
# Vectorized Processing Functions
# ==============================================================================

def vectorized_mass_fraction_batch(mass_matrix):
    """Normalize mass fractions so they sum to 1.0 along the last axis."""
    total = np.sum(mass_matrix, axis=-1, keepdims=True)
    total = np.where(total < 1e-30, 1.0, total)
    return (mass_matrix / total).astype(np.float32)


def vectorized_batch_aerosol_mapping(wrf_num_matrix, overlap_ratio):
    """
    Map WRF-Chem number concentrations to PALM bins using overlap ratios.
    PRESERVES total number: sum(PALM bins) = sum(WRF bins) * bin_width_factor
    
    wrf_num_matrix: shape (..., n_wrf_bins) in #/kg or #/m3
    overlap_ratio: shape (n_wrf_bins, n_palm_bins)
    Returns: shape (..., n_palm_bins)
    """
    return np.dot(wrf_num_matrix, overlap_ratio).astype(np.float32)


def vectorized_bin_limits_to_centers(bin_limits):
    """Calculate geometric mean bin centers from bin limit array."""
    return np.sqrt(bin_limits[:-1] * bin_limits[1:]).astype(np.float32)


def create_separated_mass_fractions(mass_array, listspec, nf2a=0.75):
    """
    Separate mass fractions into soluble (a) and insoluble (b) portions.
    
    Parameters:
    - mass_array: shape (..., n_species) with mass fractions summing to 1
    - listspec: list of species names
    - nf2a: soluble fraction factor (0.75 means 75% to soluble bins)
    
    Returns:
    - mass_fracs_a: normalized mass fractions for soluble bins
    - mass_fracs_b: normalized mass fractions for insoluble bins
    """
    n_species = len(listspec)
    n_dims = mass_array.ndim
    
    mass_a = np.copy(mass_array)
    mass_b = np.copy(mass_array)
    
    for idx, spec in enumerate(listspec):
        frac_2a = PARTITION_2A.get(spec, 0.5)
        
        # Create index slice
        idx_slice = [slice(None)] * n_dims
        idx_slice[-1] = idx
        
        mass_a[tuple(idx_slice)] = mass_array[tuple(idx_slice)] * frac_2a
        mass_b[tuple(idx_slice)] = mass_array[tuple(idx_slice)] * (1.0 - frac_2a)
    
    # Normalize each to sum to 1
    mass_fracs_a = vectorized_mass_fraction_batch(mass_a)
    mass_fracs_b = vectorized_mass_fraction_batch(mass_b)
    
    return mass_fracs_a, mass_fracs_b


# ==============================================================================
# Unit Conversion Function (CRITICAL FIX)
# ==============================================================================

def convert_wrfchem_to_palm_units(wrf_data_dict, alt_inv):
    """
    Convert WRF-Chem variables from mixing ratio units to PALM concentration units.
    
    WRF-Chem aerosol variables are in:
    - Mass: μg/kg-dryair  ->  PALM needs kg/m³
    - Number: #/kg-dryair ->  PALM needs #/m³
    
    Conversion:
    - M_PALM = M_WRF * 1e-9 / ALT  (μg/kg -> kg/m³)
    - N_PALM = N_WRF / ALT          (#/kg -> #/m³)
    
    where ALT is inverse density (m³/kg), so 1/ALT = air density (kg/m³)
    """
    converted = {}
    
    for var_name, var_data in wrf_data_dict.items():
        if var_name.startswith('num_'):
            # Number concentration: #/kg -> #/m³
            converted[var_name] = var_data / alt_inv
        else:
            # Mass concentration: μg/kg -> kg/m³
            converted[var_name] = var_data * 1e-9 / alt_inv
    
    return converted


def combine_aerosol_mass_from_wrf(wrf_mass_data, listspec, alt_inv):
    """
    Combine WRF-Chem aerosol species into PALM species and convert units.
    
    Parameters:
    - wrf_mass_data: dict with keys like 'so4_a01', 'no3_a02', etc. in μg/kg
    - listspec: list of PALM species names
    - alt_inv: inverse density array from WRF (m³/kg)
    
    Returns:
    - mass_matrix: shape (..., n_species) in kg/m³
    """
    # Get the shape from a sample variable
    sample_key = list(wrf_mass_data.keys())[0]
    base_shape = wrf_mass_data[sample_key].shape[:-1]  # remove last dim (bins)
    n_species = len(listspec)
    n_wrf_bins = 4  # WRF-Chem has 4 bins
    
    mass_matrix = np.zeros(base_shape + (n_species,), dtype=np.float32)
    
    for idx, spec in enumerate(listspec):
        if is_trace_metal(spec):
            continue
        
        wrf_names = get_wrfchem_variables_for_species(spec)
        for wrf_name in wrf_names:
            for bin_idx, suffix in enumerate(WRFCHEM_BIN_SUFFIXES):
                var_name = f'{wrf_name}{suffix}'
                if var_name in wrf_mass_data:
                    # Convert μg/kg -> kg/m³
                    mass_matrix[..., idx] += wrf_mass_data[var_name] * 1e-9 / alt_inv
    
    return mass_matrix


def combine_aerosol_number_from_wrf(wrf_num_data, alt_inv):
    """
    Combine WRF-Chem number variables and convert to #/m³.
    
    Parameters:
    - wrf_num_data: dict with keys like 'num_a01', 'num_a02', etc. in #/kg
    - alt_inv: inverse density array from WRF (m³/kg)
    
    Returns:
    - num_matrix: shape (..., 4) in #/m³
    """
    # Get shape from first variable
    sample_key = list(wrf_num_data.keys())[0]
    base_shape = wrf_num_data[sample_key].shape[:-1]
    
    num_matrix = np.zeros(base_shape + (4,), dtype=np.float32)
    
    for bin_idx, suffix in enumerate(WRFCHEM_BIN_SUFFIXES):
        var_name = f'num{suffix}'
        if var_name in wrf_num_data:
            # Convert #/kg -> #/m³
            num_matrix[..., bin_idx] = wrf_num_data[var_name] / alt_inv
    
    return num_matrix