#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
WRF4PALM - COMPLETE VECTORIZED VERSION WITH UNIT CONVERSION FIXES
Fully optimized for memory efficiency and speed with aerosol support
EXTENDED: Trace metals (Pb, Hg, Ni, Cd, As) with LU-INDEX-based classification
FIXED: #/kg -> #/m3 conversion, mass preservation in bin mapping, soluble/insoluble split
FIXED: Scalar extraction from xarray .values for air density calculation
"""

import sys
import os
import time
import gc
import warnings
import numpy as np
from datetime import datetime, timedelta
from functools import partial
from glob import glob
from math import floor, ceil

# ===== MEMORY OPTIMIZATION: Set float32 as default =====
np.float_ = np.float32
np.set_printoptions(precision=4)

# ===== PREVENT SALEM DOWNLOADS =====
os.environ['SALEM_DOWNLOAD_DEMO_FILES'] = 'False'
os.environ['SALEM_OFFLINE_MODE'] = 'True'

# ===== IMPORTS =====
import salem
import xarray as xr
import configparser
import ast
from tqdm import tqdm
from multiprocess import Pool
from pyproj import Proj, Transformer

# Import from utility files
from dynamic_util.nearest import framing_2d_cartesian
from dynamic_util.loc_dom import calc_stretch, domain_location, generate_cfg
from dynamic_util.process_wrf import multi_zinterp
from dynamic_util.geostrophic import calc_geostrophic_wind_zlevels, calc_geostrophic_wind_plevels
from dynamic_util.surface_nan_solver import solve_surface, surface_nan_uv, surface_nan_s, surface_nan_w
from dynamic_util.interp_array import interp_array_2d, interp_array_1d

# Import from wrfchem_aerosol (with vectorized functions)
from dynamic_util.wrfchem_aerosol import *
from dynamic_util.LU_classifier import *

# Suppress warnings
warnings.filterwarnings("ignore", '.*pyproj.*')
warnings.simplefilter(action='ignore', category=FutureWarning)


def setup_traffic_variables(chem_species):
    """Check if traffic variables are requested"""
    traffic_mapping = {}
    has_traffic = False
    
    for species in chem_species:
        if species.endswith('_tra'):
            base_species = species.replace('_tra', '')
            if base_species in ['no', 'no2', 'PM10', 'PM2_5_DRY']:
                traffic_mapping[base_species] = species
                has_traffic = True
                print(f"Traffic variable requested: {species}")
    
    return has_traffic, traffic_mapping


def extract_scalar_from_xarray(xr_values):
    """Safely extract a scalar float from xarray .values output."""
    # Convert to numpy array and flatten
    arr = np.asarray(xr_values).flatten()
    if len(arr) == 0:
        return 1.0  # Fallback
    return float(np.mean(arr))


def get_air_density(alt_data, varname="ALT", clip_min=0.3, clip_max=2.0):
    """
    Calculate air density from WRF ALT (inverse density) variable.
    ALT is in m³/kg, so air density = 1/ALT in kg/m³.
    Clips ALT to prevent unphysical densities.
    """
    alt_val = extract_scalar_from_xarray(alt_data)
    alt_clipped = min(max(alt_val, clip_min), clip_max)
    air_density = 1.0 / alt_clipped
    
    if alt_val != alt_clipped:
        print(f"    Note: {varname} clipped from {alt_val:.4f} to {alt_clipped:.4f} m³/kg")
    
    return air_density


#===============================================================================
# Main Execution
#===============================================================================

start = datetime.now()

# Create directories
for dir_name in ["./cfg_files", "./dynamic_files"]:
    if not os.path.exists(dir_name):
        print(f"{dir_name} folder created")
        os.makedirs(dir_name)

# Read configuration
config = configparser.RawConfigParser()
config.read(sys.argv[1])

case_name = ast.literal_eval(config.get("case", "case_name"))[0]
max_pool = ast.literal_eval(config.get("case", "max_pool"))[0]
geostr_lvl = ast.literal_eval(config.get("case", "geostrophic"))[0]

# Chemistry species
chem_species_raw = ast.literal_eval(config.get("chemistry", "species"))
print(f"Raw chemistry species: {chem_species_raw}")

if isinstance(chem_species_raw, tuple):
    if len(chem_species_raw) == 1 and isinstance(chem_species_raw[0], list):
        chem_species = chem_species_raw[0]
    else:
        chem_species = list(chem_species_raw)
elif isinstance(chem_species_raw, list):
    chem_species = chem_species_raw
else:
    chem_species = [chem_species_raw]

print(f"Final chemistry species: {chem_species}")

# Traffic variables
has_traffic_vars, traffic_mapping = setup_traffic_variables(chem_species)
original_chem_species = chem_species.copy()
chem_species_for_processing = [s for s in chem_species if not s.endswith('_tra')]

print(f"Chemistry species for processing: {chem_species_for_processing}")

# Aerosol settings
aerosol_wrfchem = False
listspec = []
nbin = [1, 7]
reglim = [3.0e-9, 1.0e-8, 2.5e-6]
wrfchem_bin_limits = [3.9e-8, 1.56e-7, 6.25e-7, 2.5e-6, 1.0e-5]
nf2a = 1.0

try:
    aerosol_wrfchem_raw = ast.literal_eval(config.get("aerosol", "aerosol_wrfchem"))
    if isinstance(aerosol_wrfchem_raw, (list, tuple)):
        aerosol_wrfchem = aerosol_wrfchem_raw[0]
    else:
        aerosol_wrfchem = aerosol_wrfchem_raw
    print(f"Aerosol processing: {aerosol_wrfchem}")
except:
    print("Aerosol settings not found, disabling aerosol processing")

if aerosol_wrfchem:
    try:
        listspec_raw = ast.literal_eval(config.get("aerosol", "listspec"))
        if isinstance(listspec_raw, (list, tuple)):
            if len(listspec_raw) > 0 and isinstance(listspec_raw[0], (list, tuple)):
                listspec = list(listspec_raw[0])
            else:
                listspec = list(listspec_raw)
        else:
            listspec = [listspec_raw]
        print(f"Aerosol composition list: {listspec}")
    except:
        listspec = ['SO4', 'OC', 'BC', 'SS', 'NH', 'NO', 'DU']
    
    try:
        nbin_raw = ast.literal_eval(config.get("aerosol", "nbin"))
        if isinstance(nbin_raw, (list, tuple)):
            if len(nbin_raw) > 0 and isinstance(nbin_raw[0], (list, tuple)):
                nbin = list(nbin_raw[0])
            else:
                nbin = list(nbin_raw)
        else:
            nbin = [nbin_raw]
        print(f"Aerosol bins: {nbin}")
    except:
        nbin = [1, 7]
    
    try:
        reglim_raw = ast.literal_eval(config.get("aerosol", "reglim"))
        if isinstance(reglim_raw, (list, tuple)):
            if len(reglim_raw) > 0 and isinstance(reglim_raw[0], (list, tuple)):
                reglim = list(reglim_raw[0])
            else:
                reglim = list(reglim_raw)
        else:
            reglim = [reglim_raw]
        print(f"Aerosol bin limits: {reglim}")
    except:
        pass
    
    try:
        wrfchem_bin_limits_raw = ast.literal_eval(config.get("aerosol", "wrfchem_bin_limits"))
        if isinstance(wrfchem_bin_limits_raw, (list, tuple)):
            if len(wrfchem_bin_limits_raw) > 0 and isinstance(wrfchem_bin_limits_raw[0], (list, tuple)):
                wrfchem_bin_limits = list(wrfchem_bin_limits_raw[0])
            else:
                wrfchem_bin_limits = list(wrfchem_bin_limits_raw)
        print(f"WRF-Chem bin limits: {wrfchem_bin_limits}")
    except:
        pass
    
    try:
        nf2a_raw = ast.literal_eval(config.get("aerosol", "nf2a"))
        if isinstance(nf2a_raw, (list, tuple)):
            nf2a = nf2a_raw[0]
        else:
            nf2a = nf2a_raw
        print(f"nf2a factor: {nf2a}")
    except:
        pass

# Radiation settings
try:
    radiation_from_wrf = ast.literal_eval(config.get("radiation", "radiation_from_wrf"))[0]
except:
    radiation_from_wrf = True

try:
    radiation_smoothing_distance = ast.literal_eval(config.get("radiation", "radiation_smoothing_distance"))[0]
except:
    radiation_smoothing_distance = 10000.0

print(f"Radiation from WRF: {radiation_from_wrf}")

# Component species for aggregation
RH_components = ["isopr", "apin", "bpin", "limon", "bcary", "myrc", 
                "benzene", "tol", "xylenes", "bigalk", "bigene", "c2h4", "c3h6"]

RO2_components = ["ch3o2", "aco3", "mco3", "alko2", "aceto2", "eto2", "pro2", 
                  "po2", "terpo2", "terp2o2", "nterpo2", "isopao2", "isopbo2", 
                  "mdialo2", "dicarbo2"]

RCHO_components = ["ald", "bzald", "glyald", "hydrald", "gly", "mgly", "hcho"]

OCSV_components = ["cvasoa2", "cvasoa3", "cvasoa4", "cvbsoa2", "cvbsoa3", "cvbsoa4"]

OCNV_components = ["cvasoaX", "cvasoa1", "cvbsoaX", "cvbsoa1"]

# Combine all species
all_component_species = []
if "RH" in chem_species:
    all_component_species.extend(RH_components)
if "RO2" in chem_species:
    all_component_species.extend(RO2_components)
if "RCHO" in chem_species:
    all_component_species.extend(RCHO_components)
if "OCSV" in chem_species:
    all_component_species.extend(OCSV_components)
if "OCNV" in chem_species:
    all_component_species.extend(OCNV_components)

all_component_species = list(set(all_component_species))

all_chem_to_process = list(set(chem_species_for_processing + all_component_species))
all_chem_to_process = [s for s in all_chem_to_process if s not in ["RH", "RO2", "RCHO", "OCSV", "OCNV"]]

# Aerosol variables
aerosol_vars = []
if aerosol_wrfchem:
    aerosol_mass_vars = get_all_wrfchem_variables(listspec)
    aerosol_num_vars = [f'num{bin_suffix}' for bin_suffix in WRFCHEM_BIN_SUFFIXES]
    aerosol_vars = aerosol_mass_vars + aerosol_num_vars
    all_chem_to_process.extend(aerosol_vars)
    print(f"Aerosol mass variables to process: {aerosol_mass_vars}")
    print(f"Aerosol number variables to process: {aerosol_num_vars}")
    print(f"Total aerosol variables: {len(aerosol_vars)}")

print(f"Total species to process: {len(all_chem_to_process)}")

# Domain parameters
palm_proj_code = ast.literal_eval(config.get("domain", "palm_proj"))[0]
centlat = ast.literal_eval(config.get("domain", "centlat"))[0]
centlon = ast.literal_eval(config.get("domain", "centlon"))[0]
dx = ast.literal_eval(config.get("domain", "dx"))[0]
dy = ast.literal_eval(config.get("domain", "dy"))[0]
dz = ast.literal_eval(config.get("domain", "dz"))[0]
nx = ast.literal_eval(config.get("domain", "nx"))[0]
ny = ast.literal_eval(config.get("domain", "ny"))[0]
nz = ast.literal_eval(config.get("domain", "nz"))[0]
z_origin = ast.literal_eval(config.get("domain", "z_origin"))[0]

# Create coordinate arrays (float32)
y = np.arange(dy/2, dy*ny + dy/2, dy, dtype=np.float32)
x = np.arange(dx/2, dx*nx + dx/2, dx, dtype=np.float32)
z = np.arange(dz/2, dz*nz, dz, dtype=np.float32)
xu = x + np.gradient(x)/2
xu = xu[:-1].astype(np.float32)
yv = y + np.gradient(y)/2
yv = yv[:-1].astype(np.float32)
zw = z + np.gradient(z)/2
zw = zw[:-1].astype(np.float32)

# Stretch grid
dz_stretch_factor = ast.literal_eval(config.get("stretch", "dz_stretch_factor"))[0]
dz_stretch_level = ast.literal_eval(config.get("stretch", "dz_stretch_level"))[0]
dz_max = ast.literal_eval(config.get("stretch", "dz_max"))[0]

if dz_stretch_factor > 1.0:
    z, zw = calc_stretch(z, dz, zw, dz_stretch_factor, dz_stretch_level, dz_max)

z += z_origin
zw += z_origin

dz_soil = np.array(ast.literal_eval(config.get("soil", "dz_soil")), dtype=np.float32)
msoil_val = np.array(ast.literal_eval(config.get("soil", "msoil")))[0]

wrf_path = ast.literal_eval(config.get("wrf", "wrf_path"))[0]
wrf_file = ast.literal_eval(config.get("wrf", "wrf_output"))
interp_mode = ast.literal_eval(config.get("wrf", "interp_mode"))[0]

start_year = ast.literal_eval(config.get("wrf", "start_year"))[0]
start_month = ast.literal_eval(config.get("wrf", "start_month"))[0]
start_day = ast.literal_eval(config.get("wrf", "start_day"))[0]
start_hour = ast.literal_eval(config.get("wrf", "start_hour"))[0]

end_year = ast.literal_eval(config.get("wrf", "end_year"))[0]
end_month = ast.literal_eval(config.get("wrf", "end_month"))[0]
end_day = ast.literal_eval(config.get("wrf", "end_day"))[0]
end_hour = ast.literal_eval(config.get("wrf", "end_hour"))[0]
dynamic_ts = ast.literal_eval(config.get("wrf", "dynamic_ts"))[0]

#===============================================================================
# Read WRF Files
#===============================================================================
print("Reading WRF files...")
if len(wrf_file) == 1:
    wrf_files = sorted(glob(wrf_path + wrf_file[0]))
else:
    wrf_files = sorted([wrf_path + file for file in wrf_file])

ds_wrf = xr.Dataset()
with salem.open_mf_wrf_dataset(wrf_files) as ds_raw:
    if len(ds_raw["time"]) == 1:
        ds_raw = ds_raw.isel(time=0)
        ds_raw = ds_raw.rename({"xtime": "time"})
    for variables in ds_raw.data_vars:
        ds_wrf[variables] = ds_raw[variables].drop_duplicates("time", keep="last").astype(np.float32)
    ds_wrf.attrs = ds_raw.attrs
del ds_raw
gc.collect()

# ===== EXTRACT ALT (INVERSE DENSITY) FOR UNIT CONVERSION =====
alt_wrf = ds_wrf['ALT'].astype(np.float32)
print(f"ALT (inverse density) loaded. Shape: {alt_wrf.shape}")

#===============================================================================
# Find Timestamps
#===============================================================================
dt_start = datetime(start_year, start_month, start_day, start_hour)
dt_end = datetime(end_year, end_month, end_day, end_hour)

wrf_ts = (ds_wrf["time"][1] - ds_wrf["time"][0]).data.astype("float64") * 1e-9

if dynamic_ts < wrf_ts:
    raise SystemExit("Invalid timesteps given. Stopping...")

num_ts = (dt_end - dt_start) / timedelta(seconds=dynamic_ts)
all_ts = [dt_start + i * timedelta(seconds=dynamic_ts) for i in range(0, floor(num_ts) + 1)]
if floor(num_ts) != ceil(num_ts):
    all_ts.append(dt_end)

all_ts = np.array(all_ts).astype("datetime64[ns]")
ds_wrf = ds_wrf.sel(time=all_ts)

times_sec = np.zeros(len(all_ts), dtype=np.float32)
for t in range(0, len(all_ts)):
    times_sec[t] = (all_ts[t] - all_ts[0]).astype('float') * 1e-9

#===============================================================================
# Locate PALM Domain in WRF
#===============================================================================
map_proj = ds_wrf.MAP_PROJ
wrf_map_dict = {1: "lcc", 2: "stere", 3: "merc", 6: "latlong"}

if map_proj not in wrf_map_dict:
    raise SystemExit("Incompatible WRF map projection, stopping...")

wgs_proj = Proj(proj='latlong', datum='WGS84', ellips='sphere')
dx_wrf, dy_wrf = ds_wrf.DX, ds_wrf.DY

if map_proj == 6:
    wrf_proj = wgs_proj
    xx_wrf = ds_wrf.lon.data
    yy_wrf = ds_wrf.lat.data
else:
    wrf_proj = Proj(proj=wrf_map_dict[map_proj],
                    lat_1=ds_wrf.TRUELAT1, lat_2=ds_wrf.TRUELAT2,
                    lat_0=ds_wrf.MOAD_CEN_LAT, lon_0=ds_wrf.STAND_LON,
                    a=6370000, b=6370000)
    trans_wgs2wrf = Transformer.from_proj(wgs_proj, wrf_proj)
    e, n = trans_wgs2wrf.transform(ds_wrf.CEN_LON, ds_wrf.CEN_LAT)
    nx_wrf, ny_wrf = ds_wrf.dims['west_east'], ds_wrf.dims['south_north']
    x0_wrf = -(nx_wrf - 1) / 2. * dx_wrf + e
    y0_wrf = -(ny_wrf - 1) / 2. * dy_wrf + n
    xx_wrf, yy_wrf = np.meshgrid(np.arange(nx_wrf) * dx_wrf + x0_wrf,
                                 np.arange(ny_wrf) * dy_wrf + y0_wrf)

print(f"\nFull WRF Projection Info:")
print(f"  MAP_PROJ: {map_proj} ({wrf_map_dict[map_proj]})")
print(f"  TRUELAT1: {ds_wrf.TRUELAT1}")
print(f"  TRUELAT2: {ds_wrf.TRUELAT2}")
print(f"  MOAD_CEN_LAT: {ds_wrf.MOAD_CEN_LAT}")
print(f"  STAND_LON: {ds_wrf.STAND_LON}")
print(f"  CEN_LAT: {ds_wrf.CEN_LAT}")
print(f"  CEN_LON: {ds_wrf.CEN_LON}")
print(f"  DX, DY: {dx_wrf}, {dy_wrf}")
print(f"  x0_wrf, y0_wrf: {x0_wrf:.1f}, {y0_wrf:.1f}")

if len(palm_proj_code) == 0:
    palm_proj = wrf_proj
else:
    palm_proj = Proj(init=palm_proj_code)

trans_wrf2palm = Transformer.from_proj(wrf_proj, palm_proj)
lons_wrf, lats_wrf = trans_wrf2palm.transform(xx_wrf, yy_wrf)

west, east, south, north, centx, centy = domain_location(palm_proj, wgs_proj, centlat, centlon,
                                                          dx, dy, nx, ny)

generate_cfg(case_name, dx, dy, dz, nx, ny, nz,
             west, east, south, north, centlat, centlon, z_origin)

west_idx, east_idx, south_idx, north_idx = framing_2d_cartesian(lons_wrf, lats_wrf, west, east, south, north, dx_wrf, dy_wrf)

if east_idx - west_idx < 0:
    east_idx, west_idx = west_idx, east_idx

if north_idx - south_idx < 1 or east_idx - west_idx < 1:
    raise SystemExit("PALM domain size is smaller than one WRF grid cell size.\nStopping...")

mask_sn = (ds_wrf.south_north >= ds_wrf.south_north[south_idx]) & (ds_wrf.south_north <= ds_wrf.south_north[north_idx])
mask_we = (ds_wrf.west_east >= ds_wrf.west_east[west_idx]) & (ds_wrf.west_east <= ds_wrf.west_east[east_idx])

ds_drop = ds_wrf.where(mask_sn & mask_we, drop=True)

print(f"  WRF grid indices:  WE=[{west_idx}:{east_idx}], SN=[{south_idx}:{north_idx}]")
print(f"  WRF cells extracted: {east_idx-west_idx+1} x {north_idx-south_idx+1}")
print("-"*40 + "\n")

ds_drop["pt"] = ds_drop["T"] + 300
ds_drop["pt"].attrs = ds_drop["T"].attrs
ds_drop["gph"] = (ds_drop["PH"] + ds_drop["PHB"]) / 9.81
ds_drop["gph"].attrs = ds_drop["PH"].attrs

#===============================================================================
# PROCESSING HEADER
#===============================================================================
print("\n" + "="*80)
print("WRF4PALM DYNAMIC DRIVER GENERATION")
print("="*80)
print(f"  Processing date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print(f"  Case: {case_name}")
print(f"  Period: {dt_start.strftime('%Y-%m-%d %H:%M')} to {dt_end.strftime('%Y-%m-%d %H:%M')} UTC ({len(all_ts)} timesteps)")
print(f"  PALM domain: {nx}x{ny}x{nz}, {dx:.0f}mx{dy:.0f}mx{dz:.0f}m ({nx*dx:.0f}m x {ny*dy:.0f}m)")
print("="*80 + "\n")

#===============================================================================
# Horizontal Interpolation (Vectorized via xarray)
#===============================================================================
print("Start horizontal interpolation...")
south_north_palm = ds_drop.south_north[0].data + y
west_east_palm = ds_drop.west_east[0].data + x
south_north_v_palm = ds_drop.south_north[0].data + yv
west_east_u_palm = ds_drop.west_east[0].data + xu

ds_drop = ds_drop.assign_coords({"west_east_palm": west_east_palm,
                                 "south_north_palm": south_north_palm,
                                 "west_east_u_palm": west_east_u_palm,
                                 "south_north_v_palm": south_north_v_palm})

chunks = {"time": 1, "south_north": -1, "west_east": -1}
ds_drop = ds_drop.chunk(chunks)

ds_interp = ds_drop.interp({"west_east": ds_drop.west_east_palm}, method=interp_mode
                          ).interp({"south_north": ds_drop.south_north_palm}, method=interp_mode)
ds_interp_u = ds_drop.interp({"west_east": ds_drop.west_east_u_palm}, method=interp_mode
                            ).interp({"south_north": ds_drop.south_north_palm}, method=interp_mode)
ds_interp_v = ds_drop.interp({"west_east": ds_drop.west_east_palm}, method=interp_mode
                            ).interp({"south_north": ds_drop.south_north_v_palm}, method=interp_mode)

ds_interp = ds_interp.drop(["west_east", "south_north"]).rename({"west_east_palm": "west_east",
                                                                  "south_north_palm": "south_north"})
ds_interp_u = ds_interp_u.drop(["west_east", "south_north"]).rename({"west_east_u_palm": "west_east",
                                                                      "south_north_palm": "south_north"})
ds_interp_v = ds_interp_v.drop(["west_east", "south_north"]).rename({"west_east_palm": "west_east",
                                                                      "south_north_v_palm": "south_north"})

for var in ds_interp.data_vars:
    ds_interp[var] = ds_interp[var].astype(np.float32)
for var in ds_interp_u.data_vars:
    ds_interp_u[var] = ds_interp_u[var].astype(np.float32)
for var in ds_interp_v.data_vars:
    ds_interp_v[var] = ds_interp_v[var].astype(np.float32)

# Handle LU_INDEX with NEAREST neighbor
if 'LU_INDEX' in ds_drop.data_vars:
    print("\n  Interpolating LU_INDEX with NEAREST neighbor...")
    ds_interp['LU_INDEX'] = ds_drop['LU_INDEX'].interp(
        {"west_east": ds_drop.west_east_palm}, 
        method='nearest'
    ).interp(
        {"south_north": ds_drop.south_north_palm}, 
        method='nearest'
    ).astype(np.int32)

# Get surface fields
zs_wrf = ds_interp.ZS[0, :, 0, 0].load().astype(np.float32)
t2_wrf = ds_interp.T2.load().astype(np.float32)
u10_wrf = ds_interp_u.U10.load().astype(np.float32)
v10_wrf = ds_interp_v.V10.load().astype(np.float32)
qv2_wrf = ds_interp.Q2.load().astype(np.float32)
psfc_wrf = ds_interp.PSFC.load().astype(np.float32)
pt2_wrf = t2_wrf * ((1000) / (psfc_wrf * 0.01)) ** 0.286

surface_var_dict = {"U": u10_wrf, "V": v10_wrf, "pt": pt2_wrf, "QVAPOR": qv2_wrf, "W": None}

#===============================================================================
# Soil Moisture and Temperature
#===============================================================================
print("Calculating soil temperature and moisture from WRF...")

watermask = ds_interp["LANDMASK"].sel(time=dt_start).load().data == 0
landmask = ds_interp["LANDMASK"].sel(time=dt_start).load().data == 1
median_smois = [np.nanmedian(ds_interp["SMOIS"][0, izs, :, :].load().data[landmask]) for izs in range(0, len(zs_wrf))]
ds_interp["soil_layers"] = zs_wrf.load().data
tslb_wrf = ds_interp["TSLB"].sel(time=dt_start).load().astype(np.float32)
smois_wrf = ds_interp["SMOIS"].sel(time=dt_start).load().astype(np.float32)
deep_soil_wrf = ds_interp["TMN"].sel(time=dt_start).load().astype(np.float32)
deep_tsoil = deep_soil_wrf.where(landmask).mean().load().data

if np.isnan(median_smois[0]):
    print("Warning: Entire PALM domain over water surface.")
    median_smois = np.ones_like(median_smois, dtype=np.float32)
    deep_tsoil = deep_soil_wrf.mean().load().data

for izs in range(0, len(zs_wrf)):
    smois_wrf.isel(soil_layers=izs).data[watermask] = median_smois[izs]
    if smois_wrf.isel(soil_layers=izs).mean() == 0.0:
        smois_wrf.isel(soil_layers=izs).data[:, :] = msoil_val

zs_palm = np.zeros_like(dz_soil, dtype=np.float32)
zs_palm[0] = dz_soil[0]
for i in range(1, len(dz_soil)):
    zs_palm[i] = np.sum(dz_soil[:i+1])

init_tsoil = np.zeros((len(dz_soil), len(y), len(x)), dtype=np.float32)
init_msoil = np.zeros((len(dz_soil), len(y), len(x)), dtype=np.float32)

for iy in tqdm(range(0, len(y)), desc="Soil interpolation", position=0, leave=True):
    for ix in range(0, len(x)):
        init_tsoil[:, iy, ix] = np.interp(zs_palm, zs_wrf.data, tslb_wrf[:, iy, ix])
        init_msoil[:, iy, ix] = np.interp(zs_palm, zs_wrf.data, smois_wrf[:, iy, ix])

#===============================================================================
# STREET TYPE CLASSIFICATION
#===============================================================================
print("\n" + "="*60)
print("STREET TYPE CLASSIFICATION")
print("="*60)

force_urban = False
if 'LU_INDEX' in ds_drop.data_vars and 'LU_INDEX' in ds_interp.data_vars:
    lu_cropped = ds_drop['LU_INDEX'].isel(time=0).values
    lu_interp = ds_interp['LU_INDEX'].isel(time=0).values
    if np.sum(lu_cropped == 13) > 0 and np.sum(lu_interp == 13) == 0:
        print("\n  WARNING: Urban pixels present in WRF but lost during interpolation!")

street_classification = classify_street_types_pixel_by_pixel(
    ds_interp=ds_interp,
    nx=nx,
    ny=ny,
    force_urban=force_urban
)

street_type_surface = street_classification['street_type_surface']
street_type_we = street_classification['street_type_we']
street_type_sn = street_classification['street_type_sn']
stats = street_classification['stats']

print("\n" + "="*60)
print("STREET TYPE CLASSIFICATION COMPLETE")
print("="*60)
print(f"  Urban pixels: {stats['urban_pixels']} ({stats['urban_pct']:.1f}%)")
print(f"  Rural pixels: {stats['rural_pixels']} ({stats['rural_pct']:.1f}%)")
print("="*60)

#===============================================================================
# Vertical Interpolation
#===============================================================================
print("\n" + "="*60)
print("Start vertical interpolation (vectorized, memory-efficient)")
print("="*60)

ds_we = ds_interp.isel(west_east=[0, -1])
ds_sn = ds_interp.isel(south_north=[0, -1])
ds_we_ustag = ds_interp_u.isel(west_east=[0, -1])
ds_we_vstag = ds_interp_v.isel(west_east=[0, -1])
ds_sn_ustag = ds_interp_u.isel(south_north=[0, -1])
ds_sn_vstag = ds_interp_v.isel(south_north=[0, -1])

varbc_list = ["W", "QVAPOR", "pt", "Z"]
varbc_list.extend(all_chem_to_process)

for var in list(ds_we.data_vars):
    if var not in varbc_list:
        ds_we = ds_we.drop(var)
        ds_sn = ds_sn.drop(var)
    if var not in ["U", "Z"] and var not in all_chem_to_process:
        ds_we_ustag = ds_we_ustag.drop(var)
        ds_sn_ustag = ds_sn_ustag.drop(var)
    if var not in ["V", "Z"] and var not in all_chem_to_process:
        ds_we_vstag = ds_we_vstag.drop(var)
        ds_sn_vstag = ds_sn_vstag.drop(var)

ds_palm_we = xr.Dataset()
ds_palm_we = ds_palm_we.assign_coords({"x": x[:2], "y": y, "time": ds_interp.time.data,
                                       "z": z, "yv": yv, "xu": xu[:2], "zw": zw})
ds_palm_sn = xr.Dataset()
ds_palm_sn = ds_palm_sn.assign_coords({"x": x, "y": y[:2], "time": ds_interp.time.data,
                                       "z": z, "yv": yv[:2], "xu": xu, "zw": zw})

# Process meteorological variables
met_vars = ["QVAPOR", "pt"]
for varbc in met_vars:
    print(f"Processing {varbc} for boundaries...")
    zeros_we = np.zeros((len(all_ts), len(z), len(y), len(x[:2])), dtype=np.float32)
    zeros_sn = np.zeros((len(all_ts), len(z), len(y[:2]), len(x)), dtype=np.float32)
    
    ds_palm_we[varbc] = xr.DataArray(zeros_we, dims=['time', 'z', 'y', 'x'])
    ds_palm_sn[varbc] = xr.DataArray(zeros_sn, dims=['time', 'z', 'y', 'x'])
    
    ds_palm_we[varbc] = multi_zinterp(max_pool, ds_we, varbc, z, ds_palm_we)
    ds_palm_sn[varbc] = multi_zinterp(max_pool, ds_sn, varbc, z, ds_palm_sn)
    
    del zeros_we, zeros_sn
    gc.collect()

# Process chemistry species
print(f"\nProcessing {len(all_chem_to_process)} chemistry species...")
available_species = [s for s in all_chem_to_process if s in list(ds_we.data_vars.keys())]
print(f"Available species: {len(available_species)}")

for i, species in enumerate(tqdm(available_species, desc="Chemistry species")):
    chem_zeros_we = np.zeros((len(all_ts), len(z), len(y), len(x[:2])), dtype=np.float32)
    chem_zeros_sn = np.zeros((len(all_ts), len(z), len(y[:2]), len(x)), dtype=np.float32)
    
    ds_palm_we[species] = xr.DataArray(chem_zeros_we, dims=['time', 'z', 'y', 'x'])
    ds_palm_sn[species] = xr.DataArray(chem_zeros_sn, dims=['time', 'z', 'y', 'x'])
    
    ds_palm_we[species] = multi_zinterp(max_pool, ds_we, species, z, ds_palm_we)
    ds_palm_sn[species] = multi_zinterp(max_pool, ds_sn, species, z, ds_palm_sn)
    
    del chem_zeros_we, chem_zeros_sn
    gc.collect()

# Process W, U, V
print("Processing W for boundaries...")
zeros_we_w = np.zeros((len(all_ts), len(zw), len(y), len(x[:2])), dtype=np.float32)
zeros_sn_w = np.zeros((len(all_ts), len(zw), len(y[:2]), len(x)), dtype=np.float32)
ds_palm_we["W"] = xr.DataArray(zeros_we_w, dims=['time', 'zw', 'y', 'x'])
ds_palm_sn["W"] = xr.DataArray(zeros_sn_w, dims=['time', 'zw', 'y', 'x'])
ds_palm_we["W"] = multi_zinterp(max_pool, ds_we, "W", zw, ds_palm_we)
ds_palm_sn["W"] = multi_zinterp(max_pool, ds_sn, "W", zw, ds_palm_sn)
del zeros_we_w, zeros_sn_w
gc.collect()

print("Processing U for boundaries...")
zeros_we_u = np.zeros((len(all_ts), len(z), len(y), len(xu[:2])), dtype=np.float32)
zeros_sn_u = np.zeros((len(all_ts), len(z), len(y[:2]), len(xu)), dtype=np.float32)
ds_palm_we["U"] = xr.DataArray(zeros_we_u, dims=['time', 'z', 'y', 'xu'])
ds_palm_we["U"] = multi_zinterp(max_pool, ds_we_ustag, "U", z, ds_palm_we)
ds_palm_sn["U"] = xr.DataArray(zeros_sn_u, dims=['time', 'z', 'y', 'xu'])
ds_palm_sn["U"] = multi_zinterp(max_pool, ds_sn_ustag, "U", z, ds_palm_sn)
del zeros_we_u, zeros_sn_u
gc.collect()

print("Processing V for boundaries...")
zeros_we_v = np.zeros((len(all_ts), len(z), len(yv), len(x[:2])), dtype=np.float32)
zeros_sn_v = np.zeros((len(all_ts), len(z), len(yv[:2]), len(x)), dtype=np.float32)
ds_palm_we["V"] = xr.DataArray(zeros_we_v, dims=['time', 'z', 'yv', 'x'])
ds_palm_we["V"] = multi_zinterp(max_pool, ds_we_vstag, "V", z, ds_palm_we)
ds_palm_sn["V"] = xr.DataArray(zeros_sn_v, dims=['time', 'z', 'yv', 'x'])
ds_palm_sn["V"] = multi_zinterp(max_pool, ds_sn_vstag, "V", z, ds_palm_sn)
del zeros_we_v, zeros_sn_v
gc.collect()

# Traffic variables
if has_traffic_vars:
    print("Setting up traffic variables in boundary conditions...")
    for base_species, traffic_species in traffic_mapping.items():
        if base_species in ds_palm_we.data_vars:
            ds_palm_we[traffic_species] = ds_palm_we[base_species].copy()
            ds_palm_sn[traffic_species] = ds_palm_sn[base_species].copy()

# Handle NaN values
print("Handling NaN values in boundary conditions...")
for species in list(ds_palm_we.data_vars):
    if np.any(np.isnan(ds_palm_we[species].data)) or np.any(np.isnan(ds_palm_sn[species].data)):
        dims_we = ds_palm_we[species].dims
        dims_sn = ds_palm_sn[species].dims
        if 'zw' in dims_we:
            ds_palm_we[species] = ds_palm_we[species].ffill('zw').bfill('zw').fillna(0)
        else:
            ds_palm_we[species] = ds_palm_we[species].ffill('z').bfill('z').fillna(0)
        if 'zw' in dims_sn:
            ds_palm_sn[species] = ds_palm_sn[species].ffill('zw').bfill('zw').fillna(0)
        else:
            ds_palm_sn[species] = ds_palm_sn[species].ffill('z').bfill('z').fillna(0)

#===============================================================================
# Top Boundary
#===============================================================================
print("\nProcessing top boundary conditions...")

u_top = np.zeros((len(all_ts), len(y), len(xu)), dtype=np.float32)
v_top = np.zeros((len(all_ts), len(yv), len(x)), dtype=np.float32)
w_top = np.zeros((len(all_ts), len(y), len(x)), dtype=np.float32)
qv_top = np.zeros((len(all_ts), len(y), len(x)), dtype=np.float32)
pt_top = np.zeros((len(all_ts), len(y), len(x)), dtype=np.float32)

chem_top = {}
available_top_species = [s for s in all_chem_to_process if s in ds_interp.data_vars]

for species in available_top_species:
    chem_top[species] = np.zeros((len(all_ts), len(y), len(x)), dtype=np.float32)

for ts in tqdm(range(len(all_ts)), desc="Top boundary"):
    u_top[ts, :, :] = ds_interp_u["U"].isel(time=ts, bottom_top=-1).astype(np.float32)
    v_top[ts, :, :] = ds_interp_v["V"].isel(time=ts, bottom_top=-1).astype(np.float32)
    w_top[ts, :, :] = ds_interp["W"].isel(time=ts, bottom_top=-1).astype(np.float32)
    pt_top[ts, :, :] = ds_interp["pt"].isel(time=ts, bottom_top=-1).astype(np.float32)
    qv_top[ts, :, :] = ds_interp["QVAPOR"].isel(time=ts, bottom_top=-1).astype(np.float32)
    
    for species in available_top_species:
        if species in ds_interp.data_vars:
            chem_top[species][ts, :, :] = ds_interp[species].isel(time=ts, bottom_top=-1).astype(np.float32)

# Aggregate species
if "RH" in chem_species:
    chem_top["RH"] = np.zeros((len(all_ts), len(y), len(x)), dtype=np.float32)
    for comp in RH_components:
        if comp in chem_top:
            chem_top["RH"] += chem_top[comp]

if "RO2" in chem_species:
    chem_top["RO2"] = np.zeros((len(all_ts), len(y), len(x)), dtype=np.float32)
    for comp in RO2_components:
        if comp in chem_top:
            chem_top["RO2"] += chem_top[comp]

if "RCHO" in chem_species:
    chem_top["RCHO"] = np.zeros((len(all_ts), len(y), len(x)), dtype=np.float32)
    for comp in RCHO_components:
        if comp in chem_top:
            chem_top["RCHO"] += chem_top[comp]

if "OCSV" in chem_species:
    chem_top["OCSV"] = np.zeros((len(all_ts), len(y), len(x)), dtype=np.float32)
    for comp in OCSV_components:
        if comp in chem_top:
            chem_top["OCSV"] += chem_top[comp]

if "OCNV" in chem_species:
    chem_top["OCNV"] = np.zeros((len(all_ts), len(y), len(x)), dtype=np.float32)
    for comp in OCNV_components:
        if comp in chem_top:
            chem_top["OCNV"] += chem_top[comp]

if has_traffic_vars:
    for base_species, traffic_species in traffic_mapping.items():
        if base_species in chem_top:
            chem_top[traffic_species] = chem_top[base_species].copy()

for species in original_chem_species + aerosol_vars:
    if species in chem_top:
        if np.any(np.isnan(chem_top[species])):
            mean_profile = np.nanmean(chem_top[species], axis=(1, 2))
            for ts in range(len(all_ts)):
                nan_mask = np.isnan(chem_top[species][ts, :, :])
                if np.any(nan_mask):
                    chem_top[species][ts, nan_mask] = mean_profile[ts]

#===============================================================================
# Geostrophic Wind
#===============================================================================
print("\nGeostrophic wind estimation...")

if geostr_lvl == "z":
    lat_geostr = ds_drop.lat[:, 0]
    gph = ds_drop.gph.load()
    ds_geostr = xr.Dataset()
    ds_geostr = ds_geostr.assign_coords({"time": ds_drop.time.data,
                                         "z": ds_drop["Z"].mean(("time", "south_north", "west_east")).data})
    ds_geostr["ug"] = xr.DataArray(np.zeros((len(all_ts), len(gph.bottom_top.data)), dtype=np.float32), dims=['time', 'z'])
    ds_geostr["vg"] = xr.DataArray(np.zeros((len(all_ts), len(gph.bottom_top.data)), dtype=np.float32), dims=['time', 'z'])

    for ts in tqdm(range(len(all_ts)), desc="Geostrophic wind"):
        for levels in gph.bottom_top.data:
            ug, vg = calc_geostrophic_wind_zlevels(gph[ts, levels, :, :].data, lat_geostr.data, dy_wrf, dx_wrf)
            ds_geostr["ug"][ts, levels] = ug
            ds_geostr["vg"][ts, levels] = vg

    ds_geostr = ds_geostr.interp({"z": z})

elif geostr_lvl == "p":
    pres = ds_drop.PRESSURE.load()
    tk = ds_drop.TK.load()
    lat_1d = ds_drop.lat[:, 0]
    lon_1d = ds_drop.lon[0, :]
    
    ds_geostr = xr.Dataset()
    ds_geostr = ds_geostr.assign_coords({"time": ds_drop.time.data,
                                         "z": ds_drop["Z"].mean(("time", "south_north", "west_east")).data})
    ds_geostr["ug"] = xr.DataArray(np.zeros((len(all_ts), len(pres.bottom_top.data)), dtype=np.float32), dims=['time', 'z'])
    ds_geostr["vg"] = xr.DataArray(np.zeros((len(all_ts), len(pres.bottom_top.data)), dtype=np.float32), dims=['time', 'z'])

    for ts in tqdm(range(len(all_ts)), desc="Geostrophic wind"):
        for levels in pres.bottom_top.data:
            geo_wind = calc_geostrophic_wind_plevels(pres[ts, levels, :, :].data, tk[ts, levels, :, :].data,
                                                      lat_1d, lon_1d, dy_wrf, dx_wrf)
            ds_geostr["ug"][ts, levels] = geo_wind[0]
            ds_geostr["vg"][ts, levels] = geo_wind[1]

    ds_geostr = ds_geostr.interp({"z": z})
else:
    ds_geostr = xr.Dataset()
    ds_geostr = ds_geostr.assign_coords({"time": all_ts, "z": z})
    ds_geostr["ug"] = xr.DataArray(np.zeros((len(all_ts), len(z)), dtype=np.float32), dims=['time', 'z'])
    ds_geostr["vg"] = xr.DataArray(np.zeros((len(all_ts), len(z)), dtype=np.float32), dims=['time', 'z'])

#===============================================================================
# Surface NaNs
#===============================================================================
print("Resolving surface NaNs...")
with Pool(max_pool) as p:
    pool_outputs = list(
        tqdm(
            p.imap(partial(solve_surface, all_ts, ds_palm_we, ds_palm_sn, surface_var_dict), 
                   surface_var_dict.keys()),
            total=len(surface_var_dict.keys()), position=0, leave=True
        )
    )
pool_dict = dict(pool_outputs)
for var in surface_var_dict.keys():
    ds_palm_we[var] = pool_dict[var][0]
    ds_palm_sn[var] = pool_dict[var][1]

if ds_geostr is not None:
    for t in range(0, len(all_ts)):
        ds_geostr["ug"][t, :] = surface_nan_w(ds_geostr["ug"][t, :].data)
        ds_geostr["vg"][t, :] = surface_nan_w(ds_geostr["vg"][t, :].data)

#===============================================================================
# Initial Profiles
#===============================================================================
print("Calculating initial profiles...")
ds_drop["bottom_top"] = ds_drop["Z"].mean(("time", "south_north", "west_east")).data

u_init = ds_drop["U"].sel(time=dt_start).mean(dim=["south_north", "west_east"]).interp(
    {"bottom_top": z}, method=interp_mode).astype(np.float32)
v_init = ds_drop["V"].sel(time=dt_start).mean(dim=["south_north", "west_east"]).interp(
    {"bottom_top": z}, method=interp_mode).astype(np.float32)
w_init = ds_drop["W"].sel(time=dt_start).mean(dim=["south_north", "west_east"]).interp(
    {"bottom_top": zw}, method=interp_mode).astype(np.float32)
qv_init = ds_drop["QVAPOR"].sel(time=dt_start).mean(dim=["south_north", "west_east"]).interp(
    {"bottom_top": z}, method=interp_mode).astype(np.float32)
pt_init = ds_drop["pt"].sel(time=dt_start).mean(dim=["south_north", "west_east"]).interp(
    {"bottom_top": z}, method=interp_mode).astype(np.float32)

u10_mean = u10_wrf.sel(time=dt_start).mean(dim=["south_north", "west_east"]).data
v10_mean = v10_wrf.sel(time=dt_start).mean(dim=["south_north", "west_east"]).data
qv2_mean = qv2_wrf.sel(time=dt_start).mean(dim=["south_north", "west_east"]).data
pt2_mean = pt2_wrf.sel(time=dt_start).mean(dim=["south_north", "west_east"]).data

u_init = surface_nan_uv(u_init.data, z, u10_mean)
v_init = surface_nan_uv(v_init.data, z, v10_mean)
w_init = surface_nan_w(w_init.data)
qv_init = surface_nan_s(qv_init.data, z, qv2_mean)
pt_init = surface_nan_s(pt_init.data, z, pt2_mean)

# Initialize chemistry profiles
chem_init = {}
for species in all_chem_to_process:
    if species in ds_drop.data_vars:
        chem_data = ds_drop[species].sel(time=dt_start).mean(
            dim=["south_north", "west_east"]).interp(
            {"bottom_top": z}, method=interp_mode).load().data.astype(np.float32)
        chem_init[species] = xr.DataArray(chem_data, dims=['z'], coords={'z': z})
    else:
        chem_init[species] = xr.DataArray(np.zeros(len(z), dtype=np.float32), dims=['z'], coords={'z': z})

# Aggregate species
if "RH" in chem_species:
    rh_init = np.zeros(len(z), dtype=np.float32)
    for comp in RH_components:
        if comp in chem_init:
            rh_init += chem_init[comp].values
    chem_init["RH"] = xr.DataArray(rh_init, dims=['z'], coords={'z': z})

if "RO2" in chem_species:
    ro2_init = np.zeros(len(z), dtype=np.float32)
    for comp in RO2_components:
        if comp in chem_init:
            ro2_init += chem_init[comp].values
    chem_init["RO2"] = xr.DataArray(ro2_init, dims=['z'], coords={'z': z})

if "RCHO" in chem_species:
    rcho_init = np.zeros(len(z), dtype=np.float32)
    for comp in RCHO_components:
        if comp in chem_init:
            rcho_init += chem_init[comp].values
    chem_init["RCHO"] = xr.DataArray(rcho_init, dims=['z'], coords={'z': z})

if "OCSV" in chem_species:
    ocsv_init = np.zeros(len(z), dtype=np.float32)
    for comp in OCSV_components:
        if comp in chem_init:
            ocsv_init += chem_init[comp].values
    chem_init["OCSV"] = xr.DataArray(ocsv_init, dims=['z'], coords={'z': z})

if "OCNV" in chem_species:
    ocnv_init = np.zeros(len(z), dtype=np.float32)
    for comp in OCNV_components:
        if comp in chem_init:
            ocnv_init += chem_init[comp].values
    chem_init["OCNV"] = xr.DataArray(ocnv_init, dims=['z'], coords={'z': z})

if has_traffic_vars:
    for base_species, traffic_species in traffic_mapping.items():
        if base_species in chem_init:
            chem_init[traffic_species] = chem_init[base_species].copy()

surface_pres = psfc_wrf[:, :, :].mean(dim=["south_north", "west_east"]).load().astype(np.float32)

#===============================================================================
# AEROSOL PROCESSING - WITH UNIT CONVERSION AND CORRECT BIN MAPPING
#===============================================================================

if aerosol_wrfchem:
    print("\n" + "="*60)
    print("PROCESSING AEROSOL DATA (with unit conversion and corrected bin mapping)")
    print("="*60)
    
    # ===== BIN DEFINITIONS =====
    dmid, bin_limits = define_bins(nbin, reglim)
    n_bins_1 = nbin[0]
    n_bins_2 = nbin[1]
    
    print(f"  PALM bin structure: {n_bins_1} subrange 1 + {n_bins_2} subrange 2")
    print(f"  PALM bin limits (nm): {[f'{l*1e9:.1f}' for l in bin_limits]}")
    
    # Calculate overlap ratio
    overlap_ratio = aerosol_binoverlap(bin_limits, wrfchem_bin_limits)
    print(f"  Overlap ratio shape: {overlap_ratio.shape}")
    
    # Total PALM bins
    if nf2a < 1.0:
        nbins_total = n_bins_1 + 2 * n_bins_2
        dmid_all = np.concatenate([dmid[:n_bins_1], dmid[n_bins_1:], dmid[n_bins_1:]])
        print(f"  Total PALM bins: {nbins_total} ({n_bins_1} 1a + {n_bins_2} 2a + {n_bins_2} 2b)")
    else:
        nbins_total = n_bins_1 + n_bins_2
        dmid_all = dmid.copy()
        print(f"  Total PALM bins: {nbins_total} ({n_bins_1} 1a + {n_bins_2} 2a)")
    
    n_species = len(listspec)
    n_wrf_bins = len(WRFCHEM_BIN_SUFFIXES)
    print(f"  Species: {n_species}, WRF-Chem bins: {n_wrf_bins}")
    print(f"  nf2a = {nf2a}")
    
    print(f"\n  Species classification:")
    print(f"    Insoluble: {INSOLUBLE_SPECIES}")
    print(f"    Soluble:   {SOLUBLE_SPECIES}")
    
    # ===== STEP 1: Initial Profile Aerosol =====
    print("\n" + "-"*40)
    print("STEP 1: Calculating aerosol initial profiles (with unit conversion)...")
    print("-"*40)
    
    # Get ALT for unit conversion (domain mean at initial time)
    alt_init_data = ds_wrf['ALT'].isel(
        time=0,
        west_east=slice(west_idx, east_idx + 1),
        south_north=slice(south_idx, north_idx + 1)
    ).mean().values
    
    air_density_init = get_air_density(alt_init_data, "ALT_init")
    
    print(f"    ALT (inverse density) from WRF: {extract_scalar_from_xarray(alt_init_data):.4f} m³/kg")
    print(f"    Air density for unit conversion: {air_density_init:.3f} kg/m³")
    
    # Initialize mass matrix (kg/m³ for each species)
    mass_matrix = np.zeros((len(z), n_species), dtype=np.float32)
    
    # Process explicit WRF-Chem species
    for idx, spec in enumerate(listspec):
        if is_trace_metal(spec):
            continue
        wrf_names = get_wrfchem_variables_for_species(spec)
        for wrf_name in wrf_names:
            for bin_suffix in WRFCHEM_BIN_SUFFIXES:
                var_name = f'{wrf_name}{bin_suffix}'
                if var_name in chem_init:
                    # Convert μg/kg-dryair -> kg/m³
                    mass_matrix[:, idx] += chem_init[var_name].values * 1e-9 * air_density_init
    
    # Add trace metals
    trace_metal_added = False
    for idx, spec in enumerate(listspec):
        if is_trace_metal(spec):
            if 'PM2_5_DRY' in chem_init:
                pm25_kg_m3 = chem_init['PM2_5_DRY'].values * 1e-9 * air_density_init
                fraction = get_trace_metal_mass_fraction(spec, street_type_surface)
                mass_matrix[:, idx] = pm25_kg_m3 * np.mean(fraction)
                trace_metal_added = True
            else:
                mass_matrix[:, idx] = 0.0
    
    if trace_metal_added:
        print("    Trace metals added using literature mass fractions")
    
    # Normalize mass fractions
    mass_fracs_orig = vectorized_mass_fraction_batch(mass_matrix)
    mass_fracs_a_init, mass_fracs_b_init = create_separated_mass_fractions(
        mass_fracs_orig, listspec, nf2a
    )
    
    # Collect WRF number concentrations (#/kg -> #/m³)
    wrf_num_matrix = np.zeros((len(z), n_wrf_bins), dtype=np.float32)
    for wbin, bin_suffix in enumerate(WRFCHEM_BIN_SUFFIXES):
        num_var = f'num{bin_suffix}'
        if num_var in chem_init:
            wrf_num_matrix[:, wbin] = chem_init[num_var].values * air_density_init
    
    print(f"    WRF number conc. range: {wrf_num_matrix.min():.2e} - {wrf_num_matrix.max():.2e} #/m³")
    
    # Map WRF 4 bins to PALM 10 bins
    aerosol_conc_10bin = vectorized_batch_aerosol_mapping(wrf_num_matrix, overlap_ratio)
    
    # Split into 1a, 2a, 2b
    if nf2a < 1.0:
        conc_1 = aerosol_conc_10bin[:, :n_bins_1]
        conc_2 = aerosol_conc_10bin[:, n_bins_1:]
        conc_2a = conc_2 * nf2a
        conc_2b = conc_2 * (1.0 - nf2a)
        aerosol_concentration_init = np.concatenate([conc_1, conc_2a, conc_2b], axis=1)
    else:
        aerosol_concentration_init = aerosol_conc_10bin
    
    chem_init['mass_fracs_a'] = xr.DataArray(mass_fracs_a_init.astype(np.float32), 
                                              dims=['z', 'composition_index'])
    chem_init['mass_fracs_b'] = xr.DataArray(mass_fracs_b_init.astype(np.float32), 
                                              dims=['z', 'composition_index'])
    chem_init['aerosol'] = xr.DataArray(aerosol_concentration_init.astype(np.float32), 
                                         dims=['z', 'Dmid'])
    
    print(f"\n    Aerosol concentration shape: {aerosol_concentration_init.shape}")
    print(f"    Concentration range: {aerosol_concentration_init.min():.2e} - {aerosol_concentration_init.max():.2e} #/m³")
    
    print(f"\n    Bin concentrations (mean over first 5 z-levels):")
    for bin_idx in range(min(nbins_total, 17)):
        if bin_idx < aerosol_concentration_init.shape[1]:
            bin_mean = np.mean(aerosol_concentration_init[:5, bin_idx])
            bin_label = ""
            if bin_idx < n_bins_1:
                bin_label = "1a"
            elif nf2a < 1.0 and bin_idx < n_bins_1 + n_bins_2:
                bin_label = "2a"
            elif nf2a < 1.0:
                bin_label = "2b"
            else:
                bin_label = "2a"
            print(f"      Bin {bin_idx+1:2d} ({bin_label}, d={dmid_all[bin_idx]*1e9:.1f} nm): {bin_mean:.2e} #/m³")
    
    print(f"\n    Composition summary (averaged over all z levels):")
    print(f"      {'Species':<8} {'mass_frac_a':<14} {'mass_frac_b':<14}")
    print(f"      {'-'*8} {'-'*14} {'-'*14}")
    for i, spec in enumerate(listspec):
        mean_a = np.mean(mass_fracs_a_init[:, i])
        mean_b = np.mean(mass_fracs_b_init[:, i])
        print(f"      {spec:<8} {mean_a:>13.6f} {mean_b:>13.6f}")
    
    # ===== STEP 2: Boundary Conditions Aerosol =====
    print("\n" + "-"*40)
    print("STEP 2: Calculating aerosol boundary conditions (with unit conversion)...")
    print("-"*40)
    
    # Initialize boundary arrays
    left_aerosol = np.zeros((len(all_ts), len(z), len(y), nbins_total), dtype=np.float32)
    right_aerosol = np.zeros((len(all_ts), len(z), len(y), nbins_total), dtype=np.float32)
    south_aerosol = np.zeros((len(all_ts), len(z), len(x), nbins_total), dtype=np.float32)
    north_aerosol = np.zeros((len(all_ts), len(z), len(x), nbins_total), dtype=np.float32)
    top_aerosol = np.zeros((len(all_ts), len(y), len(x), nbins_total), dtype=np.float32)
    
    left_mass_orig = np.zeros((len(all_ts), len(z), len(y), n_species), dtype=np.float32)
    right_mass_orig = np.zeros((len(all_ts), len(z), len(y), n_species), dtype=np.float32)
    south_mass_orig = np.zeros((len(all_ts), len(z), len(x), n_species), dtype=np.float32)
    north_mass_orig = np.zeros((len(all_ts), len(z), len(x), n_species), dtype=np.float32)
    top_mass_orig = np.zeros((len(all_ts), len(y), len(x), n_species), dtype=np.float32)
    
    # ----- West/East Boundaries -----
    print("  Processing West/East boundaries...")
    for ts in tqdm(range(len(all_ts)), desc="  West/East", leave=False):
        current_time = all_ts[ts]
        wrf_time_idx = np.argmin(np.abs(ds_wrf.time.values - current_time))
        
        alt_we_data = alt_wrf.isel(time=wrf_time_idx).isel(
            west_east=slice(west_idx, east_idx + 1),
            south_north=slice(south_idx, north_idx + 1)
        ).mean().values
        
        air_density_we = get_air_density(alt_we_data, f"ALT_we_t{ts}")
        
        for zlev in range(len(z)):
            wrf_num_left = np.zeros((len(y), n_wrf_bins), dtype=np.float32)
            wrf_num_right = np.zeros((len(y), n_wrf_bins), dtype=np.float32)
            
            for wbin, bin_suffix in enumerate(WRFCHEM_BIN_SUFFIXES):
                num_var = f'num{bin_suffix}'
                if num_var in ds_palm_we.data_vars:
                    wrf_num_left[:, wbin] = ds_palm_we[num_var].isel(time=ts, z=zlev, x=0).data * air_density_we
                    wrf_num_right[:, wbin] = ds_palm_we[num_var].isel(time=ts, z=zlev, x=-1).data * air_density_we
            
            left_conc_10bin = vectorized_batch_aerosol_mapping(wrf_num_left, overlap_ratio)
            right_conc_10bin = vectorized_batch_aerosol_mapping(wrf_num_right, overlap_ratio)
            
            if nf2a < 1.0:
                left_conc_1 = left_conc_10bin[:, :n_bins_1]
                left_conc_2 = left_conc_10bin[:, n_bins_1:]
                left_aerosol[ts, zlev, :, :] = np.concatenate([
                    left_conc_1, left_conc_2 * nf2a, left_conc_2 * (1.0 - nf2a)
                ], axis=1)
                
                right_conc_1 = right_conc_10bin[:, :n_bins_1]
                right_conc_2 = right_conc_10bin[:, n_bins_1:]
                right_aerosol[ts, zlev, :, :] = np.concatenate([
                    right_conc_1, right_conc_2 * nf2a, right_conc_2 * (1.0 - nf2a)
                ], axis=1)
            else:
                left_aerosol[ts, zlev, :, :] = left_conc_10bin
                right_aerosol[ts, zlev, :, :] = right_conc_10bin
            
            for idx, spec in enumerate(listspec):
                if is_trace_metal(spec):
                    continue
                wrf_names = get_wrfchem_variables_for_species(spec)
                for wrf_name in wrf_names:
                    for bin_suffix in WRFCHEM_BIN_SUFFIXES:
                        var_name = f'{wrf_name}{bin_suffix}'
                        if var_name in ds_palm_we.data_vars:
                            left_mass_orig[ts, zlev, :, idx] += ds_palm_we[var_name].isel(
                                time=ts, z=zlev, x=0).data * 1e-9 * air_density_we
                            right_mass_orig[ts, zlev, :, idx] += ds_palm_we[var_name].isel(
                                time=ts, z=zlev, x=-1).data * 1e-9 * air_density_we
            
            if 'PM2_5_DRY' in ds_palm_we.data_vars:
                pm25_left = ds_palm_we['PM2_5_DRY'].isel(time=ts, z=zlev, x=0).data * 1e-9 * air_density_we
                pm25_right = ds_palm_we['PM2_5_DRY'].isel(time=ts, z=zlev, x=-1).data * 1e-9 * air_density_we
                
                for idx, spec in enumerate(listspec):
                    if is_trace_metal(spec):
                        frac_left = get_trace_metal_mass_fraction(spec, street_type_we[:, 0])
                        frac_right = get_trace_metal_mass_fraction(spec, street_type_we[:, 1])
                        left_mass_orig[ts, zlev, :, idx] = pm25_left * frac_left
                        right_mass_orig[ts, zlev, :, idx] = pm25_right * frac_right
    
    # ----- South/North Boundaries -----
    print("  Processing South/North boundaries...")
    for ts in tqdm(range(len(all_ts)), desc="  South/North", leave=False):
        current_time = all_ts[ts]
        wrf_time_idx = np.argmin(np.abs(ds_wrf.time.values - current_time))
        
        alt_sn_data = alt_wrf.isel(time=wrf_time_idx).isel(
            west_east=slice(west_idx, east_idx + 1),
            south_north=slice(south_idx, north_idx + 1)
        ).mean().values
        
        air_density_sn = get_air_density(alt_sn_data, f"ALT_sn_t{ts}")
        
        for zlev in range(len(z)):
            wrf_num_south = np.zeros((len(x), n_wrf_bins), dtype=np.float32)
            wrf_num_north = np.zeros((len(x), n_wrf_bins), dtype=np.float32)
            
            for wbin, bin_suffix in enumerate(WRFCHEM_BIN_SUFFIXES):
                num_var = f'num{bin_suffix}'
                if num_var in ds_palm_sn.data_vars:
                    wrf_num_south[:, wbin] = ds_palm_sn[num_var].isel(time=ts, z=zlev, y=0).data * air_density_sn
                    wrf_num_north[:, wbin] = ds_palm_sn[num_var].isel(time=ts, z=zlev, y=-1).data * air_density_sn
            
            south_conc_10bin = vectorized_batch_aerosol_mapping(wrf_num_south, overlap_ratio)
            north_conc_10bin = vectorized_batch_aerosol_mapping(wrf_num_north, overlap_ratio)
            
            if nf2a < 1.0:
                south_conc_1 = south_conc_10bin[:, :n_bins_1]
                south_conc_2 = south_conc_10bin[:, n_bins_1:]
                south_aerosol[ts, zlev, :, :] = np.concatenate([
                    south_conc_1, south_conc_2 * nf2a, south_conc_2 * (1.0 - nf2a)
                ], axis=1)
                
                north_conc_1 = north_conc_10bin[:, :n_bins_1]
                north_conc_2 = north_conc_10bin[:, n_bins_1:]
                north_aerosol[ts, zlev, :, :] = np.concatenate([
                    north_conc_1, north_conc_2 * nf2a, north_conc_2 * (1.0 - nf2a)
                ], axis=1)
            else:
                south_aerosol[ts, zlev, :, :] = south_conc_10bin
                north_aerosol[ts, zlev, :, :] = north_conc_10bin
            
            for idx, spec in enumerate(listspec):
                if is_trace_metal(spec):
                    continue
                wrf_names = get_wrfchem_variables_for_species(spec)
                for wrf_name in wrf_names:
                    for bin_suffix in WRFCHEM_BIN_SUFFIXES:
                        var_name = f'{wrf_name}{bin_suffix}'
                        if var_name in ds_palm_sn.data_vars:
                            south_mass_orig[ts, zlev, :, idx] += ds_palm_sn[var_name].isel(
                                time=ts, z=zlev, y=0).data * 1e-9 * air_density_sn
                            north_mass_orig[ts, zlev, :, idx] += ds_palm_sn[var_name].isel(
                                time=ts, z=zlev, y=-1).data * 1e-9 * air_density_sn
            
            if 'PM2_5_DRY' in ds_palm_sn.data_vars:
                pm25_south = ds_palm_sn['PM2_5_DRY'].isel(time=ts, z=zlev, y=0).data * 1e-9 * air_density_sn
                pm25_north = ds_palm_sn['PM2_5_DRY'].isel(time=ts, z=zlev, y=-1).data * 1e-9 * air_density_sn
                
                for idx, spec in enumerate(listspec):
                    if is_trace_metal(spec):
                        frac_south = get_trace_metal_mass_fraction(spec, street_type_sn[0, :])
                        frac_north = get_trace_metal_mass_fraction(spec, street_type_sn[1, :])
                        south_mass_orig[ts, zlev, :, idx] = pm25_south * frac_south
                        north_mass_orig[ts, zlev, :, idx] = pm25_north * frac_north
    
    # ----- Top Boundary -----
    print("  Processing Top boundary...")
    for ts in tqdm(range(len(all_ts)), desc="  Top", leave=False):
        current_time = all_ts[ts]
        wrf_time_idx = np.argmin(np.abs(ds_wrf.time.values - current_time))
        
        alt_top_data = alt_wrf.isel(time=wrf_time_idx).isel(
            west_east=slice(west_idx, east_idx + 1),
            south_north=slice(south_idx, north_idx + 1)
        ).mean().values
        
        air_density_top = get_air_density(alt_top_data, f"ALT_top_t{ts}")
        
        wrf_num_top = np.zeros((len(y), len(x), n_wrf_bins), dtype=np.float32)
        for wbin, bin_suffix in enumerate(WRFCHEM_BIN_SUFFIXES):
            num_var = f'num{bin_suffix}'
            if num_var in chem_top:
                wrf_num_top[:, :, wbin] = chem_top[num_var][ts, :, :] * air_density_top
        
        wrf_num_top_flat = wrf_num_top.reshape(-1, n_wrf_bins)
        top_conc_10bin_flat = vectorized_batch_aerosol_mapping(wrf_num_top_flat, overlap_ratio)
        top_conc_10bin = top_conc_10bin_flat.reshape(len(y), len(x), -1)
        
        if nf2a < 1.0:
            top_conc_1 = top_conc_10bin[:, :, :n_bins_1]
            top_conc_2 = top_conc_10bin[:, :, n_bins_1:]
            top_aerosol[ts, :, :, :] = np.concatenate([
                top_conc_1, top_conc_2 * nf2a, top_conc_2 * (1.0 - nf2a)
            ], axis=2)
        else:
            top_aerosol[ts, :, :, :] = top_conc_10bin
        
        for idx, spec in enumerate(listspec):
            if is_trace_metal(spec):
                continue
            wrf_names = get_wrfchem_variables_for_species(spec)
            for wrf_name in wrf_names:
                for bin_suffix in WRFCHEM_BIN_SUFFIXES:
                    var_name = f'{wrf_name}{bin_suffix}'
                    if var_name in chem_top:
                        top_mass_orig[ts, :, :, idx] += chem_top[var_name][ts, :, :] * 1e-9 * air_density_top
        
        if 'PM2_5_DRY' in chem_top:
            pm25_top = chem_top['PM2_5_DRY'][ts, :, :] * 1e-9 * air_density_top
            for idx, spec in enumerate(listspec):
                if is_trace_metal(spec):
                    frac_top = get_trace_metal_mass_fraction(spec, street_type_surface)
                    top_mass_orig[ts, :, :, idx] = pm25_top * frac_top
    
    # ===== STEP 3: Normalize Boundary Mass Fractions =====
    print("\n" + "-"*40)
    print("STEP 3: Normalizing boundary mass fractions...")
    print("-"*40)
    
    def normalize_boundary_mass(mass_array):
        total = np.sum(mass_array, axis=-1, keepdims=True)
        total = np.where(total < 1e-30, 1.0, total)
        return mass_array / total
    
    left_mass_norm = normalize_boundary_mass(left_mass_orig)
    right_mass_norm = normalize_boundary_mass(right_mass_orig)
    south_mass_norm = normalize_boundary_mass(south_mass_orig)
    north_mass_norm = normalize_boundary_mass(north_mass_orig)
    top_mass_norm = normalize_boundary_mass(top_mass_orig)
    
    # Separate into a and b fractions
    left_mass_a = np.zeros_like(left_mass_norm)
    left_mass_b = np.zeros_like(left_mass_norm)
    right_mass_a = np.zeros_like(right_mass_norm)
    right_mass_b = np.zeros_like(right_mass_norm)
    south_mass_a = np.zeros_like(south_mass_norm)
    south_mass_b = np.zeros_like(south_mass_norm)
    north_mass_a = np.zeros_like(north_mass_norm)
    north_mass_b = np.zeros_like(north_mass_norm)
    top_mass_a = np.zeros_like(top_mass_norm)
    top_mass_b = np.zeros_like(top_mass_norm)
    
    for ts in tqdm(range(len(all_ts)), desc="  Separating mass fractions", leave=False):
        for zlev in range(len(z)):
            la, lb = create_separated_mass_fractions(left_mass_norm[ts, zlev, :, :], listspec, nf2a)
            left_mass_a[ts, zlev, :, :] = la
            left_mass_b[ts, zlev, :, :] = lb
            
            ra, rb = create_separated_mass_fractions(right_mass_norm[ts, zlev, :, :], listspec, nf2a)
            right_mass_a[ts, zlev, :, :] = ra
            right_mass_b[ts, zlev, :, :] = rb
            
            sa, sb = create_separated_mass_fractions(south_mass_norm[ts, zlev, :, :], listspec, nf2a)
            south_mass_a[ts, zlev, :, :] = sa
            south_mass_b[ts, zlev, :, :] = sb
            
            na, nb = create_separated_mass_fractions(north_mass_norm[ts, zlev, :, :], listspec, nf2a)
            north_mass_a[ts, zlev, :, :] = na
            north_mass_b[ts, zlev, :, :] = nb
    
    for ts in tqdm(range(len(all_ts)), desc="  Top separation", leave=False):
        flat_mass = top_mass_norm[ts, :, :, :].reshape(-1, n_species)
        flat_a, flat_b = create_separated_mass_fractions(flat_mass, listspec, nf2a)
        top_mass_a[ts, :, :, :] = flat_a.reshape(top_mass_norm[ts].shape)
        top_mass_b[ts, :, :, :] = flat_b.reshape(top_mass_norm[ts].shape)
    
    # Store boundary data
    aerosol_boundary_data = {
        'left': left_aerosol, 'right': right_aerosol,
        'south': south_aerosol, 'north': north_aerosol, 'top': top_aerosol,
        'left_mass_a': left_mass_a, 'right_mass_a': right_mass_a,
        'south_mass_a': south_mass_a, 'north_mass_a': north_mass_a, 'top_mass_a': top_mass_a,
        'left_mass_b': left_mass_b, 'right_mass_b': right_mass_b,
        'south_mass_b': south_mass_b, 'north_mass_b': north_mass_b, 'top_mass_b': top_mass_b
    }
    
    print("\n" + "="*60)
    print("AEROSOL PROCESSING COMPLETE - Verification")
    print("="*60)
    print(f"\n  Mass fraction checks:")
    print(f"    mass_fracs_a sum (init): {np.mean(np.sum(mass_fracs_a_init, axis=1)):.4f}")
    print(f"    mass_fracs_b sum (init): {np.mean(np.sum(mass_fracs_b_init, axis=1)):.4f}")
    print(f"    mass_fracs_a sum (top, t=0): {np.mean(np.sum(top_mass_a[0, :, :, :], axis=-1)):.4f}")
    print(f"    mass_fracs_b sum (top, t=0): {np.mean(np.sum(top_mass_b[0, :, :, :], axis=-1)):.4f}")
    print("="*60)

#===============================================================================
# Process Radiation Data
#===============================================================================
rad_times_sec = []
rad_values_proc = [[], [], []]

if radiation_from_wrf:
    print("\n" + "="*60)
    print("PROCESSING RADIATION DATA FROM WRF")
    print("="*60)
    
    radiation_vars_exist = all(var in ds_wrf.variables for var in ['SWDOWN', 'GLW', 'SWDDIF'])
    
    if radiation_vars_exist:
        rad_times_sec = times_sec
        rad_swdown, rad_lwdown, rad_swdiff = [], [], []
        wrf_times = ds_wrf.time.values
        
        for ts in tqdm(range(len(all_ts)), desc="Radiation"):
            current_time = all_ts[ts]
            closest_idx = np.argmin(np.abs(wrf_times - current_time))
            
            sw_cropped = ds_wrf['SWDOWN'].isel(
                time=closest_idx,
                west_east=slice(west_idx, east_idx + 1),
                south_north=slice(south_idx, north_idx + 1)
            ).values.astype(np.float32)
            
            lw_cropped = ds_wrf['GLW'].isel(
                time=closest_idx,
                west_east=slice(west_idx, east_idx + 1),
                south_north=slice(south_idx, north_idx + 1)
            ).values.astype(np.float32)
            
            dif_cropped = ds_wrf['SWDDIF'].isel(
                time=closest_idx,
                west_east=slice(west_idx, east_idx + 1),
                south_north=slice(south_idx, north_idx + 1)
            ).values.astype(np.float32)
            
            rad_swdown.append(np.mean(sw_cropped))
            rad_lwdown.append(np.mean(lw_cropped))
            rad_swdiff.append(np.mean(dif_cropped))
        
        rad_values_proc = [rad_swdown, rad_lwdown, rad_swdiff]
        print("Radiation data processed successfully")

#===============================================================================
# Output NetCDF File
#===============================================================================
print("\nPreparing NetCDF file...")

nc_output_name = f'dynamic_files/{case_name}_dynamic_{start_year}_{start_month}_{start_day}_{start_hour}'
nc_output = xr.Dataset()

nc_output.attrs['description'] = f'Contains dynamic data from WRF mesoscale. WRF output file: {wrf_file}'
nc_output.attrs['author'] = 'Meteorology: Dongqi Lin; Chemistry, Aerosol and Radiation: Sathish Kumar Vaithiyanadhan'
nc_output.attrs['institution'] = 'Chair of Model-based Environmental Exposure Science, University of Augsburg'
nc_output.attrs['history'] = 'Created at ' + time.ctime(time.time())
nc_output.attrs['source'] = 'netCDF4 python'
nc_output.attrs['origin_lat'] = float(centlat)
nc_output.attrs['origin_lon'] = float(centlon)
nc_output.attrs['z'] = float(0)
nc_output.attrs['x'] = float(0)
nc_output.attrs['y'] = float(0)
nc_output.attrs['rotation_angle'] = float(0)
nc_output.attrs['origin_time'] = str(all_ts[0]) + ' UTC'
nc_output.attrs['end_time'] = str(all_ts[-1]) + ' UTC'

nc_output['x'] = xr.DataArray(x, dims=['x'], attrs={'units': 'm'})
nc_output['y'] = xr.DataArray(y, dims=['y'], attrs={'units': 'm'})
nc_output['z'] = xr.DataArray(z - z_origin, dims=['z'], attrs={'units': 'm'})
nc_output['zsoil'] = xr.DataArray(dz_soil, dims=['zsoil'], attrs={'units': 'm'})
nc_output['xu'] = xr.DataArray(xu, dims=['xu'], attrs={'units': 'm'})
nc_output['yv'] = xr.DataArray(yv, dims=['yv'], attrs={'units': 'm'})
nc_output['zw'] = xr.DataArray(zw - z_origin, dims=['zw'], attrs={'units': 'm'})
nc_output['time'] = xr.DataArray(times_sec, dims=['time'], attrs={'units': 'seconds'})

# Soil variables
nc_output['init_soil_m'] = xr.DataArray(init_msoil.astype(np.float32), dims=['zsoil', 'y', 'x'],
    attrs={'units': 'm3/m3', 'source': 'WRF', 'long_name': 'volumetric soil moisture', 'lod': np.int32(2)})
nc_output['init_soil_t'] = xr.DataArray(init_tsoil.astype(np.float32), dims=['zsoil', 'y', 'x'],
    attrs={'units': 'K', 'source': 'WRF', 'long_name': 'soil temperature', 'lod': np.int32(2)})

# Meteorological variables
met_vars_output = [
    ('init_atmosphere_pt', pt_init.astype(np.float32), ['z'], {'units': 'K', 'lod': np.int32(1)}),
    ('ls_forcing_left_pt', ds_palm_we["pt"][:, :, :, 0].data.astype(np.float32), ['time', 'z', 'y'], {'units': 'K'}),
    ('ls_forcing_right_pt', ds_palm_we["pt"][:, :, :, -1].data.astype(np.float32), ['time', 'z', 'y'], {'units': 'K'}),
    ('ls_forcing_south_pt', ds_palm_sn["pt"][:, :, 0, :].data.astype(np.float32), ['time', 'z', 'x'], {'units': 'K'}),
    ('ls_forcing_north_pt', ds_palm_sn["pt"][:, :, -1, :].data.astype(np.float32), ['time', 'z', 'x'], {'units': 'K'}),
    ('ls_forcing_top_pt', pt_top.astype(np.float32), ['time', 'y', 'x'], {'units': 'K'}),
    ('init_atmosphere_qv', qv_init.astype(np.float32), ['z'], {'units': 'kg/kg', 'lod': np.int32(1)}),
    ('ls_forcing_left_qv', ds_palm_we["QVAPOR"][:, :, :, 0].data.astype(np.float32), ['time', 'z', 'y'], {'units': 'kg/kg'}),
    ('ls_forcing_right_qv', ds_palm_we["QVAPOR"][:, :, :, -1].data.astype(np.float32), ['time', 'z', 'y'], {'units': 'kg/kg'}),
    ('ls_forcing_south_qv', ds_palm_sn["QVAPOR"][:, :, 0, :].data.astype(np.float32), ['time', 'z', 'x'], {'units': 'kg/kg'}),
    ('ls_forcing_north_qv', ds_palm_sn["QVAPOR"][:, :, -1, :].data.astype(np.float32), ['time', 'z', 'x'], {'units': 'kg/kg'}),
    ('ls_forcing_top_qv', qv_top.astype(np.float32), ['time', 'y', 'x'], {'units': 'kg/kg'}),
    ('init_atmosphere_u', u_init.astype(np.float32), ['z'], {'units': 'm/s', 'lod': np.int32(1)}),
    ('ls_forcing_left_u', ds_palm_we["U"][:, :, :, 0].data.astype(np.float32), ['time', 'z', 'y'], {'units': 'm/s'}),
    ('ls_forcing_right_u', ds_palm_we["U"][:, :, :, -1].data.astype(np.float32), ['time', 'z', 'y'], {'units': 'm/s'}),
    ('ls_forcing_south_u', ds_palm_sn["U"][:, :, 0, :].data.astype(np.float32), ['time', 'z', 'xu'], {'units': 'm/s'}),
    ('ls_forcing_north_u', ds_palm_sn["U"][:, :, -1, :].data.astype(np.float32), ['time', 'z', 'xu'], {'units': 'm/s'}),
    ('ls_forcing_top_u', u_top.astype(np.float32), ['time', 'y', 'xu'], {'units': 'm/s'}),
    ('init_atmosphere_v', v_init.astype(np.float32), ['z'], {'units': 'm/s', 'lod': np.int32(1)}),
    ('ls_forcing_left_v', ds_palm_we["V"][:, :, :, 0].data.astype(np.float32), ['time', 'z', 'yv'], {'units': 'm/s'}),
    ('ls_forcing_right_v', ds_palm_we["V"][:, :, :, -1].data.astype(np.float32), ['time', 'z', 'yv'], {'units': 'm/s'}),
    ('ls_forcing_south_v', ds_palm_sn["V"][:, :, 0, :].data.astype(np.float32), ['time', 'z', 'x'], {'units': 'm/s'}),
    ('ls_forcing_north_v', ds_palm_sn["V"][:, :, -1, :].data.astype(np.float32), ['time', 'z', 'x'], {'units': 'm/s'}),
    ('ls_forcing_top_v', v_top.astype(np.float32), ['time', 'yv', 'x'], {'units': 'm/s'}),
    ('init_atmosphere_w', w_init.astype(np.float32), ['zw'], {'units': 'm/s', 'lod': np.int32(1)}),
    ('ls_forcing_left_w', ds_palm_we["W"][:, :, :, 0].data.astype(np.float32), ['time', 'zw', 'y'], {'units': 'm/s'}),
    ('ls_forcing_right_w', ds_palm_we["W"][:, :, :, -1].data.astype(np.float32), ['time', 'zw', 'y'], {'units': 'm/s'}),
    ('ls_forcing_south_w', ds_palm_sn["W"][:, :, 0, :].data.astype(np.float32), ['time', 'zw', 'x'], {'units': 'm/s'}),
    ('ls_forcing_north_w', ds_palm_sn["W"][:, :, -1, :].data.astype(np.float32), ['time', 'zw', 'x'], {'units': 'm/s'}),
    ('ls_forcing_top_w', w_top.astype(np.float32), ['time', 'y', 'x'], {'units': 'm/s'}),
    ('surface_forcing_surface_pressure', surface_pres.data.astype(np.float32), ['time'], {'units': 'Pa', 'lod': np.int32(1)}),
]

for var_name, data, dims, attrs in met_vars_output:
    nc_output[var_name] = xr.DataArray(data, dims=dims, attrs=attrs)

# Chemistry species output
MICROGRAM_TO_KG = 1e-9
chem_name_mapping = {
    "hno3": "HNO3", "ho2": "HO2", "ho": "OH", "no2": "NO2", "o3": "O3",
    "no": "NO", "nh3": "NH3", "so2": "SO2", "co": "CO", "sulf": "H2SO4",
    "RH": "RH", "RO2": "RO2", "RCHO": "RCHO", "PM10": "PM10", "PM2_5_DRY": "PM25"
}

for species in original_chem_species:
    if species.endswith('_tra'):
        base = species.replace('_tra', '')
        if base in chem_name_mapping:
            output_name = f"{chem_name_mapping[base]}_tra"
        else:
            output_name = f"{base.upper()}_tra"
    else:
        output_name = chem_name_mapping.get(species, species.upper())
    
    if species in chem_init:
        if species in ['PM10', 'PM2_5_DRY'] or species.replace('_tra', '') in ['PM10', 'PM2_5_DRY']:
            data = chem_init[species].data * MICROGRAM_TO_KG
            units = 'kg/m3'
        else:
            data = chem_init[species].data
            units = 'ppm'
        
        nc_output[f'init_atmosphere_{output_name}'] = xr.DataArray(data.astype(np.float32), dims=['z'], 
            attrs={'units': units, 'source': 'WRF-Chem', 'lod': np.int32(1)})
    
    if species in ds_palm_we.data_vars:
        if species in ['PM10', 'PM2_5_DRY'] or species.replace('_traffic', '') in ['PM10', 'PM2_5_DRY']:
            left_data = ds_palm_we[species][:, :, :, 0].data * MICROGRAM_TO_KG
            right_data = ds_palm_we[species][:, :, :, -1].data * MICROGRAM_TO_KG
            south_data = ds_palm_sn[species][:, :, 0, :].data * MICROGRAM_TO_KG
            north_data = ds_palm_sn[species][:, :, -1, :].data * MICROGRAM_TO_KG
            top_data = chem_top[species] * MICROGRAM_TO_KG
            units = 'kg/m3'
        else:
            left_data = ds_palm_we[species][:, :, :, 0].data
            right_data = ds_palm_we[species][:, :, :, -1].data
            south_data = ds_palm_sn[species][:, :, 0, :].data
            north_data = ds_palm_sn[species][:, :, -1, :].data
            top_data = chem_top[species]
            units = 'ppm'
        
        nc_output[f'ls_forcing_left_{output_name}'] = xr.DataArray(left_data.astype(np.float32), dims=['time', 'z', 'y'], 
            attrs={'units': units, 'source': 'WRF-Chem'})
        nc_output[f'ls_forcing_right_{output_name}'] = xr.DataArray(right_data.astype(np.float32), dims=['time', 'z', 'y'], 
            attrs={'units': units, 'source': 'WRF-Chem'})
        nc_output[f'ls_forcing_south_{output_name}'] = xr.DataArray(south_data.astype(np.float32), dims=['time', 'z', 'x'], 
            attrs={'units': units, 'source': 'WRF-Chem'})
        nc_output[f'ls_forcing_north_{output_name}'] = xr.DataArray(north_data.astype(np.float32), dims=['time', 'z', 'x'], 
            attrs={'units': units, 'source': 'WRF-Chem'})
        nc_output[f'ls_forcing_top_{output_name}'] = xr.DataArray(top_data.astype(np.float32), dims=['time', 'y', 'x'], 
            attrs={'units': units, 'source': 'WRF-Chem'})

# ===== AEROSOL OUTPUT =====
if aerosol_wrfchem and 'aerosol_boundary_data' in locals():
    print("\nAdding aerosol variables to output...")
    
    nbins = nbins_total
    n_species = len(listspec)
    max_string_length = 25
    
    char_array = np.zeros((n_species, max_string_length), dtype='S1')
    for i, name in enumerate(listspec):
        name_bytes = name.encode('utf-8')
        name_len = min(len(name_bytes), max_string_length)
        for j in range(name_len):
            char_array[i, j] = name_bytes[j:j+1]
    
    nc_output['composition_name'] = (
        ['composition_index', 'max_string_length'],
        char_array,
        {'long_name': 'aerosol species names', 'units': '-'}
    )
    
    if 'mass_fracs_a' in chem_init:
        nc_output['init_atmosphere_mass_fracs_a'] = (
            ['z', 'composition_index'],
            chem_init['mass_fracs_a'].data.astype(np.float32),
            {'units': '', 'source': 'WRF-Chem', 'long_name': 'soluble mass fractions (Mode A)', 'lod': np.int32(1)}
        )
    
    if 'mass_fracs_b' in chem_init:
        nc_output['init_atmosphere_mass_fracs_b'] = (
            ['z', 'composition_index'],
            chem_init['mass_fracs_b'].data.astype(np.float32),
            {'units': '', 'source': 'WRF-Chem', 'long_name': 'insoluble mass fractions (Mode B)', 'lod': np.int32(1)}
        )
    
    if 'aerosol' in chem_init:
        nc_output['init_atmosphere_aerosol'] = (
            ['z', 'Dmid'],
            chem_init['aerosol'].data.astype(np.float32),
            {'units': '#/m3', 'source': 'WRF-Chem', 'long_name': 'aerosol number concentration', 'lod': np.int32(1)}
        )
    
    nc_output['ls_forcing_left_aerosol'] = (
        ['time', 'z', 'y', 'Dmid'],
        aerosol_boundary_data['left'].astype(np.float32),
        {'units': '#/m3', 'long_name': 'aerosol number concentration - west boundary'}
    )
    nc_output['ls_forcing_right_aerosol'] = (
        ['time', 'z', 'y', 'Dmid'],
        aerosol_boundary_data['right'].astype(np.float32),
        {'units': '#/m3', 'long_name': 'aerosol number concentration - east boundary'}
    )
    nc_output['ls_forcing_south_aerosol'] = (
        ['time', 'z', 'x', 'Dmid'],
        aerosol_boundary_data['south'].astype(np.float32),
        {'units': '#/m3', 'long_name': 'aerosol number concentration - south boundary'}
    )
    nc_output['ls_forcing_north_aerosol'] = (
        ['time', 'z', 'x', 'Dmid'],
        aerosol_boundary_data['north'].astype(np.float32),
        {'units': '#/m3', 'long_name': 'aerosol number concentration - north boundary'}
    )
    nc_output['ls_forcing_top_aerosol'] = (
        ['time', 'y', 'x', 'Dmid'],
        aerosol_boundary_data['top'].astype(np.float32),
        {'units': '#/m3', 'long_name': 'aerosol number concentration - top boundary'}
    )
    
    if 'left_mass_a' in aerosol_boundary_data and np.any(aerosol_boundary_data['left_mass_a']):
        nc_output['ls_forcing_left_mass_fracs_a'] = (
            ['time', 'z', 'y', 'composition_index'],
            aerosol_boundary_data['left_mass_a'].astype(np.float32),
            {'units': '', 'long_name': 'soluble mass fractions - west boundary'}
        )
        nc_output['ls_forcing_right_mass_fracs_a'] = (
            ['time', 'z', 'y', 'composition_index'],
            aerosol_boundary_data['right_mass_a'].astype(np.float32),
            {'units': '', 'long_name': 'soluble mass fractions - east boundary'}
        )
        nc_output['ls_forcing_south_mass_fracs_a'] = (
            ['time', 'z', 'x', 'composition_index'],
            aerosol_boundary_data['south_mass_a'].astype(np.float32),
            {'units': '', 'long_name': 'soluble mass fractions - south boundary'}
        )
        nc_output['ls_forcing_north_mass_fracs_a'] = (
            ['time', 'z', 'x', 'composition_index'],
            aerosol_boundary_data['north_mass_a'].astype(np.float32),
            {'units': '', 'long_name': 'soluble mass fractions - north boundary'}
        )
        nc_output['ls_forcing_top_mass_fracs_a'] = (
            ['time', 'y', 'x', 'composition_index'],
            aerosol_boundary_data['top_mass_a'].astype(np.float32),
            {'units': '', 'long_name': 'soluble mass fractions - top boundary'}
        )
    
    if 'left_mass_b' in aerosol_boundary_data and np.any(aerosol_boundary_data['left_mass_b']):
        nc_output['ls_forcing_left_mass_fracs_b'] = (
            ['time', 'z', 'y', 'composition_index'],
            aerosol_boundary_data['left_mass_b'].astype(np.float32),
            {'units': '', 'long_name': 'insoluble mass fractions - west boundary'}
        )
        nc_output['ls_forcing_right_mass_fracs_b'] = (
            ['time', 'z', 'y', 'composition_index'],
            aerosol_boundary_data['right_mass_b'].astype(np.float32),
            {'units': '', 'long_name': 'insoluble mass fractions - east boundary'}
        )
        nc_output['ls_forcing_south_mass_fracs_b'] = (
            ['time', 'z', 'x', 'composition_index'],
            aerosol_boundary_data['south_mass_b'].astype(np.float32),
            {'units': '', 'long_name': 'insoluble mass fractions - south boundary'}
        )
        nc_output['ls_forcing_north_mass_fracs_b'] = (
            ['time', 'z', 'x', 'composition_index'],
            aerosol_boundary_data['north_mass_b'].astype(np.float32),
            {'units': '', 'long_name': 'insoluble mass fractions - north boundary'}
        )
        nc_output['ls_forcing_top_mass_fracs_b'] = (
            ['time', 'y', 'x', 'composition_index'],
            aerosol_boundary_data['top_mass_b'].astype(np.float32),
            {'units': '', 'long_name': 'insoluble mass fractions - top boundary'}
        )
    
    nc_output = nc_output.assign_coords(
        Dmid=('Dmid', dmid_all.astype(np.float32)),
        composition_index=('composition_index', np.arange(1, n_species + 1, dtype=np.int32)),
        max_string_length=('max_string_length', np.arange(1, max_string_length + 1, dtype=np.int32))
    )
    
    nc_output['Dmid'].attrs = {
        'units': 'm',
        'long_name': 'geometric mean diameter of aerosol size bin',
        'bin_structure': f'{n_bins_1} subrange 1 + {n_bins_2} subrange 2a + {n_bins_2} subrange 2b',
        'nf2a': str(nf2a)
    }
    nc_output['composition_index'].attrs = {'long_name': 'aerosol species index'}
    nc_output['max_string_length'].attrs = {'long_name': 'maximum string length'}

# Radiation output
if len(rad_times_sec) > 0 and len(rad_values_proc[0]) > 0:
    nc_output['time_rad'] = xr.DataArray(np.array(rad_times_sec, dtype=np.float32), dims=['time_rad'], 
        attrs={'units': 'seconds', 'long_name': 'time for radiation data'})
    nc_output['rad_sw_in'] = xr.DataArray(np.array(rad_values_proc[0], dtype=np.float32), dims=['time_rad'],
        attrs={'units': 'W/m2', 'long_name': 'incoming shortwave radiation', 'lod': np.int32(1)})
    nc_output['rad_lw_in'] = xr.DataArray(np.array(rad_values_proc[1], dtype=np.float32), dims=['time_rad'],
        attrs={'units': 'W/m2', 'long_name': 'incoming longwave radiation', 'lod': np.int32(1)})
    nc_output['rad_sw_in_dif'] = xr.DataArray(np.array(rad_values_proc[2], dtype=np.float32), dims=['time_rad'],
        attrs={'units': 'W/m2', 'long_name': 'incoming diffuse shortwave radiation', 'lod': np.int32(1)})

# ===== WRITE OUTPUT =====
print("Writing all variables to NetCDF file...")
nc_output.to_netcdf(nc_output_name)
print(f"File saved as: {nc_output_name}")

print("\n" + "="*60)
print('Add to your *_p3d file:')
print(f' soil_temperature = {[float(v) for v in init_tsoil.mean(axis=(1, 2))]}')
print(f' soil_moisture = {[float(v) for v in init_msoil.mean(axis=(1, 2))]}')
print(f' deep_soil_temperature = {float(deep_tsoil)}')
print("="*60)

with open('cfg_files/' + case_name + '.cfg', "a") as cfg:
    cfg.write(f'Add to your *_p3d file:\n soil_temperature = {[float(v) for v in init_tsoil.mean(axis=(1, 2))]}\n')
    cfg.write(f' soil_moisture = {[float(v) for v in init_msoil.mean(axis=(1, 2))]}\n')
    cfg.write(f' deep_soil_temperature = {float(deep_tsoil)}\n')

# ===== POST-PROCESSING FOR AEROSOL VARIABLES =====
if aerosol_wrfchem:
    print("="*60)
    print("POST-PROCESSING: Reformatting aerosol variables in NetCDF file")
    print("="*60)
    
    nc_output.close()
    
    import netCDF4 as nc4
    
    temp_output = nc_output_name + '_temp.nc'
    os.rename(nc_output_name, temp_output)
    
    with nc4.Dataset(temp_output, 'r') as src:
        with nc4.Dataset(nc_output_name, 'w', format='NETCDF4') as dst:
            for dim_name, dim in src.dimensions.items():
                if dim_name != 'string1':
                    dst.createDimension(dim_name, len(dim) if not dim.isunlimited() else None)
            
            for attr_name in src.ncattrs():
                dst.setncattr(attr_name, src.getncattr(attr_name))
            
            for var_name, var in src.variables.items():
                if var_name != 'composition_name':
                    dims = [d for d in var.dimensions if d != 'string1']
                    fill_value = None
                    if '_FillValue' in var.ncattrs():
                        fill_value = var.getncattr('_FillValue')
                    
                    new_var = dst.createVariable(var_name, var.datatype, dims, fill_value=fill_value)
                    for attr_name in var.ncattrs():
                        if attr_name != '_FillValue':
                            new_var.setncattr(attr_name, var.getncattr(attr_name))
                    
                    if 'string1' in var.dimensions:
                        data = var[:]
                        if data.shape[-1] == 1:
                            data = data[..., 0]
                        new_var[:] = data
                    else:
                        new_var[:] = var[:]
            
            var = src.variables['composition_name']
            fill_value = None
            if '_FillValue' in var.ncattrs():
                fill_value = var.getncattr('_FillValue')
            
            new_dims = ('composition_index', 'max_string_length')
            new_var = dst.createVariable('composition_name', var.datatype, new_dims, fill_value=fill_value)
            for attr_name in var.ncattrs():
                if attr_name != '_FillValue':
                    new_var.setncattr(attr_name, var.getncattr(attr_name))
            
            data = var[:]
            if data.ndim == 3 and data.shape[-1] == 1:
                data = data[:, :, 0]
            new_var[:] = data
    
    os.remove(temp_output)
    print(f"File saved as: {nc_output_name}")
    print("="*60)
else:
    print("\n" + "="*60)
    print("Skipping NetCDF post-processing (aerosol_wrfchem = False)")
    print("="*60)
    nc_output.close()

# ===== FINAL OUTPUT =====
end = datetime.now()
print("="*60)
print(f"PALM dynamic input file is ready!")
print(f"Script duration: {end - start}")
print(f"Start time: {all_ts[0]}")
print(f"End time: {all_ts[-1]}")
print(f"Time step: {times_sec[1] - times_sec[0]} seconds")
print(f"Output file: {nc_output_name}")
print("="*60)