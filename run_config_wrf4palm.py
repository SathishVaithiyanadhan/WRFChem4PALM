# Aerosol, Meteorology, chemistry and radiation
# -*- coding: utf-8 -*-
import sys
import os
import time
import salem
import xarray as xr
from functools import partial
from pyproj import Proj, Transformer
import configparser
import ast
from glob import glob
import numpy as np
from math import ceil, floor
from datetime import datetime, timedelta
from tqdm import tqdm
from multiprocess import Pool
from dynamic_util.nearest import framing_2d_cartesian
from dynamic_util.loc_dom import calc_stretch, domain_location, generate_cfg
from dynamic_util.process_wrf import zinterp, multi_zinterp
from dynamic_util.geostrophic import *
from dynamic_util.surface_nan_solver import *
from dynamic_util.wrfchem_aerosol import (
    AEROSOL_TRANSLATION, WRFCHEM_BIN_SUFFIXES,
    get_wrfchem_variables_for_species, get_all_wrfchem_variables,
    define_bins, aerosol_binoverlap, upwind_location
)
import warnings
## suppress warnings
warnings.filterwarnings("ignore", '.*pyproj.*')
warnings.simplefilter(action='ignore', category=FutureWarning)

#-------------------------------------------------------------------------------
# Function to setup traffic variables
#-------------------------------------------------------------------------------
def setup_traffic_variables(chem_species):
    """
    Check if traffic variables are requested and set up the mapping
    Returns: tuple (has_traffic, traffic_mapping)
    """
    traffic_mapping = {}
    has_traffic = False
    
    for species in chem_species:
        if species.endswith('_traffic'):
            base_species = species.replace('_traffic', '')
            if base_species in ['no', 'no2']:
                traffic_mapping[base_species] = species
                has_traffic = True
                print(f"Traffic variable requested: {species} (based on {base_species})")
    
    return has_traffic, traffic_mapping

start = datetime.now()

if not os.path.exists("./cfg_files"):
    print("cfg_files folder created")
    os.makedirs("./cfg_files")
if not os.path.exists("./dynamic_files"):    
    print("dynamic_files folder created")
    os.makedirs("./dynamic_files")

#--------------------------------------------------------------------------------
# Read user input namelist
#--------------------------------------------------------------------------------
config = configparser.RawConfigParser()
config.read(sys.argv[1])
case_name = ast.literal_eval(config.get("case", "case_name"))[0]
max_pool = ast.literal_eval(config.get("case", "max_pool"))[0]
geostr_lvl = ast.literal_eval(config.get("case", "geostrophic"))[0] 

# Read chemistry species from config
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

#-------------------------------------------------------------------------------
# Check for traffic variables
#-------------------------------------------------------------------------------
has_traffic_vars, traffic_mapping = setup_traffic_variables(chem_species)
original_chem_species = chem_species.copy()
chem_species_for_processing = [s for s in chem_species if not s.endswith('_traffic')]

print(f"Chemistry species for processing: {chem_species_for_processing}")

#-------------------------------------------------------------------------------
# Read aerosol settings from config
#-------------------------------------------------------------------------------
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
        print(f"Aerosol composition list (SALSA): {listspec}")
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

# Read radiation settings
try:
    radiation_from_wrf = ast.literal_eval(config.get("radiation", "radiation_from_wrf"))[0]
except:
    radiation_from_wrf = True

try:
    radiation_smoothing_distance = ast.literal_eval(config.get("radiation", "radiation_smoothing_distance"))[0]
except:
    radiation_smoothing_distance = 10000.0

print(f"Radiation from WRF: {radiation_from_wrf}")

# Define component species for aggregated species
RH_components = ["isopr", "apin", "bpin", "limon", "bcary", "myrc", 
                "benzene", "tol", "xylenes", "bigalk", "bigene", 
                "c2h4", "c3h6"]

RO2_components = ["ch3o2", "aco3", "mco3", "alko2", "aceto2",
                 "eto2", "pro2", "po2", "terpo2", "terp2o2",
                 "nterpo2", "isopao2", "isopbo2", "mdialo2", "dicarbo2"]

RCHO_components = ["ald", "bzald", "glyald", "hydrald", "gly", "mgly", "hcho"]

OCSV_components = ["cvasoa2", "cvasoa3", "cvasoa4", "cvbsoa2", "cvbsoa3", "cvbsoa4"]

OCNV_components = ["cvasoaX", "cvasoa1", "cvbsoaX", "cvbsoa1"]

# Create list of component species
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

# Combine regular chemistry species with component species for processing
all_chem_to_process = list(set(chem_species_for_processing + all_component_species))
all_chem_to_process = [s for s in all_chem_to_process if s not in ["RH", "RO2", "RCHO", "OCSV", "OCNV"]]

# Add aerosol variables to processing list if enabled
if aerosol_wrfchem:
    # Get all WRF-Chem variable names for the species in listspec
    aerosol_vars = get_all_wrfchem_variables(listspec)
    all_chem_to_process.extend(aerosol_vars)
    
    # Add number concentration variables
    for bin_suffix in WRFCHEM_BIN_SUFFIXES:
        all_chem_to_process.append(f'num{bin_suffix}')
    
    print(f"Aerosol mass variables added: {aerosol_vars}")
    print(f"Aerosol number variables added: num_a01, num_a02, num_a03, num_a04")

print(f"All species to process: {all_chem_to_process}")

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

y = np.arange(dy/2, dy*ny + dy/2, dy)
x = np.arange(dx/2, dx*nx + dx/2, dx)
z = np.arange(dz/2, dz*nz, dz)
xu = x + np.gradient(x)/2
xu = xu[:-1]
yv = y + np.gradient(y)/2
yv = yv[:-1]
zw = z + np.gradient(z)/2
zw = zw[:-1]

# Stretch grid if needed
dz_stretch_factor = ast.literal_eval(config.get("stretch", "dz_stretch_factor"))[0]
dz_stretch_level = ast.literal_eval(config.get("stretch", "dz_stretch_level"))[0]
dz_max = ast.literal_eval(config.get("stretch", "dz_max"))[0]

if dz_stretch_factor > 1.0:
    z, zw = calc_stretch(z, dz, zw, dz_stretch_factor, dz_stretch_level, dz_max)

z += z_origin
zw += z_origin

dz_soil = np.array(ast.literal_eval(config.get("soil", "dz_soil")))
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

#-------------------------------------------------------------------------------
# Read WRF
#-------------------------------------------------------------------------------
print("Reading WRF")
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
        ds_wrf[variables] = ds_raw[variables].drop_duplicates("time", keep="last")
    ds_wrf.attrs = ds_raw.attrs
del ds_raw

#-------------------------------------------------------------------------------
# Find timestamps
#-------------------------------------------------------------------------------
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

times_sec = np.zeros(len(all_ts))
for t in range(0, len(all_ts)):
    times_sec[t] = (all_ts[t] - all_ts[0]).astype('float') * 1e-9

#-------------------------------------------------------------------------------
# Locate PALM domain in WRF
#-------------------------------------------------------------------------------
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
ds_drop["pt"] = ds_drop["T"] + 300
ds_drop["pt"].attrs = ds_drop["T"].attrs
ds_drop["gph"] = (ds_drop["PH"] + ds_drop["PHB"]) / 9.81
ds_drop["gph"].attrs = ds_drop["PH"].attrs

#-------------------------------------------------------------------------------
# Horizontal interpolation
#-------------------------------------------------------------------------------
print("Start horizontal interpolation")
south_north_palm = ds_drop.south_north[0].data + y
west_east_palm = ds_drop.west_east[0].data + x
south_north_v_palm = ds_drop.south_north[0].data + yv
west_east_u_palm = ds_drop.west_east[0].data + xu

ds_drop = ds_drop.assign_coords({"west_east_palm": west_east_palm,
                                 "south_north_palm": south_north_palm,
                                 "west_east_u_palm": west_east_u_palm,
                                 "south_north_v_palm": south_north_v_palm})

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

# Get surface and soil fields
zs_wrf = ds_interp.ZS[0, :, 0, 0].load()
t2_wrf = ds_interp.T2.load()
u10_wrf = ds_interp_u.U10.load()
v10_wrf = ds_interp_v.V10.load()
qv2_wrf = ds_interp.Q2.load()
psfc_wrf = ds_interp.PSFC.load()
pt2_wrf = t2_wrf * ((1000) / (psfc_wrf * 0.01)) ** 0.286

surface_var_dict = {"U": u10_wrf, "V": v10_wrf, "pt": pt2_wrf, "QVAPOR": qv2_wrf, "W": None}

#-------------------------------------------------------------------------------
# Soil moisture and temperature
#-------------------------------------------------------------------------------
print("Calculating soil temperature and moisture from WRF")

watermask = ds_interp["LANDMASK"].sel(time=dt_start).load().data == 0
landmask = ds_interp["LANDMASK"].sel(time=dt_start).load().data == 1
median_smois = [np.nanmedian(ds_interp["SMOIS"][0, izs, :, :].load().data[landmask]) for izs in range(0, len(zs_wrf))]
ds_interp["soil_layers"] = zs_wrf.load().data
tslb_wrf = ds_interp["TSLB"].sel(time=dt_start).load()
smois_wrf = ds_interp["SMOIS"].sel(time=dt_start).load()
deep_soil_wrf = ds_interp["TMN"].sel(time=dt_start)
deep_tsoil = deep_soil_wrf.where(landmask).mean().load().data

if np.isnan(median_smois[0]):
    print("Warning: Entire PALM domain over water surface.")
    median_smois = np.ones_like(median_smois)
    deep_tsoil = deep_soil_wrf.mean().load().data

for izs in range(0, len(zs_wrf)):
    smois_wrf.isel(soil_layers=izs).data[watermask] = median_smois[izs]
    if smois_wrf.isel(soil_layers=izs).mean() == 0.0:
        smois_wrf.isel(soil_layers=izs).data[:, :] = msoil_val

zs_palm = np.zeros_like(dz_soil)
zs_palm[0] = dz_soil[0]
for i in range(1, len(dz_soil)):
    zs_palm[i] = np.sum(dz_soil[:i+1])

init_tsoil = np.zeros((len(dz_soil), len(y), len(x)))
init_msoil = np.zeros((len(dz_soil), len(y), len(x)))
for iy in tqdm(range(0, len(y)), position=0, leave=True):
    for ix in range(0, len(x)):
        init_tsoil[:, iy, ix] = np.interp(zs_palm, zs_wrf.data, tslb_wrf[:, iy, ix])
        init_msoil[:, iy, ix] = np.interp(zs_palm, zs_wrf.data, smois_wrf[:, iy, ix])

#-------------------------------------------------------------------------------
# Vertical interpolation for boundaries
#-------------------------------------------------------------------------------
print("Start vertical interpolation")
print("create empty datasets")
ds_we = ds_interp.isel(west_east=[0, -1])
ds_sn = ds_interp.isel(south_north=[0, -1])

print("create empty datasets for staggered U and V")
ds_we_ustag = ds_interp_u.isel(west_east=[0, -1])
ds_we_vstag = ds_interp_v.isel(west_east=[0, -1])
ds_sn_ustag = ds_interp_u.isel(south_north=[0, -1])
ds_sn_vstag = ds_interp_v.isel(south_north=[0, -1])

varbc_list = ["W", "QVAPOR", "pt", "Z"]
varbc_list.extend(all_chem_to_process)

print("remove unused vars from datasets")
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

print("load datasets")
ds_we = ds_we.load()
ds_sn = ds_sn.load()
ds_we_ustag = ds_we_ustag.load()
ds_sn_ustag = ds_sn_ustag.load()
ds_we_vstag = ds_we_vstag.load()
ds_sn_vstag = ds_sn_vstag.load()

print("create datasets to save data in PALM coordinates")
ds_palm_we = xr.Dataset()
ds_palm_we = ds_palm_we.assign_coords({"x": x[:2], "y": y, "time": ds_interp.time.data,
                                       "z": z, "yv": yv, "xu": xu[:2], "zw": zw})
ds_palm_sn = xr.Dataset()
ds_palm_sn = ds_palm_sn.assign_coords({"x": x, "y": y[:2], "time": ds_interp.time.data,
                                       "z": z, "yv": yv[:2], "xu": xu, "zw": zw})

print("create zeros arrays for vertical interpolation")
zeros_we = np.zeros((len(all_ts), len(z), len(y), len(x[:2])))
zeros_sn = np.zeros((len(all_ts), len(z), len(y[:2]), len(x)))

# Interpolation scalars
for varbc in ["QVAPOR", "pt"]:
    ds_palm_we[varbc] = xr.DataArray(np.copy(zeros_we), dims=['time', 'z', 'y', 'x'])
    ds_palm_sn[varbc] = xr.DataArray(np.copy(zeros_sn), dims=['time', 'z', 'y', 'x'])
    print(f"Processing {varbc} for boundaries")
    ds_palm_we[varbc] = multi_zinterp(max_pool, ds_we, varbc, z, ds_palm_we)
    ds_palm_sn[varbc] = multi_zinterp(max_pool, ds_sn, varbc, z, ds_palm_sn)

# Vertical interpolation for chemistry and aerosol species
print(f"Processing all species: {all_chem_to_process}")

# Pre-filter to only available species
available_species = [s for s in all_chem_to_process if s in list(ds_we.data_vars.keys())]
print(f"Available species for processing: {available_species}")

def process_species_batch(species_batch, ds_we, ds_sn, z, max_pool, ds_palm_we, ds_palm_sn):
    for species in species_batch:
        print(f"Processing {species}...")
        chem_dims = ds_we[species].shape
        chem_zeros_we = np.zeros((chem_dims[0], len(z), len(y), len(x[:2])))
        chem_zeros_sn = np.zeros((chem_dims[0], len(z), len(y[:2]), len(x)))
        
        ds_palm_we[species] = xr.DataArray(np.copy(chem_zeros_we), dims=['time', 'z', 'y', 'x'])
        ds_palm_sn[species] = xr.DataArray(np.copy(chem_zeros_sn), dims=['time', 'z', 'y', 'x'])
        
        ds_palm_we[species] = multi_zinterp(max_pool, ds_we, species, z, ds_palm_we)
        ds_palm_sn[species] = multi_zinterp(max_pool, ds_sn, species, z, ds_palm_sn)

batch_size = 10
for i in range(0, len(available_species), batch_size):
    batch = available_species[i:i + batch_size]
    process_species_batch(batch, ds_we, ds_sn, z, max_pool, ds_palm_we, ds_palm_sn)

# Process W for boundaries
zeros_we_w = np.zeros((len(all_ts), len(zw), len(y), len(x[:2])))
zeros_sn_w = np.zeros((len(all_ts), len(zw), len(y[:2]), len(x)))
ds_palm_we["W"] = xr.DataArray(np.copy(zeros_we_w), dims=['time', 'zw', 'y', 'x'])
ds_palm_sn["W"] = xr.DataArray(np.copy(zeros_sn_w), dims=['time', 'zw', 'y', 'x'])

print("Processing W for boundaries")
ds_palm_we["W"] = multi_zinterp(max_pool, ds_we, "W", zw, ds_palm_we)
ds_palm_sn["W"] = multi_zinterp(max_pool, ds_sn, "W", zw, ds_palm_sn)

# Process U and V
zeros_we_u = np.zeros((len(all_ts), len(z), len(y), len(xu[:2])))
zeros_sn_u = np.zeros((len(all_ts), len(z), len(y[:2]), len(xu)))
ds_palm_we["U"] = xr.DataArray(np.copy(zeros_we_u), dims=['time', 'z', 'y', 'xu'])
print("Processing U for boundaries")
ds_palm_we["U"] = multi_zinterp(max_pool, ds_we_ustag, "U", z, ds_palm_we)

ds_palm_sn["U"] = xr.DataArray(np.copy(zeros_sn_u), dims=['time', 'z', 'y', 'xu'])
print("Processing U for south/north")
ds_palm_sn["U"] = multi_zinterp(max_pool, ds_sn_ustag, "U", z, ds_palm_sn)

zeros_we_v = np.zeros((len(all_ts), len(z), len(yv), len(x[:2])))
zeros_sn_v = np.zeros((len(all_ts), len(z), len(yv[:2]), len(x)))
ds_palm_we["V"] = xr.DataArray(np.copy(zeros_we_v), dims=['time', 'z', 'yv', 'x'])
print("Processing V for west/east")
ds_palm_we["V"] = multi_zinterp(max_pool, ds_we_vstag, "V", z, ds_palm_we)

ds_palm_sn["V"] = xr.DataArray(np.copy(zeros_sn_v), dims=['time', 'z', 'yv', 'x'])
print("Processing V for south/north")
ds_palm_sn["V"] = multi_zinterp(max_pool, ds_sn_vstag, "V", z, ds_palm_sn)

#-------------------------------------------------------------------------------
# Handle traffic variables in boundary conditions
#-------------------------------------------------------------------------------
if has_traffic_vars:
    print("Setting up traffic variables in boundary conditions...")
    for base_species, traffic_species in traffic_mapping.items():
        if base_species in ds_palm_we.data_vars:
            ds_palm_we[traffic_species] = ds_palm_we[base_species].copy()
            ds_palm_sn[traffic_species] = ds_palm_sn[base_species].copy()

# Handle NaN values in boundary conditions
print("Handling NaN values in boundary conditions...")
for species in original_chem_species:
    if species in ds_palm_we.data_vars:
        if np.any(np.isnan(ds_palm_we[species].data)) or np.any(np.isnan(ds_palm_sn[species].data)):
            print(f"Found NaN values for {species} in boundaries")
            for ts in tqdm(range(len(all_ts)), desc=f"Fixing {species} NaNs", leave=False):
                for y_idx in range(len(y)):
                    west_profile = ds_palm_we[species].isel(time=ts, x=0, y=y_idx)
                    if np.any(np.isnan(west_profile.data)):
                        valid_mask = ~np.isnan(west_profile.data)
                        if np.any(valid_mask):
                            valid_z = z[valid_mask]
                            valid_values = west_profile.data[valid_mask]
                            nan_mask = np.isnan(west_profile.data)
                            if np.any(nan_mask):
                                nan_z = z[nan_mask]
                                interp_values = np.interp(nan_z, valid_z, valid_values)
                                west_data = west_profile.data.copy()
                                west_data[nan_mask] = interp_values
                                ds_palm_we[species].data[ts, :, y_idx, 0] = west_data
                
                for x_idx in range(len(x)):
                    south_profile = ds_palm_sn[species].isel(time=ts, y=0, x=x_idx)
                    if np.any(np.isnan(south_profile.data)):
                        valid_mask = ~np.isnan(south_profile.data)
                        if np.any(valid_mask):
                            valid_z = z[valid_mask]
                            valid_values = south_profile.data[valid_mask]
                            nan_mask = np.isnan(south_profile.data)
                            if np.any(nan_mask):
                                nan_z = z[nan_mask]
                                interp_values = np.interp(nan_z, valid_z, valid_values)
                                south_data = south_profile.data.copy()
                                south_data[nan_mask] = interp_values
                                ds_palm_sn[species].data[ts, :, 0, x_idx] = south_data
            
            if np.any(np.isnan(ds_palm_we[species].data)):
                ds_palm_we[species] = ds_palm_we[species].ffill('z').bfill('z')
                ds_palm_we[species] = ds_palm_we[species].ffill('y').bfill('y')
                ds_palm_we[species] = ds_palm_we[species].ffill('time').bfill('time')
            
            if np.any(np.isnan(ds_palm_sn[species].data)):
                ds_palm_sn[species] = ds_palm_sn[species].ffill('z').bfill('z')
                ds_palm_sn[species] = ds_palm_sn[species].ffill('x').bfill('x')
                ds_palm_sn[species] = ds_palm_sn[species].ffill('time').bfill('time')

#-------------------------------------------------------------------------------
# Top boundary
#-------------------------------------------------------------------------------
print("Processing top boundary conditions...")
u_top = np.zeros((len(all_ts), len(y), len(xu)))
v_top = np.zeros((len(all_ts), len(yv), len(x)))
w_top = np.zeros((len(all_ts), len(y), len(x)))
qv_top = np.zeros((len(all_ts), len(y), len(x)))
pt_top = np.zeros((len(all_ts), len(y), len(x)))

chem_top = {}
available_top_species = [s for s in all_chem_to_process if s in ds_interp.data_vars]
print(f"Available species for top boundary: {available_top_species}")

for species in available_top_species:
    chem_top[species] = np.zeros((len(all_ts), len(y), len(x)))

for var in list(ds_interp.data_vars):
    if var not in varbc_list and var not in all_chem_to_process:
        ds_interp = ds_interp.drop(var)
for var in list(ds_interp_u.data_vars):
    if var not in ["U", "Z"] and var not in all_chem_to_process:
        ds_interp_u = ds_interp_u.drop(var)
for var in list(ds_interp_v.data_vars):
    if var not in ["V", "Z"] and var not in all_chem_to_process:
        ds_interp_v = ds_interp_v.drop(var)

print("Processing top boundary datasets...")
ds_interp_top = xr.Dataset()
ds_interp_u_top = xr.Dataset()
ds_interp_v_top = xr.Dataset()

for var in ["QVAPOR", "pt"]:
    ds_interp_top[var] = ds_interp.salem.wrf_zlevel(var, levels=z[-1]).copy()

for species in available_top_species:
    if species in ds_interp.data_vars:
        ds_interp_top[species] = ds_interp.salem.wrf_zlevel(species, levels=z[-1]).copy()

ds_interp_top["W"] = ds_interp.salem.wrf_zlevel("W", levels=zw[-1]).copy()
ds_interp_u_top["U"] = ds_interp_u.salem.wrf_zlevel("U", levels=z[-1]).copy()
ds_interp_v_top["V"] = ds_interp_v.salem.wrf_zlevel("V", levels=z[-1]).copy()

print("Processing top boundary data for all timestamps...")
for ts in tqdm(range(0, len(all_ts)), total=len(all_ts), position=0, leave=True):
    u_top[ts, :, :] = ds_interp_u_top["U"].isel(time=ts)
    v_top[ts, :, :] = ds_interp_v_top["V"].isel(time=ts)
    w_top[ts, :, :] = ds_interp_top["W"].isel(time=ts)
    pt_top[ts, :, :] = ds_interp_top["pt"].isel(time=ts)
    qv_top[ts, :, :] = ds_interp_top["QVAPOR"].isel(time=ts)
    
    for species in available_top_species:
        if species in ds_interp_top.data_vars:
            chem_top[species][ts, :, :] = ds_interp_top[species].isel(time=ts)

# Calculate aggregated species for top boundary
if "RH" in chem_species:
    chem_top["RH"] = np.zeros((len(all_ts), len(y), len(x)))
    for ts in range(len(all_ts)):
        for comp in RH_components:
            if comp in chem_top:
                chem_top["RH"][ts, :, :] += chem_top[comp][ts, :, :]

if "RO2" in chem_species:
    chem_top["RO2"] = np.zeros((len(all_ts), len(y), len(x)))
    for ts in range(len(all_ts)):
        for comp in RO2_components:
            if comp in chem_top:
                chem_top["RO2"][ts, :, :] += chem_top[comp][ts, :, :]

if "RCHO" in chem_species:
    chem_top["RCHO"] = np.zeros((len(all_ts), len(y), len(x)))
    for ts in range(len(all_ts)):
        for comp in RCHO_components:
            if comp in chem_top:
                chem_top["RCHO"][ts, :, :] += chem_top[comp][ts, :, :]

if "OCSV" in chem_species:
    chem_top["OCSV"] = np.zeros((len(all_ts), len(y), len(x)))
    for ts in range(len(all_ts)):
        for comp in OCSV_components:
            if comp in chem_top:
                chem_top["OCSV"][ts, :, :] += chem_top[comp][ts, :, :]

if "OCNV" in chem_species:
    chem_top["OCNV"] = np.zeros((len(all_ts), len(y), len(x)))
    for ts in range(len(all_ts)):
        for comp in OCNV_components:
            if comp in chem_top:
                chem_top["OCNV"][ts, :, :] += chem_top[comp][ts, :, :]

# Handle traffic variables in top boundary
if has_traffic_vars:
    for base_species, traffic_species in traffic_mapping.items():
        if base_species in chem_top:
            chem_top[traffic_species] = chem_top[base_species].copy()

# Handle NaN values in top boundary
for species in original_chem_species:
    if species in chem_top:
        if np.any(np.isnan(chem_top[species])):
            mean_profile = np.nanmean(chem_top[species], axis=(1, 2))
            for ts in range(len(all_ts)):
                nan_mask = np.isnan(chem_top[species][ts, :, :])
                if np.any(nan_mask):
                    chem_top[species][ts, nan_mask] = mean_profile[ts]

#-------------------------------------------------------------------------------
# Geostrophic wind estimation
#-------------------------------------------------------------------------------
print("Geostrophic wind estimation...")
ds_geostr = None

if geostr_lvl == "z":
    lat_geostr = ds_drop.lat[:, 0]
    dx_wrf = ds_drop.DX
    dy_wrf = ds_drop.DY
    gph = ds_drop.gph.load()
    ds_geostr_z = xr.Dataset()
    ds_geostr_z = ds_geostr_z.assign_coords({"time": ds_drop.time.data,
                                             "z": ds_drop["Z"].mean(("time", "south_north", "west_east")).data})
    ds_geostr_z["ug"] = xr.DataArray(np.zeros((len(all_ts), len(gph.bottom_top.data))), dims=['time', 'z'])
    ds_geostr_z["vg"] = xr.DataArray(np.zeros((len(all_ts), len(gph.bottom_top.data))), dims=['time', 'z'])

    for ts in tqdm(range(0, len(all_ts)), total=len(all_ts), position=0, leave=True):
        for levels in gph.bottom_top.data:
            ds_geostr_z["ug"][ts, levels], ds_geostr_z["vg"][ts, levels] = calc_geostrophic_wind_zlevels(
                gph[ts, levels, :, :].data, lat_geostr.data, dy_wrf, dx_wrf)

    ds_geostr = ds_geostr_z.interp({"z": z})

elif geostr_lvl == "p":
    pres = ds_drop.PRESSURE.load()
    tk = ds_drop.TK.load()
    lat_1d = ds_drop.lat[:, 0]
    lon_1d = ds_drop.lon[0, :]

    ds_geostr_p = xr.Dataset()
    ds_geostr_p = ds_geostr_p.assign_coords({"time": ds_drop.time.data,
                                             "z": ds_drop["Z"].mean(("time", "south_north", "west_east")).data})
    ds_geostr_p["ug"] = xr.DataArray(np.zeros((len(all_ts), len(pres.bottom_top.data))), dims=['time', 'z'])
    ds_geostr_p["vg"] = xr.DataArray(np.zeros((len(all_ts), len(pres.bottom_top.data))), dims=['time', 'z'])

    for ts in tqdm(range(0, len(all_ts)), total=len(all_ts), position=0, leave=True):
        for levels in pres.bottom_top.data:
            ds_geostr_p["ug"][ts, levels], ds_geostr_p["vg"][ts, levels] = calc_geostrophic_wind_plevels(
                pres[ts, levels, :, :].data, tk[ts, levels, :, :].data, lat_1d, lon_1d, dy_wrf, dx_wrf)

    ds_geostr = ds_geostr_p.interp({"z": z})
else:
    print(f"Warning: geostr_lvl '{geostr_lvl}' not recognized.")
    ds_geostr = xr.Dataset()
    ds_geostr = ds_geostr.assign_coords({"time": all_ts, "z": z})
    ds_geostr["ug"] = xr.DataArray(np.zeros((len(all_ts), len(z))), dims=['time', 'z'])
    ds_geostr["vg"] = xr.DataArray(np.zeros((len(all_ts), len(z))), dims=['time', 'z'])

#-------------------------------------------------------------------------------
# Surface NaNs
#-------------------------------------------------------------------------------
print("Resolving surface NaNs...")
with Pool(max_pool) as p:
    pool_outputs = list(
        tqdm(
            p.imap(partial(solve_surface, all_ts, ds_palm_we, ds_palm_sn, surface_var_dict), surface_var_dict.keys()),
            total=len(surface_var_dict.keys()), position=0, leave=True
        )
    )
p.join()
pool_dict = dict(pool_outputs)
for var in surface_var_dict.keys():
    ds_palm_we[var] = pool_dict[var][0]
    ds_palm_sn[var] = pool_dict[var][1]

if ds_geostr is not None:
    for t in range(0, len(all_ts)):
        ds_geostr["ug"][t, :] = surface_nan_w(ds_geostr["ug"][t, :].data)
        ds_geostr["vg"][t, :] = surface_nan_w(ds_geostr["vg"][t, :].data)

#-------------------------------------------------------------------------------
# Calculate initial profiles
#-------------------------------------------------------------------------------
ds_drop["bottom_top"] = ds_drop["Z"].mean(("time", "south_north", "west_east")).data

u_init = ds_drop["U"].sel(time=dt_start).mean(dim=["south_north", "west_east"]).interp(
    {"bottom_top": z}, method=interp_mode)
v_init = ds_drop["V"].sel(time=dt_start).mean(dim=["south_north", "west_east"]).interp(
    {"bottom_top": z}, method=interp_mode)
w_init = ds_drop["W"].sel(time=dt_start).mean(dim=["south_north", "west_east"]).interp(
    {"bottom_top": zw}, method=interp_mode)
qv_init = ds_drop["QVAPOR"].sel(time=dt_start).mean(dim=["south_north", "west_east"]).interp(
    {"bottom_top": z}, method=interp_mode)
pt_init = ds_drop["pt"].sel(time=dt_start).mean(dim=["south_north", "west_east"]).interp(
    {"bottom_top": z}, method=interp_mode)

u_init = surface_nan_uv(u_init.load().data, z, u10_wrf.sel(time=dt_start).mean(
    dim=["south_north", "west_east"]).data)
v_init = surface_nan_uv(v_init.load().data, z, v10_wrf.sel(time=dt_start).mean(
    dim=["south_north", "west_east"]).data)
w_init = surface_nan_w(w_init.load().data)
qv_init = surface_nan_s(qv_init.load().data, z, qv2_wrf.sel(time=dt_start).mean(
    dim=["south_north", "west_east"]).data)
pt_init = surface_nan_s(pt_init.load().data, z, pt2_wrf.sel(time=dt_start).mean(
    dim=["south_north", "west_east"]).data)

# Initialize chemistry species profiles
chem_init = {}
for species in all_chem_to_process:
    if species in ds_drop.data_vars:
        chem_data = ds_drop[species].sel(time=dt_start).mean(
            dim=["south_north", "west_east"]).interp(
            {"bottom_top": z}, method=interp_mode).load().data
        chem_init[species] = xr.DataArray(chem_data, dims=['z'], coords={'z': z})
    else:
        chem_init[species] = xr.DataArray(np.zeros(len(z)), dims=['z'], coords={'z': z})

# Calculate aggregated species initial profiles
if "RH" in chem_species:
    rh_init = np.zeros(len(z))
    for comp in RH_components:
        if comp in chem_init:
            rh_init += chem_init[comp].values
    chem_init["RH"] = xr.DataArray(rh_init, dims=['z'], coords={'z': z})

if "RO2" in chem_species:
    ro2_init = np.zeros(len(z))
    for comp in RO2_components:
        if comp in chem_init:
            ro2_init += chem_init[comp].values
    chem_init["RO2"] = xr.DataArray(ro2_init, dims=['z'], coords={'z': z})

if "RCHO" in chem_species:
    rcho_init = np.zeros(len(z))
    for comp in RCHO_components:
        if comp in chem_init:
            rcho_init += chem_init[comp].values
    chem_init["RCHO"] = xr.DataArray(rcho_init, dims=['z'], coords={'z': z})

if "OCSV" in chem_species:
    ocsv_init = np.zeros(len(z))
    for comp in OCSV_components:
        if comp in chem_init:
            ocsv_init += chem_init[comp].values
    chem_init["OCSV"] = xr.DataArray(ocsv_init, dims=['z'], coords={'z': z})

if "OCNV" in chem_species:
    ocnv_init = np.zeros(len(z))
    for comp in OCNV_components:
        if comp in chem_init:
            ocnv_init += chem_init[comp].values
    chem_init["OCNV"] = xr.DataArray(ocnv_init, dims=['z'], coords={'z': z})

#-------------------------------------------------------------------------------
# AEROSOL INITIAL PROFILES - Using upwind location method
#-------------------------------------------------------------------------------
if aerosol_wrfchem:
    print("\n" + "="*60)
    print("CALCULATING AEROSOL INITIAL PROFILES (Upwind Method)")
    print("="*60)
    
    # Define PALM aerosol bins
    dmid, bin_limits = define_bins(nbin, reglim)
    nbins = len(dmid)
    n_species = len(listspec)
    print(f"  Number of PALM bins: {nbins}")
    print(f"  Number of species: {n_species}")
    
    # Calculate overlap with WRF-Chem bins
    open_bins, overlap_ratio = aerosol_binoverlap(bin_limits, wrfchem_bin_limits)
    open_bins = sorted(set(open_bins), key=open_bins.index)
    print(f"  Open WRF-Chem bins: {open_bins}")
    
    # Get u and v profiles for upwind calculation
    u_3d = ds_drop["U"].sel(time=dt_start).load().data
    v_3d = ds_drop["V"].sel(time=dt_start).load().data
    
    # Get WRF vertical levels for height matching
    wrf_z = ds_drop["Z"].mean(("time", "south_north", "west_east")).data
    
    # Calculate aerosol mass fractions using upwind method
    aero_massfrac_a = np.zeros((len(z), n_species))
    
    print("  Calculating mass fractions...")
    for zlev in tqdm(range(len(z)), desc="  Mass fractions"):
        # Find closest WRF level
        closest_wrf_lev = np.argmin(np.abs(wrf_z - z[zlev]))
        
        # Get upwind location at this level
        upwind_x, upwind_y = upwind_location(closest_wrf_lev, 
                                              u_3d[np.newaxis, :, :, :], 
                                              v_3d[np.newaxis, :, :, :])
        
        # Calculate total mass at upwind location for each PALM species
        total_mass = 0
        spec_mass = np.zeros(n_species)
        
        for idx, spec in enumerate(listspec):
            mass_val = 0
            # Get all WRF-Chem base names for this PALM species
            wrfchem_names = get_wrfchem_variables_for_species(spec)
            
            for wrfchem_name in wrfchem_names:
                for bin_suffix in WRFCHEM_BIN_SUFFIXES:
                    var_name = f'{wrfchem_name}{bin_suffix}'
                    if var_name in ds_drop.data_vars:
                        mass_val += ds_drop[var_name].sel(time=dt_start).isel(
                            bottom_top=closest_wrf_lev,
                            south_north=upwind_y,
                            west_east=upwind_x
                        ).load().data
            
            spec_mass[idx] = mass_val
            total_mass += mass_val
        
        # Calculate mass fractions
        if total_mass > 0:
            aero_massfrac_a[zlev, :] = spec_mass / total_mass
        else:
            aero_massfrac_a[zlev, :] = 0
    
    # Store in chem_init with appropriate naming
    chem_init['mass_fracs_a'] = xr.DataArray(aero_massfrac_a.astype(np.float32), 
                                              dims=['z', 'composition_index'])
    
    # Mass fraction for B mode (insoluble)
    if nf2a == 1.0:
        aero_massfrac_b = np.zeros_like(aero_massfrac_a)
    elif nf2a < 1.0:
        aero_massfrac_b = 1 - aero_massfrac_a
    else:
        aero_massfrac_b = np.zeros_like(aero_massfrac_a)
    
    chem_init['mass_fracs_b'] = xr.DataArray(aero_massfrac_b.astype(np.float32), 
                                              dims=['z', 'composition_index'])
    
    # Calculate aerosol number concentration
    aero_concentration = np.zeros((len(z), nbins))
    
    print("  Calculating number concentration...")
    for zlev in tqdm(range(len(z)), desc="  Number concentration"):
        closest_wrf_lev = np.argmin(np.abs(wrf_z - z[zlev]))
        upwind_x, upwind_y = upwind_location(closest_wrf_lev, 
                                              u_3d[np.newaxis, :, :, :], 
                                              v_3d[np.newaxis, :, :, :])
        
        for n_dmid in range(nbins):
            outval = 0.0
            for abin_idx, bin_name in enumerate(open_bins):
                var_name = f'num{bin_name}'
                if var_name in ds_drop.data_vars:
                    inval = ds_drop[var_name].sel(time=dt_start).isel(
                        bottom_top=closest_wrf_lev,
                        south_north=upwind_y,
                        west_east=upwind_x
                    ).load().data
                    outval += inval * overlap_ratio[n_dmid, abin_idx]
            aero_concentration[zlev, n_dmid] = outval
    
    chem_init['aerosol'] = xr.DataArray(aero_concentration.astype(np.float32), 
                                         dims=['z', 'Dmid'])
    chem_init['Dmid'] = xr.DataArray(dmid.astype(np.float32), dims=['Dmid'])
    
    print(f"Aerosol initial profiles calculated:")
    print(f"  - Mass fractions: {aero_massfrac_a.shape}")
    print(f"  - Number concentration: {aero_concentration.shape}")
    print(f"  - Non-zero concentration values: {np.sum(aero_concentration > 0)} out of {aero_concentration.size}")
    print("="*60)

# Handle traffic variables in initial profiles
if has_traffic_vars:
    for base_species, traffic_species in traffic_mapping.items():
        if base_species in chem_init:
            chem_init[traffic_species] = chem_init[base_species].copy()

surface_pres = psfc_wrf[:, :, :].mean(dim=["south_north", "west_east"]).load()

#-------------------------------------------------------------------------------
# Process radiation data from WRF
#-------------------------------------------------------------------------------
rad_times_sec = []
rad_values_proc = [[], [], []]

if radiation_from_wrf:
    print("\n" + "="*60)
    print("PROCESSING RADIATION DATA FROM WRF")
    print("="*60)
    
    radiation_vars_exist = all(var in ds_wrf.variables for var in ['SWDOWN', 'GLW', 'SWDDIF'])
    
    if radiation_vars_exist:
        rad_times_sec = times_sec
        
        n_wrf_x = east_idx - west_idx + 1
        n_wrf_y = north_idx - south_idx + 1
        ngrids = n_wrf_x * n_wrf_y
        
        if ngrids > 0:
            rad_swdown, rad_lwdown, rad_swdiff = [], [], []
            wrf_times = ds_wrf.time.values
            
            for ts in tqdm(range(len(all_ts)), desc="Radiation processing", unit="timestep"):
                current_time = all_ts[ts]
                time_diffs = np.abs(wrf_times - current_time)
                closest_idx = np.argmin(time_diffs)
                
                sw_cropped = ds_wrf['SWDOWN'].isel(
                    time=closest_idx,
                    west_east=slice(west_idx, east_idx + 1),
                    south_north=slice(south_idx, north_idx + 1)
                ).values
                
                lw_cropped = ds_wrf['GLW'].isel(
                    time=closest_idx,
                    west_east=slice(west_idx, east_idx + 1),
                    south_north=slice(south_idx, north_idx + 1)
                ).values
                
                dif_cropped = ds_wrf['SWDDIF'].isel(
                    time=closest_idx,
                    west_east=slice(west_idx, east_idx + 1),
                    south_north=slice(south_idx, north_idx + 1)
                ).values
                
                rad_swdown.append(np.mean(sw_cropped))
                rad_lwdown.append(np.mean(lw_cropped))
                rad_swdiff.append(np.mean(dif_cropped))
            
            rad_values_proc = [rad_swdown, rad_lwdown, rad_swdiff]
    else:
        rad_times_sec = []
        rad_values_proc = [[], [], []]
else:
    rad_times_sec = []
    rad_values_proc = [[], [], []]

print("\n" + "="*60)

#-------------------------------------------------------------------------------
# Output NetCDF file
#-------------------------------------------------------------------------------
nc_output_name = f'dynamic_files/{case_name}_dynamic_{start_year}_{start_month}_{start_day}_{start_hour}'
print('Writing NetCDF file', flush=True)
nc_output = xr.Dataset()
res_origin = str(dx) + 'x' + str(dy) + ' m'

# Global attributes
nc_output.attrs['description'] = f'Contains dynamic data from WRF mesoscale. WRF output file: {wrf_file}'
nc_output.attrs['author'] = 'Meteorology: Dongqi Lin (dongqi.lin@pg.canterbury.ac.nz); Chemistry, Aerosol and Radiation: Sathish Kumar Vaithiyanadhan (sathishvaithiyanadhan@gmail.com)'
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

# Add aerosol-related global attributes if aerosol processing is enabled
if aerosol_wrfchem:
    # Define bin sizes (geometric mean diameters)
    dmid, bin_limits = define_bins(nbin, reglim)
    bin_sizes_str = ', '.join([f'{d:.3e}' for d in dmid])
    nc_output.attrs['aerosol_bin_sizes_m'] = bin_sizes_str
    nc_output.attrs['aerosol_bin_sizes_description'] = 'Geometric mean diameters of aerosol bins'
    
    # Add reglim used
    reglim_str = ', '.join([f'{r:.3e}' for r in reglim])
    nc_output.attrs['aerosol_reglim_m'] = reglim_str
    nc_output.attrs['aerosol_reglim_description'] = 'Aerosol bin limits for PALM SALSA scheme [lower1, upper1, upper2]'
    
    # Add listspec species order
    listspec_str = ', '.join(listspec)
    nc_output.attrs['aerosol_species_order'] = listspec_str
    nc_output.attrs['aerosol_species_description'] = 'Order of aerosol chemical components in composition_index'
    
    # Add nbin information
    nbin_str = f'{nbin[0]}, {nbin[1]}' if len(nbin) == 2 else str(nbin)
    nc_output.attrs['aerosol_nbin'] = nbin_str
    nc_output.attrs['aerosol_nbin_description'] = 'Number of bins in each subrange [subrange1, subrange2]'
    
    # Add nf2a factor
    nc_output.attrs['aerosol_nf2a'] = str(nf2a)
    nc_output.attrs['aerosol_nf2a_description'] = 'Soluble/insoluble fraction factor (1.0 = all soluble, <1.0 = partially soluble)'
    
    # Add WRF-Chem bin limits if available
    if 'wrfchem_bin_limits' in locals():
        wrfchem_limits_str = ', '.join([f'{l:.3e}' for l in wrfchem_bin_limits])
        nc_output.attrs['wrfchem_bin_limits_m'] = wrfchem_limits_str
        nc_output.attrs['wrfchem_bin_limits_description'] = 'WRF-Chem aerosol bin limits used for overlap calculation'
    
    print(f"  Added aerosol global attributes:")
    print(f"    - Bin sizes: {bin_sizes_str}")
    print(f"    - Reglim: {reglim_str}")
    print(f"    - Species order: {listspec_str}")
    print(f"    - nbin: {nbin_str}")
    print(f"    - nf2a: {nf2a}")

nc_output['x'] = xr.DataArray(x, dims=['x'], attrs={'units': 'm'})
nc_output['y'] = xr.DataArray(y, dims=['y'], attrs={'units': 'm'})
nc_output['z'] = xr.DataArray(z - z_origin, dims=['z'], attrs={'units': 'm'})
nc_output['zsoil'] = xr.DataArray(dz_soil, dims=['zsoil'], attrs={'units': 'm'})
nc_output['xu'] = xr.DataArray(xu, dims=['xu'], attrs={'units': 'm'})
nc_output['yv'] = xr.DataArray(yv, dims=['yv'], attrs={'units': 'm'})
nc_output['zw'] = xr.DataArray(zw - z_origin, dims=['zw'], attrs={'units': 'm'})
nc_output['time'] = xr.DataArray(times_sec, dims=['time'], attrs={'units': 'seconds'})

nc_output.to_netcdf(nc_output_name)

nc_output['init_soil_m'] = xr.DataArray(init_msoil, dims=['zsoil', 'y', 'x'],
    attrs={'units': 'm^3/m^3', 'lod': np.int32(2), 'source': 'WRF', 'long_name': 'volumetric soil moisture'})
nc_output['init_soil_t'] = xr.DataArray(init_tsoil, dims=['zsoil', 'y', 'x'],
    attrs={'units': 'K', 'lod': np.int32(2), 'source': 'WRF', 'long_name': 'soil temperature'})

# Output boundary conditions (basic meteorological variables)
nc_output['init_atmosphere_pt'] = xr.DataArray(pt_init, dims=['z'],
    attrs={'units': 'K', 'lod': np.int32(1), 'source': 'WRF', 'res_origin': res_origin})
nc_output['ls_forcing_left_pt'] = xr.DataArray(ds_palm_we["pt"][:, :, :, 0].data, dims=['time', 'z', 'y'],
    attrs={'units': 'K', 'source': 'WRF', 'res_origin': res_origin})
nc_output['ls_forcing_right_pt'] = xr.DataArray(ds_palm_we["pt"][:, :, :, -1].data, dims=['time', 'z', 'y'],
    attrs={'units': 'K', 'source': 'WRF', 'res_origin': res_origin})
nc_output['ls_forcing_south_pt'] = xr.DataArray(ds_palm_sn["pt"][:, :, 0, :].data, dims=['time', 'z', 'x'],
    attrs={'units': 'K', 'source': 'WRF', 'res_origin': res_origin})
nc_output['ls_forcing_north_pt'] = xr.DataArray(ds_palm_sn["pt"][:, :, -1, :].data, dims=['time', 'z', 'x'],
    attrs={'units': 'K', 'source': 'WRF', 'res_origin': res_origin})
nc_output['ls_forcing_top_pt'] = xr.DataArray(pt_top[:, :, :], dims=['time', 'y', 'x'],
    attrs={'units': 'K', 'source': 'WRF', 'res_origin': res_origin})

nc_output['init_atmosphere_qv'] = xr.DataArray(qv_init, dims=['z'],
    attrs={'units': 'kg/kg', 'lod': np.int32(1), 'source': 'WRF', 'res_origin': res_origin})
nc_output['ls_forcing_left_qv'] = xr.DataArray(ds_palm_we["QVAPOR"][:, :, :, 0].data, dims=['time', 'z', 'y'],
    attrs={'units': 'kg/kg', 'source': 'WRF', 'res_origin': res_origin})
nc_output['ls_forcing_right_qv'] = xr.DataArray(ds_palm_we["QVAPOR"][:, :, :, -1].data, dims=['time', 'z', 'y'],
    attrs={'units': 'kg/kg', 'source': 'WRF', 'res_origin': res_origin})
nc_output['ls_forcing_south_qv'] = xr.DataArray(ds_palm_sn["QVAPOR"][:, :, 0, :].data, dims=['time', 'z', 'x'],
    attrs={'units': 'kg/kg', 'source': 'WRF', 'res_origin': res_origin})
nc_output['ls_forcing_north_qv'] = xr.DataArray(ds_palm_sn["QVAPOR"][:, :, -1, :].data, dims=['time', 'z', 'x'],
    attrs={'units': 'kg/kg', 'source': 'WRF', 'res_origin': res_origin})
nc_output['ls_forcing_top_qv'] = xr.DataArray(qv_top[:, :, :], dims=['time', 'y', 'x'],
    attrs={'units': 'kg/kg', 'source': 'WRF', 'res_origin': res_origin})

nc_output['init_atmosphere_u'] = xr.DataArray(u_init, dims=['z'],
    attrs={'units': 'm/s', 'lod': np.int32(1), 'source': 'WRF', 'res_origin': res_origin})
nc_output['ls_forcing_left_u'] = xr.DataArray(ds_palm_we["U"][:, :, :, 0].data, dims=['time', 'z', 'y'],
    attrs={'units': 'm/s', 'source': 'WRF', 'res_origin': res_origin})
nc_output['ls_forcing_right_u'] = xr.DataArray(ds_palm_we["U"][:, :, :, -1].data, dims=['time', 'z', 'y'],
    attrs={'units': 'm/s', 'source': 'WRF', 'res_origin': res_origin})
nc_output['ls_forcing_south_u'] = xr.DataArray(ds_palm_sn["U"][:, :, 0, :].data, dims=['time', 'z', 'xu'],
    attrs={'units': 'm/s', 'source': 'WRF', 'res_origin': res_origin})
nc_output['ls_forcing_north_u'] = xr.DataArray(ds_palm_sn["U"][:, :, -1, :].data, dims=['time', 'z', 'xu'],
    attrs={'units': 'm/s', 'source': 'WRF', 'res_origin': res_origin})
nc_output['ls_forcing_top_u'] = xr.DataArray(u_top[:, :, :], dims=['time', 'y', 'xu'],
    attrs={'units': 'm/s', 'source': 'WRF', 'res_origin': res_origin})

nc_output['init_atmosphere_v'] = xr.DataArray(v_init, dims=['z'],
    attrs={'units': 'm/s', 'lod': np.int32(1), 'source': 'WRF', 'res_origin': res_origin})
nc_output['ls_forcing_left_v'] = xr.DataArray(ds_palm_we["V"][:, :, :, 0].data, dims=['time', 'z', 'yv'],
    attrs={'units': 'm/s', 'source': 'WRF', 'res_origin': res_origin})
nc_output['ls_forcing_right_v'] = xr.DataArray(ds_palm_we["V"][:, :, :, -1].data, dims=['time', 'z', 'yv'],
    attrs={'units': 'm/s', 'source': 'WRF', 'res_origin': res_origin})
nc_output['ls_forcing_south_v'] = xr.DataArray(ds_palm_sn["V"][:, :, 0, :].data, dims=['time', 'z', 'x'],
    attrs={'units': 'm/s', 'source': 'WRF', 'res_origin': res_origin})
nc_output['ls_forcing_north_v'] = xr.DataArray(ds_palm_sn["V"][:, :, -1, :].data, dims=['time', 'z', 'x'],
    attrs={'units': 'm/s', 'source': 'WRF', 'res_origin': res_origin})
nc_output['ls_forcing_top_v'] = xr.DataArray(v_top[:, :, :], dims=['time', 'yv', 'x'],
    attrs={'units': 'm/s', 'source': 'WRF', 'res_origin': res_origin})

nc_output['init_atmosphere_w'] = xr.DataArray(w_init, dims=['zw'],
    attrs={'units': 'm/s', 'lod': np.int32(1), 'source': 'WRF', 'res_origin': res_origin})
nc_output['ls_forcing_left_w'] = xr.DataArray(ds_palm_we["W"][:, :, :, 0].data, dims=['time', 'zw', 'y'],
    attrs={'units': 'm/s', 'source': 'WRF', 'res_origin': res_origin})
nc_output['ls_forcing_right_w'] = xr.DataArray(ds_palm_we["W"][:, :, :, -1].data, dims=['time', 'zw', 'y'],
    attrs={'units': 'm/s', 'source': 'WRF', 'res_origin': res_origin})
nc_output['ls_forcing_south_w'] = xr.DataArray(ds_palm_sn["W"][:, :, 0, :].data, dims=['time', 'zw', 'x'],
    attrs={'units': 'm/s', 'source': 'WRF', 'res_origin': res_origin})
nc_output['ls_forcing_north_w'] = xr.DataArray(ds_palm_sn["W"][:, :, -1, :].data, dims=['time', 'zw', 'x'],
    attrs={'units': 'm/s', 'source': 'WRF', 'res_origin': res_origin})
nc_output['ls_forcing_top_w'] = xr.DataArray(w_top[:, :, :], dims=['time', 'y', 'x'],
    attrs={'units': 'm/s', 'source': 'WRF', 'res_origin': res_origin})

nc_output['surface_forcing_surface_pressure'] = xr.DataArray(surface_pres.data, dims=['time'],
    attrs={'units': 'Pa', 'lod': np.int32(1), 'source': 'WRF', 'res_origin': res_origin})

# Chemistry species output
MICROGRAM_TO_KG = 1e-9

chem_name_mapping = {
    "hno3": "HNO3", "ho2": "HO2", "ho": "OH", "no2": "NO2", "o3": "O3",
    "no": "NO", "qvapor": "H2O", "nh3": "NH3", "so2": "SO2", "co": "CO",
    "sulf": "H2SO4", "RH": "RH", "RO2": "RO2", "RCHO": "RCHO",
    "OCSV": "OCSV", "OCNV": "OCNV", "PM10": "PM10", "PM2_5_DRY": "PM25"
}

for species in original_chem_species:
    if species.endswith('_traffic'):
        base = species.replace('_traffic', '')
        if base in chem_name_mapping:
            output_species_name = f"{chem_name_mapping[base]}_traffic"
        else:
            output_species_name = f"{base.upper()}_traffic"
    else:
        output_species_name = chem_name_mapping.get(species, species.upper())
    
    if species in chem_init:
        if species in ['PM10', 'PM2_5_DRY']:
            converted_data = chem_init[species].data * MICROGRAM_TO_KG
            nc_output[f'init_atmosphere_{output_species_name}'] = xr.DataArray(converted_data, dims=['z'],
                attrs={'units': 'kg/m3', 'lod': np.int32(1), 'source': 'WRF-Chem', 'res_origin': res_origin})
        else:
            nc_output[f'init_atmosphere_{output_species_name}'] = xr.DataArray(chem_init[species].data, dims=['z'],
                attrs={'units': 'ppm', 'lod': np.int32(1), 'source': 'WRF-Chem', 'res_origin': res_origin})
    
    if species in ds_palm_we.data_vars:
        if species in ['PM10', 'PM2_5_DRY']:
            left_data = ds_palm_we[species][:, :, :, 0].data * MICROGRAM_TO_KG
            right_data = ds_palm_we[species][:, :, :, -1].data * MICROGRAM_TO_KG
            south_data = ds_palm_sn[species][:, :, 0, :].data * MICROGRAM_TO_KG
            north_data = ds_palm_sn[species][:, :, -1, :].data * MICROGRAM_TO_KG
            top_data = chem_top[species] * MICROGRAM_TO_KG
            unit = "kg/m3"
        else:
            left_data = ds_palm_we[species][:, :, :, 0].data
            right_data = ds_palm_we[species][:, :, :, -1].data
            south_data = ds_palm_sn[species][:, :, 0, :].data
            north_data = ds_palm_sn[species][:, :, -1, :].data
            top_data = chem_top[species]
            unit = "ppm"
        
        nc_output[f'ls_forcing_left_{output_species_name}'] = xr.DataArray(left_data, dims=['time', 'z', 'y'],
            attrs={'units': unit, 'source': 'WRF-Chem', 'res_origin': res_origin})
        nc_output[f'ls_forcing_right_{output_species_name}'] = xr.DataArray(right_data, dims=['time', 'z', 'y'],
            attrs={'units': unit, 'source': 'WRF-Chem', 'res_origin': res_origin})
        nc_output[f'ls_forcing_south_{output_species_name}'] = xr.DataArray(south_data, dims=['time', 'z', 'x'],
            attrs={'units': unit, 'source': 'WRF-Chem', 'res_origin': res_origin})
        nc_output[f'ls_forcing_north_{output_species_name}'] = xr.DataArray(north_data, dims=['time', 'z', 'x'],
            attrs={'units': unit, 'source': 'WRF-Chem', 'res_origin': res_origin})
        nc_output[f'ls_forcing_top_{output_species_name}'] = xr.DataArray(top_data, dims=['time', 'y', 'x'],
            attrs={'units': unit, 'source': 'WRF-Chem', 'res_origin': res_origin})

#-------------------------------------------------------------------------------
# Aerosol output
#-------------------------------------------------------------------------------
if aerosol_wrfchem:
    print("\n" + "="*60)
    print("ADDING AEROSOL VARIABLES TO OUTPUT")
    print("="*60)
    
    # Get bin definitions
    dmid, bin_limits = define_bins(nbin, reglim)
    nbins = len(dmid)
    n_species = len(listspec)
    
    # Add Dmid dimension and variable
    nc_output['Dmid'] = xr.DataArray(dmid.astype(np.float32), dims=['Dmid'],
        attrs={'units': 'm', 'long_name': 'aerosol bin geometric mean diameters'})
    print(f"  Added Dmid variable with {nbins} bins")
    
    # Add composition_index dimension
    nc_output['composition_index'] = xr.DataArray(
        np.arange(1, n_species + 1, dtype=np.int32), dims=['composition_index'],
        attrs={'long_name': 'aerosol species index', 'units': '1'})
    print(f"  Added composition_index with {n_species} indices")
    
    # Add max_string_length dimension
    max_string_length = 25
    nc_output['max_string_length'] = xr.DataArray(
        np.arange(1, max_string_length + 1, dtype=np.int32), dims=['max_string_length'],
        attrs={'units': '-', 'long_name': 'maximum string length'})
    
    # Add composition_name as character array
    char_array = np.zeros((n_species, max_string_length), dtype='S1')
    for i, name in enumerate(listspec):
        name_bytes = name.encode('utf-8')
        name_len = min(len(name_bytes), max_string_length)
        char_array[i, :name_len] = [bytes([b]) for b in name_bytes[:name_len]]
    
    nc_output['composition_name'] = xr.DataArray(
        char_array, dims=['composition_index', 'max_string_length'],
        attrs={'long_name': 'aerosol species names', 'units': '-'})
    print(f"  Added composition_name with species: {listspec}")
    
    # Initial profiles
    if 'mass_fracs_a' in chem_init:
        nc_output['init_atmosphere_mass_fracs_a'] = xr.DataArray(
            chem_init['mass_fracs_a'].data, dims=['z', 'composition_index'],
            attrs={'units': '', 'source': 'WRF-Chem', 'long_name': 'initial mass fraction - SOLUBLE components (Mode A)'})
        print(f"  Added init_atmosphere_mass_fracs_a with shape {chem_init['mass_fracs_a'].data.shape}")
    
    if 'mass_fracs_b' in chem_init:
        nc_output['init_atmosphere_mass_fracs_b'] = xr.DataArray(
            chem_init['mass_fracs_b'].data, dims=['z', 'composition_index'],
            attrs={'units': '', 'source': 'WRF-Chem', 'long_name': 'initial mass fraction - INSOLUBLE components (Mode B)'})
        print(f"  Added init_atmosphere_mass_fracs_b with shape {chem_init['mass_fracs_b'].data.shape}")
    
    if 'aerosol' in chem_init:
        nc_output['init_atmosphere_aerosol'] = xr.DataArray(
            chem_init['aerosol'].data, dims=['z', 'Dmid'],
            attrs={'units': '#/m3', 'lod': 1, 'source': 'WRF-Chem', 'long_name': 'initial aerosol number concentration per bin'})
        print(f"  Added init_atmosphere_aerosol with shape {chem_init['aerosol'].data.shape}")
        print(f"  Non-zero values: {np.sum(chem_init['aerosol'].data > 0)} out of {chem_init['aerosol'].data.size}")
    
    # Calculate boundary conditions for aerosol number concentration
    print("  Calculating boundary conditions for aerosols...")
    
    # Get bin definitions and overlap
    open_bins, overlap_ratio = aerosol_binoverlap(bin_limits, wrfchem_bin_limits)
    open_bins = sorted(set(open_bins), key=open_bins.index)
    
    n_species = len(listspec)
    
    # Initialize arrays
    left_aerosol = np.zeros((len(all_ts), len(z), len(y), nbins))
    right_aerosol = np.zeros((len(all_ts), len(z), len(y), nbins))
    south_aerosol = np.zeros((len(all_ts), len(z), len(x), nbins))
    north_aerosol = np.zeros((len(all_ts), len(z), len(x), nbins))
    top_aerosol = np.zeros((len(all_ts), len(y), len(x), nbins))
    
    # Mass fractions
    left_mass_a = np.zeros((len(all_ts), len(z), len(y), n_species))
    right_mass_a = np.zeros((len(all_ts), len(z), len(y), n_species))
    south_mass_a = np.zeros((len(all_ts), len(z), len(x), n_species))
    north_mass_a = np.zeros((len(all_ts), len(z), len(x), n_species))
    top_mass_a = np.zeros((len(all_ts), len(y), len(x), n_species))
    
    # Process boundary conditions
    print("  Processing boundaries...")
    for ts in tqdm(range(len(all_ts)), desc="  Boundaries"):
        for zlev in range(len(z)):
            for yidx in range(len(y)):
                # Left and right boundaries - number concentration
                for n_dmid in range(nbins):
                    outval_left = 0.0
                    outval_right = 0.0
                    for abin_idx, bin_name in enumerate(open_bins):
                        var_name = f'num{bin_name}'
                        if var_name in ds_palm_we.data_vars:
                            inval_left = ds_palm_we[var_name].isel(time=ts, z=zlev, x=0, y=yidx).data
                            inval_right = ds_palm_we[var_name].isel(time=ts, z=zlev, x=-1, y=yidx).data
                            outval_left += inval_left * overlap_ratio[n_dmid, abin_idx]
                            outval_right += inval_right * overlap_ratio[n_dmid, abin_idx]
                    left_aerosol[ts, zlev, yidx, n_dmid] = outval_left
                    right_aerosol[ts, zlev, yidx, n_dmid] = outval_right
                
                # Left and right boundaries - mass fractions
                total_mass_left = 0
                total_mass_right = 0
                spec_mass_left = np.zeros(n_species)
                spec_mass_right = np.zeros(n_species)
                
                for idx, spec in enumerate(listspec):
                    mass_val_left = 0
                    mass_val_right = 0
                    wrfchem_names = get_wrfchem_variables_for_species(spec)
                    for wrfchem_name in wrfchem_names:
                        for bin_suffix in WRFCHEM_BIN_SUFFIXES:
                            var_name = f'{wrfchem_name}{bin_suffix}'
                            if var_name in ds_palm_we.data_vars:
                                mass_val_left += ds_palm_we[var_name].isel(time=ts, z=zlev, x=0, y=yidx).data
                                mass_val_right += ds_palm_we[var_name].isel(time=ts, z=zlev, x=-1, y=yidx).data
                    spec_mass_left[idx] = mass_val_left
                    spec_mass_right[idx] = mass_val_right
                    total_mass_left += mass_val_left
                    total_mass_right += mass_val_right
                
                if total_mass_left > 0:
                    left_mass_a[ts, zlev, yidx, :] = spec_mass_left / total_mass_left
                if total_mass_right > 0:
                    right_mass_a[ts, zlev, yidx, :] = spec_mass_right / total_mass_right
            
            for xidx in range(len(x)):
                # South and north boundaries - number concentration
                for n_dmid in range(nbins):
                    outval_south = 0.0
                    outval_north = 0.0
                    for abin_idx, bin_name in enumerate(open_bins):
                        var_name = f'num{bin_name}'
                        if var_name in ds_palm_sn.data_vars:
                            inval_south = ds_palm_sn[var_name].isel(time=ts, z=zlev, y=0, x=xidx).data
                            inval_north = ds_palm_sn[var_name].isel(time=ts, z=zlev, y=-1, x=xidx).data
                            outval_south += inval_south * overlap_ratio[n_dmid, abin_idx]
                            outval_north += inval_north * overlap_ratio[n_dmid, abin_idx]
                    south_aerosol[ts, zlev, xidx, n_dmid] = outval_south
                    north_aerosol[ts, zlev, xidx, n_dmid] = outval_north
                
                # South and north boundaries - mass fractions
                total_mass_south = 0
                total_mass_north = 0
                spec_mass_south = np.zeros(n_species)
                spec_mass_north = np.zeros(n_species)
                
                for idx, spec in enumerate(listspec):
                    mass_val_south = 0
                    mass_val_north = 0
                    wrfchem_names = get_wrfchem_variables_for_species(spec)
                    for wrfchem_name in wrfchem_names:
                        for bin_suffix in WRFCHEM_BIN_SUFFIXES:
                            var_name = f'{wrfchem_name}{bin_suffix}'
                            if var_name in ds_palm_sn.data_vars:
                                mass_val_south += ds_palm_sn[var_name].isel(time=ts, z=zlev, y=0, x=xidx).data
                                mass_val_north += ds_palm_sn[var_name].isel(time=ts, z=zlev, y=-1, x=xidx).data
                    spec_mass_south[idx] = mass_val_south
                    spec_mass_north[idx] = mass_val_north
                    total_mass_south += mass_val_south
                    total_mass_north += mass_val_north
                
                if total_mass_south > 0:
                    south_mass_a[ts, zlev, xidx, :] = spec_mass_south / total_mass_south
                if total_mass_north > 0:
                    north_mass_a[ts, zlev, xidx, :] = spec_mass_north / total_mass_north
        
        # Top boundary
        for yidx in range(len(y)):
            for xidx in range(len(x)):
                # Number concentration
                for n_dmid in range(nbins):
                    outval_top = 0.0
                    for abin_idx, bin_name in enumerate(open_bins):
                        var_name = f'num{bin_name}'
                        if var_name in chem_top:
                            inval_top = chem_top[var_name][ts, yidx, xidx]
                            outval_top += inval_top * overlap_ratio[n_dmid, abin_idx]
                    top_aerosol[ts, yidx, xidx, n_dmid] = outval_top
                
                # Mass fractions
                total_mass_top = 0
                spec_mass_top = np.zeros(n_species)
                for idx, spec in enumerate(listspec):
                    mass_val_top = 0
                    wrfchem_names = get_wrfchem_variables_for_species(spec)
                    for wrfchem_name in wrfchem_names:
                        for bin_suffix in WRFCHEM_BIN_SUFFIXES:
                            var_name = f'{wrfchem_name}{bin_suffix}'
                            if var_name in chem_top:
                                mass_val_top += chem_top[var_name][ts, yidx, xidx]
                    spec_mass_top[idx] = mass_val_top
                    total_mass_top += mass_val_top
                
                if total_mass_top > 0:
                    top_mass_a[ts, yidx, xidx, :] = spec_mass_top / total_mass_top
    
    # Set B mode mass fractions
    if nf2a == 1.0:
        left_mass_b = np.zeros_like(left_mass_a)
        right_mass_b = np.zeros_like(right_mass_a)
        south_mass_b = np.zeros_like(south_mass_a)
        north_mass_b = np.zeros_like(north_mass_a)
        top_mass_b = np.zeros_like(top_mass_a)
    else:
        left_mass_b = 1 - left_mass_a
        right_mass_b = 1 - right_mass_a
        south_mass_b = 1 - south_mass_a
        north_mass_b = 1 - north_mass_a
        top_mass_b = 1 - top_mass_a
    
    # Add to output
    nc_output['ls_forcing_left_aerosol'] = xr.DataArray(left_aerosol.astype(np.float32), 
        dims=['time', 'z', 'y', 'Dmid'],
        attrs={'units': '#/m3', 'lod': 1, 'source': 'WRF-Chem', 'long_name': 'aerosol number concentration - west boundary'})
    
    nc_output['ls_forcing_right_aerosol'] = xr.DataArray(right_aerosol.astype(np.float32), 
        dims=['time', 'z', 'y', 'Dmid'],
        attrs={'units': '#/m3', 'lod': 1, 'source': 'WRF-Chem', 'long_name': 'aerosol number concentration - east boundary'})
    
    nc_output['ls_forcing_south_aerosol'] = xr.DataArray(south_aerosol.astype(np.float32), 
        dims=['time', 'z', 'x', 'Dmid'],
        attrs={'units': '#/m3', 'lod': 1, 'source': 'WRF-Chem', 'long_name': 'aerosol number concentration - south boundary'})
    
    nc_output['ls_forcing_north_aerosol'] = xr.DataArray(north_aerosol.astype(np.float32), 
        dims=['time', 'z', 'x', 'Dmid'],
        attrs={'units': '#/m3', 'lod': 1, 'source': 'WRF-Chem', 'long_name': 'aerosol number concentration - north boundary'})
    
    nc_output['ls_forcing_top_aerosol'] = xr.DataArray(top_aerosol.astype(np.float32), 
        dims=['time', 'y', 'x', 'Dmid'],
        attrs={'units': '#/m3', 'lod': 1, 'source': 'WRF-Chem', 'long_name': 'aerosol number concentration - top boundary'})
    
    nc_output['ls_forcing_left_mass_fracs_a'] = xr.DataArray(left_mass_a.astype(np.float32), 
        dims=['time', 'z', 'y', 'composition_index'],
        attrs={'units': '', 'source': 'WRF-Chem', 'long_name': 'soluble mass fraction (Mode A) - west boundary'})
    
    nc_output['ls_forcing_right_mass_fracs_a'] = xr.DataArray(right_mass_a.astype(np.float32), 
        dims=['time', 'z', 'y', 'composition_index'],
        attrs={'units': '', 'source': 'WRF-Chem', 'long_name': 'soluble mass fraction (Mode A) - east boundary'})
    
    nc_output['ls_forcing_south_mass_fracs_a'] = xr.DataArray(south_mass_a.astype(np.float32), 
        dims=['time', 'z', 'x', 'composition_index'],
        attrs={'units': '', 'source': 'WRF-Chem', 'long_name': 'soluble mass fraction (Mode A) - south boundary'})
    
    nc_output['ls_forcing_north_mass_fracs_a'] = xr.DataArray(north_mass_a.astype(np.float32), 
        dims=['time', 'z', 'x', 'composition_index'],
        attrs={'units': '', 'source': 'WRF-Chem', 'long_name': 'soluble mass fraction (Mode A) - north boundary'})
    
    nc_output['ls_forcing_top_mass_fracs_a'] = xr.DataArray(top_mass_a.astype(np.float32), 
        dims=['time', 'y', 'x', 'composition_index'],
        attrs={'units': '', 'source': 'WRF-Chem', 'long_name': 'soluble mass fraction (Mode A) - top boundary'})
    
    nc_output['ls_forcing_left_mass_fracs_b'] = xr.DataArray(left_mass_b.astype(np.float32), 
        dims=['time', 'z', 'y', 'composition_index'],
        attrs={'units': '', 'source': 'WRF-Chem', 'long_name': 'insoluble mass fraction (Mode B) - west boundary'})
    
    nc_output['ls_forcing_right_mass_fracs_b'] = xr.DataArray(right_mass_b.astype(np.float32), 
        dims=['time', 'z', 'y', 'composition_index'],
        attrs={'units': '', 'source': 'WRF-Chem', 'long_name': 'insoluble mass fraction (Mode B) - east boundary'})
    
    nc_output['ls_forcing_south_mass_fracs_b'] = xr.DataArray(south_mass_b.astype(np.float32), 
        dims=['time', 'z', 'x', 'composition_index'],
        attrs={'units': '', 'source': 'WRF-Chem', 'long_name': 'insoluble mass fraction (Mode B) - south boundary'})
    
    nc_output['ls_forcing_north_mass_fracs_b'] = xr.DataArray(north_mass_b.astype(np.float32), 
        dims=['time', 'z', 'x', 'composition_index'],
        attrs={'units': '', 'source': 'WRF-Chem', 'long_name': 'insoluble mass fraction (Mode B) - north boundary'})
    
    nc_output['ls_forcing_top_mass_fracs_b'] = xr.DataArray(top_mass_b.astype(np.float32), 
        dims=['time', 'y', 'x', 'composition_index'],
        attrs={'units': '', 'source': 'WRF-Chem', 'long_name': 'insoluble mass fraction (Mode B) - top boundary'})
    
    print("  All aerosol variables added to output")
    print("="*60)

#-------------------------------------------------------------------------------
# Add radiation data to output
#-------------------------------------------------------------------------------
if len(rad_times_sec) > 0 and len(rad_values_proc[0]) > 0:
    print("Adding radiation data to output file...")
    
    nc_output['time_rad'] = xr.DataArray(rad_times_sec, dims=['time_rad'], 
        attrs={'units': 'seconds', 'long_name': 'time since simulation start for radiation'})
    
    nc_output['rad_sw_in'] = xr.DataArray(rad_values_proc[0], dims=['time_rad'],
        attrs={'units': 'W/m2', 'lod': 1, 'long_name': 'shortwave radiation incoming'})
    nc_output['rad_lw_in'] = xr.DataArray(rad_values_proc[1], dims=['time_rad'],
        attrs={'units': 'W/m2', 'lod': 1, 'long_name': 'longwave radiation incoming'})
    nc_output['rad_sw_in_dif'] = xr.DataArray(rad_values_proc[2], dims=['time_rad'],
        attrs={'units': 'W/m2', 'lod': 1, 'long_name': 'shortwave radiation incoming diffuse'})
    print("Radiation data added successfully")
else:
    print("No radiation data to add to output file")

# Write all variables
for var in nc_output.data_vars:
    if var == "composition_name":
        encoding = {var: {'dtype': 'S1', 'zlib': True, '_FillValue': None}}
    else:
        encoding = {var: {'dtype': 'float32', '_FillValue': -9999.0, 'zlib': True}}
    
    nc_output[var].to_netcdf(nc_output_name, encoding=encoding, mode='a')

print('Add to your *_p3d file: ' + '\n soil_temperature = ' +
      str([value for value in init_tsoil.mean(axis=(1, 2))]) +
      '\n soil_moisture = ' + str([value for value in init_msoil.mean(axis=(1, 2))]) +
      '\n deep_soil_temperature = ' + str(deep_tsoil) + '\n')

with open('cfg_files/' + case_name + '.cfg', "a") as cfg:
    cfg.write('Add to your *_p3d file: ' + '\n soil_temperature = ' +
              str([value for value in init_tsoil.mean(axis=(1, 2))]) +
              '\n soil_moisture = ' + str([value for value in init_msoil.mean(axis=(1, 2))]) +
              '\n deep_soil_temperature = ' + str(deep_tsoil) + '\n')

end = datetime.now()
print('PALM dynamic input file is ready. Script duration: {}'.format(end - start))
print('Start time: ' + str(all_ts[0]))
print('End time: ' + str(all_ts[-1]))
print('Time step: ' + str(times_sec[1] - times_sec[0]) + ' seconds')