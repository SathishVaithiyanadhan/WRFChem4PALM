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
from functools import partial
from multiprocess import Pool
from dynamic_util.nearest import framing_2d_cartesian
from dynamic_util.loc_dom import calc_stretch, domain_location, generate_cfg
from dynamic_util.process_wrf import zinterp, multi_zinterp
from dynamic_util.geostrophic import *
from dynamic_util.surface_nan_solver import *
import warnings
## supress warnings
## switch to other actions if needed
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
    
    # Check for traffic variables in the species list
    for species in chem_species:
        if species.endswith('_traffic'):
            base_species = species.replace('_traffic', '')
            if base_species in ['no', 'no2']:  # Only support NO and NO2 for now
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
settings_cfg = configparser.ConfigParser(inline_comment_prefixes='#')
config = configparser.RawConfigParser()
config.read(sys.argv[1])
case_name =  ast.literal_eval(config.get("case", "case_name"))[0]
max_pool  =  ast.literal_eval(config.get("case", "max_pool" ))[0]
geostr_lvl =  ast.literal_eval(config.get("case", "geostrophic" ))[0] 

# Read chemistry species from config and convert to list
chem_species_raw = ast.literal_eval(config.get("chemistry", "species"))
print(f"Raw chemistry species: {chem_species_raw}, type: {type(chem_species_raw)}")

# chemistry species list
if isinstance(chem_species_raw, tuple):
    # Handle case where it's a tuple containing a list
    if len(chem_species_raw) == 1 and isinstance(chem_species_raw[0], list):
        chem_species = chem_species_raw[0]
    else:
        chem_species = list(chem_species_raw)
elif isinstance(chem_species_raw, list):
    chem_species = chem_species_raw
else:
    # Handle case where it might be a single string
    chem_species = [chem_species_raw]

print(f"Final chemistry species: {chem_species}")

#-------------------------------------------------------------------------------
# Check for traffic variables
#-------------------------------------------------------------------------------
has_traffic_vars, traffic_mapping = setup_traffic_variables(chem_species)

# Remove traffic variants from the processing list since they're duplicates
# But keep them in the output species list
original_chem_species = chem_species.copy()  # Keep original for output
chem_species_for_processing = [s for s in chem_species if not s.endswith('_traffic')]

print(f"Chemistry species for processing: {chem_species_for_processing}")
print(f"Traffic variables to create: {traffic_mapping}")

# Read radiation settings from config
try:
    radiation_from_wrf = ast.literal_eval(config.get("radiation", "radiation_from_wrf"))[0]
except:
    radiation_from_wrf = True  # Default to True if not specified
    print("Radiation setting not found in config, defaulting to True")

try:
    radiation_smoothing_distance = ast.literal_eval(config.get("radiation", "radiation_smoothing_distance"))[0]
except:
    radiation_smoothing_distance = 10000.0  # Default value if not specified
    print("Radiation smoothing distance not found in config, defaulting to 10000.0 m")

print(f"Radiation from WRF: {radiation_from_wrf}")
print(f"Radiation smoothing distance: {radiation_smoothing_distance} m")

# Define component species for aggregated species (needed for processing)
RH_components = ["isopr", "apin", "bpin", "limon", "bcary", "myrc", 
                "benzene", "tol", "xylenes", "bigalk", "bigene", 
                "c2h4", "c3h6"]  #"c2h2",

RO2_components = ["ch3o2", "aco3", "mco3", "alko2", "aceto2", #"benzo2", 
                 "eto2", "pro2", "po2", "terpo2", "terp2o2",  #"xo2",
                 "nterpo2", "isopao2", "isopbo2", "mdialo2", "dicarbo2"]

RCHO_components = ["ald", "bzald", "glyald", "hydrald", "gly", "mgly", "hcho"]  #

OCSV_components = [ "cvasoa2", "cvasoa3", "cvasoa4",  "cvbsoa2", "cvbsoa3", "cvbsoa4"]

OCNV_components = ["cvasoaX", "cvasoa1", "cvbsoaX", "cvbsoa1"]

# Create a list of all component species needed for aggregation
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

# Remove duplicates
all_component_species = list(set(all_component_species))

# Combine regular chemistry species with component species for processing
all_chem_to_process = list(set(chem_species_for_processing + all_component_species))
# added later
all_chem_to_process = [s for s in all_chem_to_process if s not in ["RH", "RO2", "RCHO", "OCSV", "OCNV"]]

print(f"All chemistry species to process: {all_chem_to_process}")

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

y = np.arange(dy/2,dy*ny+dy/2,dy)
x = np.arange(dx/2,dx*nx+dx/2,dx)
z = np.arange(dz/2, dz*nz, dz)
xu = x + np.gradient(x)/2
xu = xu[:-1]
yv = y + np.gradient(y)/2
yv = yv[:-1]
zw = z + np.gradient(z)/2
zw = zw[:-1]

## stretch factor for a vertically stretched grid
# set this to 1 if no streching required
dz_stretch_factor = ast.literal_eval(config.get("stretch", "dz_stretch_factor"))[0]

## Height level above which the grid is to be stretched vertically (in m)
dz_stretch_level = ast.literal_eval(config.get("stretch", "dz_stretch_level"))[0]

## allowed maximum vertical grid spacing (in m)
dz_max = ast.literal_eval(config.get("stretch", "dz_max"))[0]

if dz_stretch_factor>1.0:
    z, zw = calc_stretch(z, dz, zw, dz_stretch_factor, dz_stretch_level, dz_max)

z += z_origin
zw += z_origin

dz_soil = np.array(ast.literal_eval(config.get("soil", "dz_soil")))
msoil_val = np.array(ast.literal_eval(config.get("soil", "msoil")))[0]


wrf_path = ast.literal_eval(config.get("wrf", "wrf_path"))[0]
wrf_file = ast.literal_eval(config.get("wrf", "wrf_output"))

interp_mode = ast.literal_eval(config.get("wrf", "interp_mode"))[0]

start_year  = ast.literal_eval(config.get("wrf", "start_year"))[0]
start_month = ast.literal_eval(config.get("wrf", "start_month"))[0]
start_day   = ast.literal_eval(config.get("wrf", "start_day"))[0]
start_hour  = ast.literal_eval(config.get("wrf", "start_hour"))[0]

end_year  = ast.literal_eval(config.get("wrf", "end_year"))[0]
end_month = ast.literal_eval(config.get("wrf", "end_month"))[0]
end_day   = ast.literal_eval(config.get("wrf", "end_day"))[0]
end_hour  = ast.literal_eval(config.get("wrf", "end_hour"))[0]
dynamic_ts = ast.literal_eval(config.get("wrf", "dynamic_ts"))[0]


#-------------------------------------------------------------------------------
# Read WRF
#-------------------------------------------------------------------------------
## the input can be one wrf file, a list of files,
# or a string glob in the form "path/to/my/files/*.nc"
print("Reading WRF")
if len(wrf_file) == 1:
    wrf_files = sorted(glob(wrf_path+wrf_file[0]))
else:
    wrf_files = sorted([wrf_path+file for file in wrf_file ])

## use salem to read WRF
# remove duplicated timestamps
ds_wrf = xr.Dataset()
with salem.open_mf_wrf_dataset(wrf_files) as ds_raw:
    ## in case xtime is created as time dimension
    if len(ds_raw["time"])==1:
        ds_raw = ds_raw.isel(time=0)
        ds_raw = ds_raw.rename({"xtime": "time"})
    for variables in ds_raw.data_vars:
        ds_wrf[variables] = ds_raw[variables].drop_duplicates("time", keep="last")
    ds_wrf.attrs = ds_raw.attrs

del ds_raw


#-------------------------------------------------------------------------------
# Find timestamps
#-------------------------------------------------------------------------------
dt_start = datetime(start_year, start_month, start_day, start_hour,)
dt_end = datetime(end_year, end_month, end_day, end_hour,)

## check WRF temporal frequency; convert ns to s
wrf_ts = (ds_wrf["time"][1]-ds_wrf["time"][0]).data.astype("float64")* 1e-9

## temporal interpolation currently not supported in WRF4PALM
if dynamic_ts<wrf_ts:
    raise SystemExit(
    "Invalid timesteps given. Stopping..."
    )


## find how many timestamps to interpolate
num_ts = (dt_end - dt_start)/timedelta(seconds=dynamic_ts)
## generate a list of timestamps
all_ts = [dt_start+i*timedelta(seconds=dynamic_ts) for i in range(0,floor(num_ts)+1)]
## round up the end time index so that PALM doesn't crash
# when data of the final timestamp is not given
if floor(num_ts) != ceil(num_ts):
    all_ts.append(dt_end)

all_ts = np.array(all_ts).astype("datetime64[ns]")
## select required timestamps
ds_wrf = ds_wrf.sel(time=all_ts)
# calculate timestamp in seconds
time_step_sec = ((dt_end-dt_start)).total_seconds()
times_sec = np.zeros(len(all_ts))
for t in range(0,len(all_ts)):
    times_sec[t] = (all_ts[t]-all_ts[0]).astype('float')*1e-9

#-------------------------------------------------------------------------------
# Locate PALM domain in WRF
#-------------------------------------------------------------------------------
## find WRF map projection
map_proj = ds_wrf.MAP_PROJ

wrf_map_dict = {
                1: "lcc",
                2: "stere",
                3: "merc",
                6: "latlong",
}

if map_proj not in wrf_map_dict:
    raise SystemExit(
    "Incompatible WRF map projection, stopping..."
    )

wgs_proj = Proj(proj='latlong', datum='WGS84', ellips='sphere')
dx_wrf, dy_wrf = ds_wrf.DX, ds_wrf.DY

if map_proj == 6:
    wrf_proj = wgs_proj
    xx_wrf = ds_wrf.lon.data
    yy_wrf = ds_wrf.lat.data

else:
    wrf_proj = Proj(proj=wrf_map_dict[map_proj], # projection type
                    lat_1=ds_wrf.TRUELAT1, lat_2=ds_wrf.TRUELAT2,
                    lat_0=ds_wrf.MOAD_CEN_LAT, lon_0=ds_wrf.STAND_LON,
                    a=6370000, b=6370000) # The Earth is a perfect sphere in WRF

    # Easting and Northings of the domains center point
    trans_wgs2wrf = Transformer.from_proj(wgs_proj, wrf_proj)
    e, n = trans_wgs2wrf.transform(ds_wrf.CEN_LON, ds_wrf.CEN_LAT)
    # WRF Grid parameters
    nx_wrf, ny_wrf = ds_wrf.dims['west_east'], ds_wrf.dims['south_north']
    # Down left corner of the domain
    x0_wrf = -(nx_wrf-1) / 2. * dx_wrf + e
    y0_wrf = -(ny_wrf-1) / 2. * dy_wrf + n
    # 2d grid
    xx_wrf, yy_wrf = np.meshgrid(np.arange(nx_wrf) * dx_wrf + x0_wrf,
                                 np.arange(ny_wrf) * dy_wrf + y0_wrf)

## if no PALM projection is given by user,
#  then use WGS84 lat/lon and WRF projection to locate domain
# otherwise use the user specified projection
if len(palm_proj_code) == 0:
    palm_proj = wrf_proj
else:
    palm_proj = Proj(init = palm_proj_code)

trans_wrf2palm = Transformer.from_proj(wrf_proj, palm_proj)
lons_wrf,lats_wrf = trans_wrf2palm.transform(xx_wrf, yy_wrf)

west, east, south, north, centx, centy = domain_location(palm_proj, wgs_proj, centlat, centlon,
                                           dx, dy, nx, ny)

## write a cfg file for future reference

generate_cfg(case_name, dx, dy, dz, nx, ny, nz,
             west, east, south, north, centlat, centlon,z_origin)

# find indices of closest values
west_idx,east_idx,south_idx,north_idx = framing_2d_cartesian(lons_wrf,lats_wrf, west,east,south,north,dx_wrf, dy_wrf)

# in case negative longitudes are used
if east_idx-west_idx<0:
    east_idx, west_idx = west_idx, east_idx

# If PALM domain smaller than one WRF grid spacing
if north_idx-south_idx<1 or east_idx-west_idx<1:
    print(north_idx, south_idx,  east_idx, west_idx)
    raise SystemExit(
    "PALM domain size is smaller than one WRF grid cell size.\n"+
    "Please consider re-configure your PALM domain.\n"+
    "Stopping...\n"
    )

## drop data outside of PALM domain area
mask_sn = (ds_wrf.south_north>=ds_wrf.south_north[south_idx]) & (ds_wrf.south_north<=ds_wrf.south_north[north_idx])
mask_we = (ds_wrf.west_east>=ds_wrf.west_east[west_idx]) & (ds_wrf.west_east<=ds_wrf.west_east[east_idx])

ds_drop = ds_wrf.where(mask_sn & mask_we, drop=True)
ds_drop["pt"] = ds_drop["T"] + 300
ds_drop["pt"].attrs = ds_drop["T"].attrs
ds_drop["gph"] = (ds_drop["PH"] + ds_drop["PHB"])/9.81
ds_drop["gph"].attrs = ds_drop["PH"].attrs


#-------------------------------------------------------------------------------
# Horizontal interpolation
#-------------------------------------------------------------------------------
print("Start horizontal interpolation")
# assign new coordinates based on PALM
south_north_palm = ds_drop.south_north[0].data+y
west_east_palm = ds_drop.west_east[0].data+x
# staggered coordinates
south_north_v_palm = ds_drop.south_north[0].data+yv
west_east_u_palm = ds_drop.west_east[0].data+xu

# interpolation
ds_drop = ds_drop.assign_coords({"west_east_palm": west_east_palm,
                                 "south_north_palm": south_north_palm,
                                 "west_east_u_palm": west_east_u_palm,
                                 "south_north_v_palm": south_north_v_palm})
ds_interp = ds_drop.interp({"west_east": ds_drop.west_east_palm,}, method = interp_mode
                          ).interp({"south_north": ds_drop.south_north_palm}, method = interp_mode)
ds_interp_u = ds_drop.interp({"west_east": ds_drop.west_east_u_palm,}, method = interp_mode
                          ).interp({"south_north": ds_drop.south_north_palm}, method = interp_mode)
ds_interp_v = ds_drop.interp({"west_east": ds_drop.west_east_palm,}, method = interp_mode
                          ).interp({"south_north": ds_drop.south_north_v_palm}, method = interp_mode)

ds_interp = ds_interp.drop(["west_east", "south_north"]
                          ).rename({"west_east_palm": "west_east",
                                    "south_north_palm": "south_north"})

ds_interp_u = ds_interp_u.drop(["west_east", "south_north"]
                          ).rename({"west_east_u_palm": "west_east",
                                    "south_north_palm": "south_north"})

ds_interp_v = ds_interp_v.drop(["west_east", "south_north"]
                          ).rename({"west_east_palm": "west_east",
                                    "south_north_v_palm": "south_north"})

## get surface and soil fields
zs_wrf = ds_interp.ZS[0,:,0,0].load()
t2_wrf = ds_interp.T2.load()
u10_wrf = ds_interp_u.U10.load()
v10_wrf = ds_interp_v.V10.load()
qv2_wrf = ds_interp.Q2.load()
psfc_wrf = ds_interp.PSFC.load()
pt2_wrf = t2_wrf*((1000)/(psfc_wrf*0.01))**0.286

surface_var_dict = {"U": u10_wrf,
                   "V": v10_wrf,
                   "pt": pt2_wrf,
                   "QVAPOR": qv2_wrf,
                   "W": None}

#-------------------------------------------------------------------------------
# soil moisture and temperature
#-------------------------------------------------------------------------------
print("Calculating soil temperature and moisture from WRF")

watermask = ds_interp["LANDMASK"].sel(time=dt_start).load().data == 0
landmask = ds_interp["LANDMASK"].sel(time=dt_start).load().data == 1
median_smois = [np.nanmedian(ds_interp["SMOIS"][0,izs,:,:].load().data[landmask]) for izs in range(0,len(zs_wrf))]
ds_interp["soil_layers"] = zs_wrf.load().data
tslb_wrf = ds_interp["TSLB"].sel(time=dt_start).load()
smois_wrf = ds_interp["SMOIS"].sel(time=dt_start).load()
deep_soil_wrf = ds_interp["TMN"].sel(time=dt_start)
deep_tsoil = deep_soil_wrf.where(landmask).mean().load().data
## in case the entire PALM domain is over water surface
if np.isnan(median_smois[0]):
    print("Warning: Entire PALM domain over water surface.")
    median_smois = np.ones_like(median_smois)
    deep_tsoil = deep_soil_wrf.mean().load().data
            
for izs in range(0,len(zs_wrf)):
    smois_wrf.isel(soil_layers=izs).data[watermask] = median_smois[izs]
    if smois_wrf.isel(soil_layers=izs).mean()== 0.0:
        smois_wrf.isel(soil_layers=izs).data[:,:] = msoil_val
# convert soil thickness to depth
zs_palm = np.zeros_like(dz_soil)
zs_palm[0] = dz_soil[0]
for i in range(1,len(dz_soil)):
    zs_palm[i] = np.sum(dz_soil[:i+1])
        
init_tsoil = np.zeros((len(dz_soil), len(y), len(x)))
init_msoil = np.zeros((len(dz_soil), len(y), len(x)))
for iy in tqdm(range(0,len(y)),position=0, leave=True):
    for ix in range(0, len(x)):
        init_tsoil[:,iy,ix] = np.interp(zs_palm, zs_wrf.data, tslb_wrf[:,iy,ix])
        init_msoil[:,iy,ix] = np.interp(zs_palm, zs_wrf.data, smois_wrf[:,iy,ix])

#-------------------------------------------------------------------------------
# Vertical interpolation
#-------------------------------------------------------------------------------
print("Start vertical interpolation")
# create an empty dataset to store interpolated data
print("create empty datasets")
ds_we = ds_interp.isel(west_east=[0,-1])
ds_sn = ds_interp.isel(south_north=[0,-1])

print("create empty datasets for staggered U and V (west&east boundaries)")
ds_we_ustag = ds_interp_u.isel(west_east=[0,-1])
ds_we_vstag = ds_interp_v.isel(west_east=[0,-1])

print("create empty datasets for staggered U and V (south&north boundaries)")
ds_sn_ustag = ds_interp_u.isel(south_north=[0,-1])
ds_sn_vstag = ds_interp_v.isel(south_north=[0,-1])

varbc_list = ["W", "QVAPOR","pt","Z"]
# Add ALL chemistry species to variable list (including components)
varbc_list.extend(all_chem_to_process)

print("remove unused vars from datasets")
for var in ds_we.data_vars:
    if var not in varbc_list:
        ds_we = ds_we.drop(var)
        ds_sn = ds_sn.drop(var)
    if var not in ["U", "Z"] and var not in all_chem_to_process:
        ds_we_ustag = ds_we_ustag.drop(var)
        ds_sn_ustag = ds_sn_ustag.drop(var)
    if var not in ["V", "Z"] and var not in all_chem_to_process:
        ds_we_vstag = ds_we_vstag.drop(var)
        ds_sn_vstag = ds_sn_vstag.drop(var)

print("load dataset for west&east boundaries")
ds_we = ds_we.load()
print("load dataset for south&north boundaries")
ds_sn = ds_sn.load()

print("load dataset for west&east boundaries (staggered U)")
ds_we_ustag = ds_we_ustag.load()
print("load dataset for south&north boundaries (staggered U)")
ds_sn_ustag = ds_sn_ustag.load()

print("load dataset for west&east boundaries (staggered V)")
ds_we_vstag = ds_we_vstag.load()
print("load dataset for south&north boundaries (staggered V)")
ds_sn_vstag = ds_sn_vstag.load()

print("create datasets to save data in PALM coordinates")
ds_palm_we = xr.Dataset()
ds_palm_we = ds_palm_we.assign_coords({"x": x[:2],"y": y, "time":ds_interp.time.data,
                                       "z": z, "yv": yv, "xu": xu[:2], "zw":zw})
ds_palm_sn = xr.Dataset()
ds_palm_sn = ds_palm_sn.assign_coords({"x": x,"y": y[:2], "time":ds_interp.time.data,
                                       "z": z, "yv": yv[:2], "xu": xu, "zw":zw})
print("create zeros arrays for vertical interpolation")
zeros_we = np.zeros((len(all_ts), len(z), len(y), len(x[:2])))
zeros_sn = np.zeros((len(all_ts), len(z), len(y[:2]), len(x)))

# interpolation scalars
for varbc in ["QVAPOR","pt"]:
    ds_palm_we[varbc] = xr.DataArray(np.copy(zeros_we), dims=['time','z','y', 'x'])
    ds_palm_sn[varbc] = xr.DataArray(np.copy(zeros_sn), dims=['time','z','y', 'x'])
    print(f"Processing {varbc} for west and east boundaries")
    ds_palm_we[varbc] = multi_zinterp(max_pool, ds_we, varbc, z, ds_palm_we)
    print(f"Processing {varbc} for south and north boundaries")
    ds_palm_sn[varbc] = multi_zinterp(max_pool, ds_sn, varbc, z, ds_palm_sn)

# Vertical interpolation for chemistry species
print(f"Processing ALL chemistry species: {all_chem_to_process}")

# Pre-filter to only available species
available_chem_species = [s for s in all_chem_to_process if s in list(ds_we.data_vars.keys())]
print(f"Available chemistry species for processing: {available_chem_species}")

# Process chemistry species in batches for better performance
def process_chemistry_batch(species_batch, ds_we, ds_sn, z, max_pool, ds_palm_we, ds_palm_sn):
    """Process a batch of chemistry species"""
    for species in species_batch:
        print(f"Processing {species}...")
        chem_dims = ds_we[species].shape
        chem_zeros_we = np.zeros((chem_dims[0], len(z), len(y), len(x[:2])))
        chem_zeros_sn = np.zeros((chem_dims[0], len(z), len(y[:2]), len(x)))
        
        ds_palm_we[species] = xr.DataArray(np.copy(chem_zeros_we), dims=['time','z','y', 'x'])
        ds_palm_sn[species] = xr.DataArray(np.copy(chem_zeros_sn), dims=['time','z','y', 'x'])
        
        # Process boundaries in parallel if possible
        ds_palm_we[species] = multi_zinterp(max_pool, ds_we, species, z, ds_palm_we)
        ds_palm_sn[species] = multi_zinterp(max_pool, ds_sn, species, z, ds_palm_sn)

# Process in batches for better memory management
batch_size = 10  # Adjust based on available memory
for i in range(0, len(available_chem_species), batch_size):
    batch = available_chem_species[i:i + batch_size]
    process_chemistry_batch(batch, ds_we, ds_sn, z, max_pool, ds_palm_we, ds_palm_sn)

#-------------------------------------------------------------------------------
# Calculate aggregated species from interpolated components
#-------------------------------------------------------------------------------
print("Calculating aggregated species from interpolated components...")

def calculate_aggregated_from_interpolated(ds_palm_we, ds_palm_sn, species_name, component_list):
    """Calculate aggregated species from interpolated component data"""
    aggregated_we = None
    aggregated_sn = None
    available_components = []
    
    for component in component_list:
        if component in ds_palm_we.data_vars:
            available_components.append(component)
            if aggregated_we is None:
                aggregated_we = ds_palm_we[component].copy()
                aggregated_sn = ds_palm_sn[component].copy()
            else:
                aggregated_we = aggregated_we + ds_palm_we[component]
                aggregated_sn = aggregated_sn + ds_palm_sn[component]
    
    if aggregated_we is None:
        print(f"Warning: No components found for {species_name}")
        # Create zeros array with correct dimensions
        aggregated_we = xr.zeros_like(ds_palm_we[list(ds_palm_we.data_vars.keys())[0]])
        aggregated_sn = xr.zeros_like(ds_palm_sn[list(ds_palm_sn.data_vars.keys())[0]])
    else:
        print(f"Calculated {species_name} from {len(available_components)} interpolated components: {available_components}")
    
    return aggregated_we, aggregated_sn

# Calculate aggregated species and add to palm datasets
if "RH" in chem_species:
    print("Calculating RH from interpolated components...")
    ds_palm_we["RH"], ds_palm_sn["RH"] = calculate_aggregated_from_interpolated(
        ds_palm_we, ds_palm_sn, "RH", RH_components)

if "RO2" in chem_species:
    print("Calculating RO2 from interpolated components...")
    ds_palm_we["RO2"], ds_palm_sn["RO2"] = calculate_aggregated_from_interpolated(
        ds_palm_we, ds_palm_sn, "RO2", RO2_components)

if "RCHO" in chem_species:
    print("Calculating RCHO from interpolated components...")
    ds_palm_we["RCHO"], ds_palm_sn["RCHO"] = calculate_aggregated_from_interpolated(
        ds_palm_we, ds_palm_sn, "RCHO", RCHO_components)

if "OCSV" in chem_species:
    print("Calculating OCSV from interpolated components...")
    ds_palm_we["OCSV"], ds_palm_sn["OCSV"] = calculate_aggregated_from_interpolated(
        ds_palm_we, ds_palm_sn, "OCSV", OCSV_components)

if "OCNV" in chem_species:
    print("Calculating OCNV from interpolated components...")
    ds_palm_we["OCNV"], ds_palm_sn["OCNV"] = calculate_aggregated_from_interpolated(
        ds_palm_we, ds_palm_sn, "OCNV", OCNV_components)

# interpolate w
zeros_we_w = np.zeros((len(all_ts), len(zw), len(y), len(x[:2])))
zeros_sn_w = np.zeros((len(all_ts), len(zw), len(y[:2]), len(x)))
ds_palm_we["W"] = xr.DataArray(np.copy(zeros_we_w), dims=['time','zw','y', 'x'])
ds_palm_sn["W"] = xr.DataArray(np.copy(zeros_sn_w), dims=['time','zw','y', 'x'])

print("Processing W for west and east boundaries")
ds_palm_we["W"] = multi_zinterp(max_pool, ds_we, "W", zw, ds_palm_we)
print("Processing W for south and north boundaries")
ds_palm_sn["W"] = multi_zinterp(max_pool, ds_sn, "W", zw, ds_palm_sn)

# interpolate u and v
zeros_we_u = np.zeros((len(all_ts), len(z), len(y), len(xu[:2])))
zeros_sn_u = np.zeros((len(all_ts), len(z), len(y[:2]), len(xu)))
ds_palm_we["U"] = xr.DataArray(np.copy(zeros_we_u), dims=['time','z','y', 'xu'])
print("Processing U for west and east boundaries")
ds_palm_we["U"] = multi_zinterp(max_pool, ds_we_ustag, "U", z, ds_palm_we)

ds_palm_sn["U"] = xr.DataArray(np.copy(zeros_sn_u), dims=['time','z','y', 'xu'])
print("Processing U for south and north boundaries")
ds_palm_sn["U"] = multi_zinterp(max_pool, ds_sn_ustag, "U", z, ds_palm_sn)

zeros_we_v = np.zeros((len(all_ts), len(z), len(yv), len(x[:2])))
zeros_sn_v = np.zeros((len(all_ts), len(z), len(yv[:2]), len(x)))
ds_palm_we["V"] = xr.DataArray(np.copy(zeros_we_v), dims=['time','z','yv', 'x'])
print("Processing V for west and east boundaries")
ds_palm_we["V"] = multi_zinterp(max_pool, ds_we_vstag, "V", z, ds_palm_we)

ds_palm_sn["V"] = xr.DataArray(np.copy(zeros_sn_v), dims=['time','z','yv', 'x'])
print("Processing V for south and north boundaries")
ds_palm_sn["V"] = multi_zinterp(max_pool, ds_sn_vstag, "V", z, ds_palm_sn)

#-------------------------------------------------------------------------------
# Handle traffic variables in boundary conditions
#-------------------------------------------------------------------------------
if has_traffic_vars:
    print("Setting up traffic variables in boundary conditions...")
    for base_species, traffic_species in traffic_mapping.items():
        # Copy boundary data from base species to traffic species
        if base_species in ds_palm_we.data_vars:
            ds_palm_we[traffic_species] = ds_palm_we[base_species].copy()
            ds_palm_sn[traffic_species] = ds_palm_sn[base_species].copy()
            print(f"  Created {traffic_species} boundary conditions from {base_species}")

#-------------------------------------------------------------------------------
# Handle NaN values in chemistry boundary conditions
#-------------------------------------------------------------------------------
print("Handling NaN values in chemistry boundary conditions...")
# Now use the original chem_species list for NaN handling (includes aggregated species)
for species in original_chem_species:
    if species in ds_palm_we.data_vars:
        print(f"Checking for NaN values in {species} boundary conditions...")
        
        # Check if there are NaN values that need to be handled
        if np.any(np.isnan(ds_palm_we[species].data)) or np.any(np.isnan(ds_palm_sn[species].data)):
            print(f"Found NaN values for {species} in boundaries")
            
            # Use vertical interpolation to fill NaN values
            # For each time and horizontal position, interpolate vertically
            for ts in tqdm(range(len(all_ts)), desc=f"Fixing {species} NaNs", leave=False):
                for y_idx in range(len(y)):
                    # West boundary
                    west_profile = ds_palm_we[species].isel(time=ts, x=0, y=y_idx)
                    if np.any(np.isnan(west_profile.data)):
                        # Get valid values and their heights
                        valid_mask = ~np.isnan(west_profile.data)
                        if np.any(valid_mask):
                            valid_z = z[valid_mask]
                            valid_values = west_profile.data[valid_mask]
                            # Interpolate to fill NaN values
                            nan_mask = np.isnan(west_profile.data)
                            if np.any(nan_mask):
                                nan_z = z[nan_mask]
                                interp_values = np.interp(nan_z, valid_z, valid_values)
                                # Update the data
                                west_data = west_profile.data.copy()
                                west_data[nan_mask] = interp_values
                                ds_palm_we[species].data[ts, :, y_idx, 0] = west_data
                
                for x_idx in range(len(x)):
                    # South boundary
                    south_profile = ds_palm_sn[species].isel(time=ts, y=0, x=x_idx)
                    if np.any(np.isnan(south_profile.data)):
                        # Get valid values and their heights
                        valid_mask = ~np.isnan(south_profile.data)
                        if np.any(valid_mask):
                            valid_z = z[valid_mask]
                            valid_values = south_profile.data[valid_mask]
                            # Interpolate to fill NaN values
                            nan_mask = np.isnan(south_profile.data)
                            if np.any(nan_mask):
                                nan_z = z[nan_mask]
                                interp_values = np.interp(nan_z, valid_z, valid_values)
                                # Update the data
                                south_data = south_profile.data.copy()
                                south_data[nan_mask] = interp_values
                                ds_palm_sn[species].data[ts, :, 0, x_idx] = south_data
            
            # Final check and fill any remaining NaNs with nearest valid value
            if np.any(np.isnan(ds_palm_we[species].data)):
                print(f"Filling remaining NaNs for {species} in west/east with nearest values")
                # Forward and backward fill along all dimensions
                ds_palm_we[species] = ds_palm_we[species].ffill('z').bfill('z')
                ds_palm_we[species] = ds_palm_we[species].ffill('y').bfill('y')
                ds_palm_we[species] = ds_palm_we[species].ffill('time').bfill('time')
            
            if np.any(np.isnan(ds_palm_sn[species].data)):
                print(f"Filling remaining NaNs for {species} in south/north with nearest values")
                ds_palm_sn[species] = ds_palm_sn[species].ffill('z').bfill('z')
                ds_palm_sn[species] = ds_palm_sn[species].ffill('x').bfill('x')
                ds_palm_sn[species] = ds_palm_sn[species].ffill('time').bfill('time')
            
            print(f"Completed NaN handling for {species}")

#-------------------------------------------------------------------------------
# top boundary
#-------------------------------------------------------------------------------
print("Processing top boundary conditions...")
u_top = np.zeros((len(all_ts), len(y), len(xu)))
v_top = np.zeros((len(all_ts), len(yv), len(x)))
w_top = np.zeros((len(all_ts), len(y), len(x)))
qv_top = np.zeros((len(all_ts), len(y), len(x)))
pt_top = np.zeros((len(all_ts), len(y), len(x)))

# Initialize arrays for chemistry species top boundary - ONLY for species that exist
chem_top = {}
# First, identify which species actually exist in the dataset
available_top_species = [s for s in all_chem_to_process if s in ds_interp.data_vars]
print(f"Available species for top boundary: {available_top_species}")

# Initialize arrays only for available species - use the unstaggered grid dimensions (y, x)
for species in available_top_species:
    chem_top[species] = np.zeros((len(all_ts), len(y), len(x)))

# Use all_chem_to_process for dropping variables (includes components but not aggregated)
for var in list(ds_interp.data_vars):  # Create a list to avoid modification during iteration
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

# Process basic variables
for var in ["QVAPOR", "pt"]:
    ds_interp_top[var] = ds_interp.salem.wrf_zlevel(var, levels=z[-1]).copy()

# Process chemistry species for top boundary - ONLY available species
for species in available_top_species:
    ds_interp_top[species] = ds_interp.salem.wrf_zlevel(species, levels=z[-1]).copy()

ds_interp_top["W"] = ds_interp.salem.wrf_zlevel("W", levels=zw[-1]).copy()        
ds_interp_u_top["U"] = ds_interp_u.salem.wrf_zlevel("U", levels=z[-1]).copy()
ds_interp_v_top["V"] = ds_interp_v.salem.wrf_zlevel("V", levels=z[-1]).copy()

# Process each timestamp individually with tqdm
print("Processing top boundary data for all timestamps...")
for ts in tqdm(range(0,len(all_ts)), total=len(all_ts), position=0, leave=True):
    u_top[ts,:,:] = ds_interp_u_top["U"].isel(time=ts)
    v_top[ts,:,:] = ds_interp_v_top["V"].isel(time=ts)
    w_top[ts,:,:] = ds_interp_top["W"].isel(time=ts)  
    pt_top[ts,:,:] = ds_interp_top["pt"].isel(time=ts) 
    qv_top[ts,:,:] = ds_interp_top["QVAPOR"].isel(time=ts) 
    
    # Process individual chemistry species top boundary - only for available species
    for species in available_top_species:
        chem_top[species][ts,:,:] = ds_interp_top[species].isel(time=ts)

# Calculate aggregated species for top boundary using the same timestamp-by-timestamp approach
if "RH" in chem_species:
    print("Calculating RH for top boundary...")
    chem_top["RH"] = np.zeros((len(all_ts), len(y), len(x)))
    for ts in range(len(all_ts)):
        for comp in RH_components:
            if comp in chem_top:
                chem_top["RH"][ts, :, :] += chem_top[comp][ts, :, :]

if "RO2" in chem_species:
    print("Calculating RO2 for top boundary...")
    chem_top["RO2"] = np.zeros((len(all_ts), len(y), len(x)))
    for ts in range(len(all_ts)):
        for comp in RO2_components:
            if comp in chem_top:
                chem_top["RO2"][ts, :, :] += chem_top[comp][ts, :, :]

if "RCHO" in chem_species:
    print("Calculating RCHO for top boundary...")
    chem_top["RCHO"] = np.zeros((len(all_ts), len(y), len(x)))
    for ts in range(len(all_ts)):
        for comp in RCHO_components:
            if comp in chem_top:
                chem_top["RCHO"][ts, :, :] += chem_top[comp][ts, :, :]

if "OCSV" in chem_species:
    print("Calculating OCSV for top boundary...")
    chem_top["OCSV"] = np.zeros((len(all_ts), len(y), len(x)))
    for ts in range(len(all_ts)):
        for comp in OCSV_components:
            if comp in chem_top:
                chem_top["OCSV"][ts, :, :] += chem_top[comp][ts, :, :]

if "OCNV" in chem_species:
    print("Calculating OCNV for top boundary...")
    chem_top["OCNV"] = np.zeros((len(all_ts), len(y), len(x)))
    for ts in range(len(all_ts)):
        for comp in OCNV_components:
            if comp in chem_top:
                chem_top["OCNV"][ts, :, :] += chem_top[comp][ts, :, :]

#-------------------------------------------------------------------------------
# Handle traffic variables in top boundary
#-------------------------------------------------------------------------------
if has_traffic_vars:
    print("Setting up traffic variables in top boundary...")
    for base_species, traffic_species in traffic_mapping.items():
        # Copy top boundary data from base species to traffic species
        if base_species in chem_top:
            chem_top[traffic_species] = chem_top[base_species].copy()
            print(f"  Created {traffic_species} top boundary from {base_species}")

# Handle NaN values in top boundary chemistry data using the same approach
print("Handling NaN values in top boundary...")
for species in original_chem_species:
    if species in chem_top:
        if np.any(np.isnan(chem_top[species])):
            print(f"Found NaN values for {species} in top boundary")
            # Use proper interpolation instead of filling with zeros
            # Get the mean profile and use it to fill missing values
            mean_profile = np.nanmean(chem_top[species], axis=(1, 2))
            for ts in range(len(all_ts)):
                nan_mask = np.isnan(chem_top[species][ts, :, :])
                if np.any(nan_mask):
                    chem_top[species][ts, nan_mask] = mean_profile[ts]

#-------------------------------------------------------------------------------
# Geostrophic wind estimation
#-------------------------------------------------------------------------------
print("Geostrophic wind estimation...")
## Check which levels should be used for geostrophic winds calculation
ds_geostr = None  # Initialize ds_geostr

if geostr_lvl == "z":
    lat_geostr = ds_drop.lat[:,0]
    dx_wrf = ds_drop.DX
    dy_wrf = ds_drop.DY
    gph = ds_drop.gph
    print("Geostrophic wind loading data...")
    gph = gph.load()
    ds_geostr_z = xr.Dataset()
    ds_geostr_z = ds_geostr_z.assign_coords({"time":ds_drop.time.data,
                                         "z": ds_drop["Z"].mean(("time", "south_north", "west_east")).data})
    ds_geostr_z["ug"] = xr.DataArray(np.zeros((len(all_ts),len(gph.bottom_top.data))),
                                   dims=['time','z'])
    ds_geostr_z["vg"] = xr.DataArray(np.zeros((len(all_ts),len(gph.bottom_top.data))),
                                   dims=['time','z'])

    for ts in tqdm(range(0,len(all_ts)), total=len(all_ts), position=0, leave=True):
        for levels in gph.bottom_top.data:
            ds_geostr_z["ug"][ts,levels], ds_geostr_z["vg"][ts,levels] = calc_geostrophic_wind_zlevels(
            gph[ts,levels, :,:].data, lat_geostr.data, dy_wrf, dx_wrf)

    # interpolate to PALM vertical levels
    ds_geostr = ds_geostr_z.interp({"z": z})

elif geostr_lvl == "p":
    pres = ds_drop.PRESSURE.load()
    tk = ds_drop.TK.load()

    lat_1d = ds_drop.lat[:,0]
    lon_1d = ds_drop.lon[0,:]

    ds_geostr_p = xr.Dataset()
    ds_geostr_p = ds_geostr_p.assign_coords({"time":ds_drop.time.data,
                                         "z": ds_drop["Z"].mean(("time", "south_north", "west_east")).data})
    ds_geostr_p["ug"] = xr.DataArray(np.zeros((len(all_ts),len(pres.bottom_top.data))),
                                   dims=['time','z'])
    ds_geostr_p["vg"] = xr.DataArray(np.zeros((len(all_ts),len(pres.bottom_top.data))),
                                   dims=['time','z'])

    for ts in tqdm(range(0,len(all_ts)), total=len(all_ts), position=0, leave=True):
        for levels in pres.bottom_top.data:
            ds_geostr_p["ug"][ts,levels], ds_geostr_p["vg"][ts,levels] = calc_geostrophic_wind_plevels(
            pres[ts,levels, :,:].data, tk[ts,levels, :,:].data, lat_1d, lon_1d, dy_wrf, dx_wrf)

    # interpolate to PALM vertical levels
    ds_geostr = ds_geostr_p.interp({"z": z})
else:
    # If geostr_lvl is neither "z" nor "p", create empty dataset with proper structure
    print(f"Warning: geostr_lvl '{geostr_lvl}' not recognized. Creating empty geostrophic wind dataset.")
    ds_geostr = xr.Dataset()
    ds_geostr = ds_geostr.assign_coords({"time": all_ts, "z": z})
    ds_geostr["ug"] = xr.DataArray(np.zeros((len(all_ts), len(z))), dims=['time','z'])
    ds_geostr["vg"] = xr.DataArray(np.zeros((len(all_ts), len(z))), dims=['time','z'])

#-------------------------------------------------------------------------------
# surface NaNs
#-------------------------------------------------------------------------------
print("Resolving surface NaNs...")
# apply multiprocessing
with Pool(max_pool) as p:
    pool_outputs = list(
        tqdm(
            p.imap(partial(solve_surface,all_ts, ds_palm_we, ds_palm_sn, surface_var_dict),surface_var_dict.keys()),
            total=len(surface_var_dict.keys()),position=0, leave=True
        )
    )
p.join()
pool_dict = dict(pool_outputs)
for var in surface_var_dict.keys():
    ds_palm_we[var]= pool_dict[var][0]
    ds_palm_sn[var]= pool_dict[var][1]
    
# near surface geostrophic wind - ONLY if ds_geostr exists
if ds_geostr is not None:
    for t in range(0,len(all_ts)):
        ds_geostr["ug"][t,:] =  surface_nan_w(ds_geostr["ug"][t,:].data)
        ds_geostr["vg"][t,:] =  surface_nan_w(ds_geostr["vg"][t,:].data)
else:
    print("Warning: ds_geostr not defined, skipping geostrophic wind surface NaN processing")

#-------------------------------------------------------------------------------
# calculate initial profiles
#-------------------------------------------------------------------------------
ds_drop["bottom_top"] = ds_drop["Z"].mean(("time", "south_north", "west_east")).data

u_init = ds_drop["U"].sel(time=dt_start).mean(
    dim=["south_north", "west_east"]).interp(
    {"bottom_top": z}, method = interp_mode)
v_init = ds_drop["V"].sel(time=dt_start).mean(
    dim=["south_north", "west_east"]).interp(
    {"bottom_top": z}, method = interp_mode)
# stagger w
w_init = ds_drop["W"].sel(time=dt_start).mean(
    dim=["south_north", "west_east"]).interp(
    {"bottom_top": zw}, method = interp_mode)
qv_init = ds_drop["QVAPOR"].sel(time=dt_start).mean(
    dim=["south_north", "west_east"]).interp(
    {"bottom_top": z}, method = interp_mode)
pt_init = ds_drop["pt"].sel(time=dt_start).mean(
    dim=["south_north", "west_east"]).interp(
    {"bottom_top": z}, method = interp_mode)

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
# First process individual species 
for species in all_chem_to_process:
    if species in ds_drop.data_vars:
        # Load the data to convert from Dask to NumPy array
        chem_data = ds_drop[species].sel(time=dt_start).mean(
            dim=["south_north", "west_east"]).interp(
            {"bottom_top": z}, method = interp_mode).load().data
        chem_init[species] = xr.DataArray(chem_data, dims=['z'], coords={'z': z})
    else:
        # If species not found, create zeros array
        chem_init[species] = xr.DataArray(np.zeros(len(z)), dims=['z'], coords={'z': z})

# Calculate aggregated species initial profiles
if "RH" in chem_species:
    rh_init = np.zeros(len(z))
    for comp in RH_components:
        if comp in chem_init:
            # Use .values instead of .data to get NumPy array
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
# Handle traffic variables in initial profiles
#-------------------------------------------------------------------------------
if has_traffic_vars:
    print("Setting up traffic variables in initial profiles...")
    for base_species, traffic_species in traffic_mapping.items():
        if base_species in chem_init:
            # Create traffic variable as a copy of the base species
            chem_init[traffic_species] = chem_init[base_species].copy()
            print(f"  Created {traffic_species} initial profile from {base_species}")

surface_pres = psfc_wrf[:, :,:].mean(dim=["south_north", "west_east"]).load()

#-------------------------------------------------------------------------------
# Process radiation data from WRF
#-------------------------------------------------------------------------------
rad_times_sec = []
rad_values_proc = [[], [], []]  # Initialize empty lists

if radiation_from_wrf:
    print("\n" + "="*60)
    print("PROCESSING RADIATION DATA FROM WRF")
    print("="*60)
    
    # Verify radiation variables exist in the WRF file
    radiation_vars_exist = all(var in ds_wrf.variables for var in ['SWDOWN', 'GLW', 'SWDDIF'])
    
    if radiation_vars_exist:
        print(" Found radiation variables in WRF output:")
        print("  - SWDOWN")
        print("  - GLW")
        print("  - SWDDIF")
        
        # Define rad_times_sec for the output file
        rad_times_sec = times_sec 
        print(f"\n Radiation times: {len(rad_times_sec)} timestamps from {all_ts[0]} to {all_ts[-1]}")
        
        # Use the same indices that were calculated for meteorology/chemistry
        # These define the PALM domain boundaries within the WRF grid
        print(f"\n PALM domain boundaries in WRF grid indices:")
        print(f"  - west_east:  from {west_idx} to {east_idx}")
        print(f"  - south_north: from {south_idx} to {north_idx}")
        
        # Calculate the number of WRF cells covering the PALM domain
        n_wrf_x = east_idx - west_idx + 1
        n_wrf_y = north_idx - south_idx + 1
        ngrids = n_wrf_x * n_wrf_y
        print(f"  - Grid size: {n_wrf_x} x {n_wrf_y} = {ngrids} WRF cells")
        
        if ngrids > 0:
            # --- GEOGRAPHIC COVERAGE CONFIRMATION ---
            # Extract the actual lons/lats for the PALM domain area
            if 'lon' in ds_wrf.coords:
                lons_verify = ds_wrf.lon.isel(
                    west_east=slice(west_idx, east_idx + 1),
                    south_north=slice(south_idx, north_idx + 1)
                ).values
                lats_verify = ds_wrf.lat.isel(
                    west_east=slice(west_idx, east_idx + 1),
                    south_north=slice(south_idx, north_idx + 1)
                ).values
            else:
                lons_verify = ds_wrf.XLONG.isel(
                    time=0,
                    west_east=slice(west_idx, east_idx + 1),
                    south_north=slice(south_idx, north_idx + 1)
                ).values
                lats_verify = ds_wrf.XLAT.isel(
                    time=0,
                    west_east=slice(west_idx, east_idx + 1),
                    south_north=slice(south_idx, north_idx + 1)
                ).values

            print("\n--- GEOGRAPHIC COVERAGE CONFIRMATION ---")
            print(f"WRF cells covering PALM domain:")
            print(f"  Longitude range: {lons_verify.min():.5f} to {lons_verify.max():.5f}")
            print(f"  Latitude range:  {lats_verify.min():.5f} to {lats_verify.max():.5f}")
            print(f"  Your PALM Center: {centlon:.5f}, {centlat:.5f}")
            
            # Convert PALM domain boundaries to lat/lon for verification
            trans_palm2wgs = Transformer.from_proj(palm_proj, wgs_proj, always_xy=True)
            sw_lon, sw_lat = trans_palm2wgs.transform(west, south)
            ne_lon, ne_lat = trans_palm2wgs.transform(east, north)
            print(f"\nYour PALM Domain (lat/lon):")
            print(f"  SW corner: ({sw_lon:.5f}, {sw_lat:.5f})")
            print(f"  NE corner: ({ne_lon:.5f}, {ne_lat:.5f})")
            
            # Verify coverage
            if (lons_verify.min() <= sw_lon <= lons_verify.max() and
                lons_verify.min() <= ne_lon <= lons_verify.max() and
                lats_verify.min() <= sw_lat <= lats_verify.max() and
                lats_verify.min() <= ne_lat <= lats_verify.max()):
                print("WRF cells correctly cover your PALM domain")
            else:
                print("⚠ WARNING: Possible coordinate mismatch!")
            print("----------------------------------------\n")
            
            # Process radiation data for ALL cells within PALM domain
            print(f"\n Processing radiation data over {ngrids} WRF cells covering PALM domain...")
            
            rad_swdown, rad_lwdown, rad_swdiff = [], [], []
            wrf_times = ds_wrf.time.values
            
            for ts in tqdm(range(len(all_ts)), desc="  Radiation processing", unit="timestep"):
                current_time = all_ts[ts]
                time_diffs = np.abs(wrf_times - current_time)
                closest_idx = np.argmin(time_diffs)
                
                # Extract the cropped radiation data for this timestep
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
                
                # Average over all WRF cells in the PALM domain
                rad_swdown.append(np.mean(sw_cropped))
                rad_lwdown.append(np.mean(lw_cropped))
                rad_swdiff.append(np.mean(dif_cropped))
                
            
            rad_values_proc = [rad_swdown, rad_lwdown, rad_swdiff]
            
            print("\nRadiation processing complete!")
            print("Statistics over all timestamps (domain-averaged values):")
            print(f"SWDOWN (W/m²) - Mean: {np.mean(rad_swdown):.2f}, Min: {np.min(rad_swdown):.2f}, Max: {np.max(rad_swdown):.2f}")
            print(f"GLW    (W/m²) - Mean: {np.mean(rad_lwdown):.2f}, Min: {np.min(rad_lwdown):.2f}, Max: {np.max(rad_lwdown):.2f}")
            print(f"SWDDIF (W/m²) - Mean: {np.mean(rad_swdiff):.2f}, Min: {np.min(rad_swdiff):.2f}, Max: {np.max(rad_swdiff):.2f}")
            
    else:
        print(" Warning: Radiation variables (SWDOWN, GLW, SWDDIF) not found in WRF output")
        print(f"  Available variables in WRF output (first 20):")
        for i, var in enumerate(list(ds_wrf.variables.keys())[:20]):
            print(f"    {i+1:2d}. {var}")
        print("  Creating empty radiation arrays")
        rad_times_sec = []
        rad_values_proc = [[], [], []]
else:
    print("\n Radiation from WRF is disabled in config file")
    rad_times_sec = []
    rad_values_proc = [[], [], []]

print("\n" + "="*60)
    
#-------------------------------------------------------------------------------
# soil moisture and temperature
#-------------------------------------------------------------------------------
nc_output_name = f'dynamic_files/{case_name}_dynamic_{start_year}_{start_month}_{start_day}_{start_hour}'
print('Writing NetCDF file',flush=True)
nc_output = xr.Dataset()
res_origin = str(dx) + 'x' + str(dy) + ' m'
nc_output.attrs['description'] = f'Contains dynamic data from WRF mesoscale. WRF output file: {wrf_file}'
nc_output.attrs['author'] = 'Dongqi Lin (dongqi.lin@pg.canterbury.ac.nz)'
nc_output.attrs['history'] = 'Created at ' + time.ctime(time.time())
nc_output.attrs['source']= 'netCDF4 python'
nc_output.attrs['origin_lat'] = float(centlat)
nc_output.attrs['origin_lon'] = float(centlon)
nc_output.attrs['z'] = float(0)
nc_output.attrs['x'] = float(0)
nc_output.attrs['y'] = float(0)
nc_output.attrs['rotation_angle'] = float(0)
nc_output.attrs['origin_time'] =  str(all_ts[0]) + ' UTC'
nc_output.attrs['end_time'] =  str(all_ts[-1]) + ' UTC'


nc_output['x'] = xr.DataArray(x, dims=['x'], attrs={'units':'m'})
nc_output['y'] = xr.DataArray(y, dims=['y'], attrs={'units':'m'})
nc_output['z'] = xr.DataArray(z-z_origin, dims=['z'], attrs={'units':'m'})
nc_output['zsoil'] = xr.DataArray(dz_soil, dims=['zsoil'], attrs={'units':'m'})
nc_output['xu'] = xr.DataArray(xu, dims=['xu'], attrs={'units':'m'})
nc_output['yv'] = xr.DataArray(yv, dims=['yv'], attrs={'units':'m'})
nc_output['zw'] = xr.DataArray(zw-z_origin, dims=['zw'], attrs={'units':'m'})
nc_output['time'] = xr.DataArray(times_sec, dims=['time'], attrs={'units':'seconds'})


nc_output.to_netcdf(nc_output_name)
nc_output['init_soil_m'] = xr.DataArray(init_msoil, dims=['zsoil','y','x'],
         attrs={'units':'m^3/m^3','lod':np.int32(2), 'source':'WRF', 'long_name':'volumetric soil moisture (m^3/m^3)'})
nc_output['init_soil_t'] = xr.DataArray(init_tsoil, dims=['zsoil','y','x'],
         attrs={'units':'K', 'lod':np.int32(2), 'source':'WRF', 'long_name':'soil temperature (K)'})

# output boundary conditions to PALM input
# directions: 0 west, 1 east
#             0 south, 1 north

nc_output['init_atmosphere_pt'] = xr.DataArray(pt_init,dims=['z'],
         attrs={'units':'K', 'lod':np.int32(1), 'source':'WRF', 'res_origin':res_origin})
nc_output['ls_forcing_left_pt'] = xr.DataArray(ds_palm_we["pt"][:,:,:,0].data,dims=['time', 'z', 'y'],
         attrs={'units':'K', 'source':'WRF', 'res_origin':res_origin})
nc_output['ls_forcing_right_pt'] = xr.DataArray(ds_palm_we["pt"][:,:,:,-1].data,dims=['time', 'z', 'y'],
         attrs={'units':'K', 'source':'WRF', 'res_origin':res_origin})
nc_output['ls_forcing_south_pt'] = xr.DataArray(ds_palm_sn["pt"][:,:,0,:].data,dims=['time', 'z', 'x'],
         attrs={'units':'K', 'source':'WRF', 'res_origin':res_origin})
nc_output['ls_forcing_north_pt'] = xr.DataArray(ds_palm_sn["pt"][:,:,-1,:].data,dims=['time', 'z', 'x'],
         attrs={'units':'K', 'source':'WRF', 'res_origin':res_origin})
## top
nc_output['ls_forcing_top_pt'] = xr.DataArray(pt_top[:,:,:],dims=['time', 'y', 'x'],
         attrs={'units':'K', 'source':'WRF', 'res_origin':res_origin})

nc_output['init_atmosphere_qv'] = xr.DataArray(qv_init,dims=['z'],
         attrs={'units':'kg/kg', 'lod':np.int32(1), 'source':'WRF', 'res_origin':res_origin})
nc_output['ls_forcing_left_qv'] = xr.DataArray(ds_palm_we["QVAPOR"][:,:,:,0].data,dims=['time', 'z', 'y'],
         attrs={'units':'kg/kg', 'source':'WRF', 'res_origin':res_origin})
nc_output['ls_forcing_right_qv'] = xr.DataArray(ds_palm_we["QVAPOR"][:,:,:,-1].data,dims=['time', 'z', 'y'],
         attrs={'units':'kg/kg', 'source':'WRF', 'res_origin':res_origin})
nc_output['ls_forcing_south_qv'] = xr.DataArray(ds_palm_sn["QVAPOR"][:,:,0,:].data,dims=['time', 'z', 'x'],
         attrs={'units':'kg/kg', 'source':'WRF', 'res_origin':res_origin})
nc_output['ls_forcing_north_qv'] = xr.DataArray(ds_palm_sn["QVAPOR"][:,:,-1,:].data,dims=['time', 'z', 'x'],
         attrs={'units':'kg/kg', 'source':'WRF', 'res_origin':res_origin})
nc_output['ls_forcing_top_qv'] = xr.DataArray(qv_top[:,:,:],dims=['time', 'y', 'x'],
         attrs={'units':'kg/kg', 'source':'WRF', 'res_origin':res_origin})

nc_output['init_atmosphere_u'] = xr.DataArray(u_init,dims=['z'],
         attrs={'units':'m/s', 'lod':np.int32(1), 'source':'WRF', 'res_origin':res_origin})
nc_output['ls_forcing_left_u'] = xr.DataArray(ds_palm_we["U"][:,:,:,0].data,dims=['time', 'z', 'y'],
         attrs={'units':'m/s', 'source':'WRF', 'res_origin':res_origin})
nc_output['ls_forcing_right_u'] = xr.DataArray(ds_palm_we["U"][:,:,:,-1].data,dims=['time', 'z', 'y'],
         attrs={'units':'m/s', 'source':'WRF', 'res_origin':res_origin})
nc_output['ls_forcing_south_u'] = xr.DataArray(ds_palm_sn["U"][:,:,0,:].data,dims=['time', 'z', 'xu'],
         attrs={'units':'m/s', 'source':'WRF', 'res_origin':res_origin})
nc_output['ls_forcing_north_u'] = xr.DataArray(ds_palm_sn["U"][:,:,-1,:].data,dims=['time', 'z', 'xu'],
         attrs={'units':'m/s', 'source':'WRF', 'res_origin':res_origin})
nc_output['ls_forcing_top_u'] = xr.DataArray(u_top[:,:,:],dims=['time', 'y', 'xu'],
         attrs={'units':'m/s', 'source':'WRF', 'res_origin':res_origin})

nc_output['init_atmosphere_v'] = xr.DataArray(v_init,dims=['z'],
         attrs={'units':'m/s', 'lod':np.int32(1), 'source':'WRF', 'res_origin':res_origin})
nc_output['ls_forcing_left_v'] = xr.DataArray(ds_palm_we["V"][:,:,:,0].data,dims=['time', 'z', 'yv'],
         attrs={'units':'m/s', 'source':'WRF', 'res_origin':res_origin})
nc_output['ls_forcing_right_v'] = xr.DataArray(ds_palm_we["V"][:,:,:,-1].data,dims=['time', 'z', 'yv'],
         attrs={'units':'m/s', 'source':'WRF', 'res_origin':res_origin})
nc_output['ls_forcing_south_v'] = xr.DataArray(ds_palm_sn["V"][:,:,0,:].data,dims=['time', 'z', 'x'],
         attrs={'units':'m/s', 'source':'WRF', 'res_origin':res_origin})
nc_output['ls_forcing_north_v'] = xr.DataArray(ds_palm_sn["V"][:,:,-1,:].data,dims=['time', 'z', 'x'],
         attrs={'units':'m/s', 'source':'WRF', 'res_origin':res_origin})
nc_output['ls_forcing_top_v'] = xr.DataArray(v_top[:,:,:],dims=['time', 'yv', 'x'],
         attrs={'units':'m/s', 'source':'WRF', 'res_origin':res_origin})

nc_output['init_atmosphere_w'] = xr.DataArray(w_init,dims=['zw'],
         attrs={'units':'m/s', 'lod':np.int32(1), 'source':'WRF', 'res_origin':res_origin})
nc_output['ls_forcing_left_w'] = xr.DataArray(ds_palm_we["W"][:,:,:,0].data,dims=['time', 'zw', 'y'],
         attrs={'units':'m/s', 'source':'WRF', 'res_origin':res_origin})
nc_output['ls_forcing_right_w'] = xr.DataArray(ds_palm_we["W"][:,:,:,-1].data,dims=['time', 'zw', 'y'],
         attrs={'units':'m/s', 'source':'WRF', 'res_origin':res_origin})
nc_output['ls_forcing_south_w'] = xr.DataArray(ds_palm_sn["W"][:,:,0,:].data,dims=['time', 'zw', 'x'],
         attrs={'units':'m/s', 'source':'WRF', 'res_origin':res_origin})
nc_output['ls_forcing_north_w'] = xr.DataArray(ds_palm_sn["W"][:,:,-1,:].data,dims=['time', 'zw', 'x'],
         attrs={'units':'m/s', 'source':'WRF', 'res_origin':res_origin})
nc_output['ls_forcing_top_w'] = xr.DataArray(w_top[:,:,:],dims=['time', 'y', 'x'],
         attrs={'units':'m/s', 'source':'WRF', 'res_origin':res_origin})

nc_output['surface_forcing_surface_pressure'] = xr.DataArray(surface_pres.data, dims=['time'],
         attrs={'units':'Pa', 'lod':np.int32(1), 'source':'WRF', 'res_origin':res_origin})


#nc_output['ls_forcing_ug'] = xr.DataArray(ds_geostr["ug"].data,dims=['time','z'],
#         attrs={'units':'m/s', 'long_name':'u wind component geostrophic', 'source':'WRF', 'res_origin':res_origin})
#nc_output['ls_forcing_vg'] = xr.DataArray(ds_geostr["vg"].data,dims=['time','z'],
#         attrs={'units':'m/s', 'long_name':'v wind component geostrophic', 'source':'WRF', 'res_origin':res_origin})

# Add chemistry species to output
# Conversion factor from microgram/m3 to kg/m3
MICROGRAM_TO_KG = 1e-9

# Mapping from WRF variable names to PALM dynamic driver names
chem_name_mapping = {
    "hno3": "HNO3",
    "ho2": "HO2", 
    "ho": "OH",
    "no2": "NO2",
    "o3": "O3",
    "no": "NO",
    "qvapor": "H2O",
    "nh3": "NH3",
    "so2": "SO2",
    "co": "CO",
    "sulf": "H2SO4",
    "RH": "RH",
    "RO2": "RO2", 
    "RCHO": "RCHO",
    "OCSV": "OCSV",
    "OCNV": "OCNV",
    "PM10": "PM10",
    "PM2_5_DRY": "PM25"
}

# Use the original chem_species list for output (includes aggregated species and traffic variables)
for species in original_chem_species:
    # Get the output species name from mapping or use the species name directly
    if species.endswith('_traffic'):
        # For traffic variables, keep the suffix in lowercase
        base = species.replace('_traffic', '')
        if base in chem_name_mapping:
            output_species_name = f"{chem_name_mapping[base]}_traffic"
        else:
            output_species_name = f"{base.upper()}_traffic"
    else:
        output_species_name = chem_name_mapping.get(species, species.upper())
    
    # Add initial profiles
    if species in chem_init:
        # Convert PM values from microgram/m3 to kg/m3, gas species remain in ppm
        if species in ['PM10', 'PM2_5_DRY']:
            converted_data = chem_init[species].data * MICROGRAM_TO_KG
            nc_output[f'init_atmosphere_{output_species_name}'] = xr.DataArray(converted_data, dims=['z'],
                 attrs={'units':'kg/m3', 'lod':np.int32(1), 'source':'WRF-Chem', 'res_origin':res_origin})
        else:
            # For gas species - output in ppm (WRF-Chem outputs are in ppmv which is equivalent to ppm for trace gases)
            nc_output[f'init_atmosphere_{output_species_name}'] = xr.DataArray(chem_init[species].data, dims=['z'],
                 attrs={'units':'ppm', 'lod':np.int32(1), 'source':'WRF-Chem', 'res_origin':res_origin})
    
    # Add boundary conditions
    if species in ds_palm_we.data_vars:
        # West & East boundaries
        if species in ['PM10', 'PM2_5_DRY']:
            # Convert PM values from microgram/m3 to kg/m3
            left_data = ds_palm_we[species][:,:,:,0].data * MICROGRAM_TO_KG
            right_data = ds_palm_we[species][:,:,:,-1].data * MICROGRAM_TO_KG
            south_data = ds_palm_sn[species][:,:,0,:].data * MICROGRAM_TO_KG
            north_data = ds_palm_sn[species][:,:,-1,:].data * MICROGRAM_TO_KG
            top_data = chem_top[species] * MICROGRAM_TO_KG
            unit = "kg/m3"
        else:
            # For gas species - use ppm units
            left_data = ds_palm_we[species][:,:,:,0].data
            right_data = ds_palm_we[species][:,:,:,-1].data
            south_data = ds_palm_sn[species][:,:,0,:].data
            north_data = ds_palm_sn[species][:,:,-1,:].data
            top_data = chem_top[species]
            unit = "ppm"
        
        nc_output[f'ls_forcing_left_{output_species_name}'] = xr.DataArray(left_data, dims=['time', 'z', 'y'],
             attrs={'units':unit, 'source':'WRF-Chem', 'res_origin':res_origin})
        nc_output[f'ls_forcing_right_{output_species_name}'] = xr.DataArray(right_data, dims=['time', 'z', 'y'],
             attrs={'units':unit, 'source':'WRF-Chem', 'res_origin':res_origin})
        nc_output[f'ls_forcing_south_{output_species_name}'] = xr.DataArray(south_data, dims=['time', 'z', 'x'],
             attrs={'units':unit, 'source':'WRF-Chem', 'res_origin':res_origin})
        nc_output[f'ls_forcing_north_{output_species_name}'] = xr.DataArray(north_data, dims=['time', 'z', 'x'],
             attrs={'units':unit, 'source':'WRF-Chem', 'res_origin':res_origin})
        nc_output[f'ls_forcing_top_{output_species_name}'] = xr.DataArray(top_data, dims=['time', 'y', 'x'],
             attrs={'units':unit, 'source':'WRF-Chem', 'res_origin':res_origin})

#-------------------------------------------------------------------------------
# Add radiation data to output
#-------------------------------------------------------------------------------
if len(rad_times_sec) > 0 and len(rad_values_proc[0]) > 0:
    print("Adding radiation data to output file...")
    
    # Create radiation time dimension
    nc_output['time_rad'] = xr.DataArray(rad_times_sec, dims=['time_rad'], 
                                         attrs={'units':'seconds', 'long_name':'time since simulation start for radiation'})
    
    # Add radiation variables
    nc_output['rad_sw_in'] = xr.DataArray(rad_values_proc[0], dims=['time_rad'],
                                         attrs={'units':'W/m2', 'lod':1, 'long_name':'shortwave radiation incoming'})
    nc_output['rad_lw_in'] = xr.DataArray(rad_values_proc[1], dims=['time_rad'],
                                         attrs={'units':'W/m2', 'lod':1, 'long_name':'longwave radiation incoming'})
    nc_output['rad_sw_in_dif'] = xr.DataArray(rad_values_proc[2], dims=['time_rad'],
                                             attrs={'units':'W/m2', 'lod':1, 'long_name':'shortwave radiation incoming diffuse'})
    print("Radiation data added successfully")
else:
    print("No radiation data to add to output file")

for var in nc_output.data_vars:
    encoding = {var: {'dtype': 'float32', '_FillValue': -9999, 'zlib':True}}
    nc_output[var].to_netcdf(nc_output_name, encoding=encoding, mode='a')


print('Add to your *_p3d file: ' + '\n soil_temperature = ' +
              str([value for value in init_tsoil.mean(axis=(1,2))]) +
      '\n soil_moisture = ' + str([value for value in init_msoil.mean(axis=(1,2))])
        + '\n deep_soil_temperature = ' + str(deep_tsoil)+'\n')

with open('cfg_files/'+ case_name + '.cfg', "a") as cfg:
    cfg.write('Add to your *_p3d file: ' + '\n soil_temperature = ' +
              str([value for value in init_tsoil.mean(axis=(1,2))]) +
      '\n soil_moisture = ' + str([value for value in init_msoil.mean(axis=(1,2))])
        + '\n deep_soil_temperature = ' + str(deep_tsoil)+'\n')


end = datetime.now()
print('PALM dynamic input file is ready. Script duration: {}'.format(end - start))
print('Start time: '+str(all_ts[0]))
print('End time: '+str(all_ts[-1]))
print('Time step: '+str(times_sec[1]-times_sec[0])+' seconds')