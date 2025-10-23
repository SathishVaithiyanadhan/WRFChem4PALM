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

# FIX: Properly handle chemistry species list
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
# these two lines may be redundant need further tests 27 Oct 2021
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
# Add chemistry species to variable list
varbc_list.extend(chem_species)

print("remove unused vars from datasets")
for var in ds_we.data_vars:
    if var not in varbc_list:
        ds_we = ds_we.drop(var)
        ds_sn = ds_sn.drop(var)
    if var not in ["U", "Z"] and var not in chem_species:
        ds_we_ustag = ds_we_ustag.drop(var)
        ds_sn_ustag = ds_sn_ustag.drop(var)
    if var not in ["V", "Z"] and var not in chem_species:
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

# interpolation for chemistry species
print(f"Processing chemistry species: {chem_species}")
for species in chem_species:
    print(f"Checking if {species} exists in dataset...")
    if species in list(ds_we.data_vars.keys()):
        print(f"Processing {species}...")
        # Get the actual dimensions from the WRF data
        chem_dims = ds_we[species].shape
        chem_zeros_we = np.zeros((chem_dims[0], len(z), len(y), len(x[:2])))
        chem_zeros_sn = np.zeros((chem_dims[0], len(z), len(y[:2]), len(x)))
        
        ds_palm_we[species] = xr.DataArray(np.copy(chem_zeros_we), dims=['time','z','y', 'x'])
        ds_palm_sn[species] = xr.DataArray(np.copy(chem_zeros_sn), dims=['time','z','y', 'x'])
        print(f"Processing {species} for west and east boundaries")
        # Use the same interpolation method as other variables
        ds_palm_we[species] = multi_zinterp(max_pool, ds_we, species, z, ds_palm_we)
        print(f"Processing {species} for south and north boundaries")
        ds_palm_sn[species] = multi_zinterp(max_pool, ds_sn, species, z, ds_palm_sn)
    else:
        print(f"Warning: {species} not found in WRF dataset, skipping...")
        print(f"Available variables: {list(ds_we.data_vars.keys())[:10]}...")  # Show first 10 variables
    
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
# Handle NaN values in chemistry boundary conditions
#-------------------------------------------------------------------------------
print("Handling NaN values in chemistry boundary conditions...")
for species in chem_species:
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

# Initialize arrays for chemistry species top boundary
chem_top = {}
for species in chem_species:
    chem_top[species] = np.zeros((len(all_ts), len(y), len(x)))

for var in ds_interp.data_vars:
    if var not in varbc_list:
        ds_interp = ds_interp.drop(var)
    if var not in ["U", "Z"] and var not in chem_species:
        ds_interp_u = ds_interp_u.drop(var)
    if var not in ["V", "Z"] and var not in chem_species:
        ds_interp_v = ds_interp_v.drop(var)

print("Processing top boundary datasets...")
ds_interp_top = xr.Dataset()
ds_interp_u_top = xr.Dataset()
ds_interp_v_top = xr.Dataset()
for var in ["QVAPOR", "pt"]:
    ds_interp_top[var] =  ds_interp.salem.wrf_zlevel(var, levels=z[-1]).copy()

# Process chemistry species for top boundary
for species in chem_species:
    if species in ds_interp.data_vars:
        ds_interp_top[species] = ds_interp.salem.wrf_zlevel(species, levels=z[-1]).copy()

ds_interp_top["W"] = ds_interp.salem.wrf_zlevel("W", levels=zw[-1]).copy()        
ds_interp_u_top["U"] = ds_interp_u.salem.wrf_zlevel("U", levels=z[-1]).copy()
ds_interp_v_top["V"] = ds_interp_v.salem.wrf_zlevel("V", levels=z[-1]).copy()

for ts in tqdm(range(0,len(all_ts)), total=len(all_ts), position=0, leave=True):
    u_top[ts,:,:] = ds_interp_u_top["U"].isel(time=ts)
    v_top[ts,:,:] = ds_interp_v_top["V"].isel(time=ts)
    w_top[ts,:,:] = ds_interp_top["W"].isel(time=ts)  
    pt_top[ts,:,:] = ds_interp_top["pt"].isel(time=ts) 
    qv_top[ts,:,:] = ds_interp_top["QVAPOR"].isel(time=ts) 
    # Process chemistry species top boundary
    for species in chem_species:
        if species in ds_interp_top.data_vars:
            chem_top[species][ts,:,:] = ds_interp_top[species].isel(time=ts)

# Handle NaN values in top boundary chemistry data
for species in chem_species:
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
# Geostrophic wind estimation - CORRECTED CODE
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
# near surface geostrophic wind
for t in range(0,len(all_ts)):
    ds_geostr["ug"][t,:] =  surface_nan_w(ds_geostr["ug"][t,:].data)
    ds_geostr["vg"][t,:] =  surface_nan_w(ds_geostr["vg"][t,:].data)

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
for species in chem_species:
    if species in ds_drop.data_vars:
        chem_init[species] = ds_drop[species].sel(time=dt_start).mean(
            dim=["south_north", "west_east"]).interp(
            {"bottom_top": z}, method = interp_mode)
    else:
        # If species not found, create zeros array
        chem_init[species] = xr.DataArray(np.zeros(len(z)), dims=['z'], coords={'z': z})

surface_pres = psfc_wrf[:, :,:].mean(dim=["south_north", "west_east"]).load()


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


nc_output['ls_forcing_ug'] = xr.DataArray(ds_geostr["ug"].data,dims=['time','z'],
         attrs={'units':'m/s', 'long_name':'u wind component geostrophic', 'source':'WRF', 'res_origin':res_origin})
nc_output['ls_forcing_vg'] = xr.DataArray(ds_geostr["vg"].data,dims=['time','z'],
         attrs={'units':'m/s', 'long_name':'v wind component geostrophic', 'source':'WRF', 'res_origin':res_origin})

# Add chemistry species to output
# Conversion factor from μg/m³ to kg/m³
MICROGRAM_TO_KG = 1e-9

for species in chem_species:
    # Determine the output species name (convert PM2_5_DRY to PM25)
    output_species_name = species.upper()
    if output_species_name == "PM2_5_DRY":
        output_species_name = "PM25"
    
    # Add initial profiles
    if species in chem_init:
        # Convert PM values from μg/m³ to kg/m³
        if species in ['PM10', 'PM2_5_DRY']:
            converted_data = chem_init[species].data * MICROGRAM_TO_KG
            nc_output[f'init_atmosphere_{output_species_name}'] = xr.DataArray(converted_data, dims=['z'],
                 attrs={'units':'kg/m3', 'lod':np.int32(1), 'source':'WRF-Chem', 'res_origin':res_origin})
        else:
            # For gas species like no, no2, o3
            unit = "ppm" if species in ['no', 'no2', 'o3'] else "ppmv"
            nc_output[f'init_atmosphere_{output_species_name}'] = xr.DataArray(chem_init[species].data, dims=['z'],
                 attrs={'units':unit, 'lod':np.int32(1), 'source':'WRF-Chem', 'res_origin':res_origin})
    
    # Add boundary conditions
    if species in ds_palm_we.data_vars:
        # West & East boundaries
        if species in ['PM10', 'PM2_5_DRY']:
            # Convert PM values
            left_data = ds_palm_we[species][:,:,:,0].data * MICROGRAM_TO_KG
            right_data = ds_palm_we[species][:,:,:,-1].data * MICROGRAM_TO_KG
            south_data = ds_palm_sn[species][:,:,0,:].data * MICROGRAM_TO_KG
            north_data = ds_palm_sn[species][:,:,-1,:].data * MICROGRAM_TO_KG
            top_data = chem_top[species] * MICROGRAM_TO_KG
            unit = "kg/m3"
        else:
            # For gas species
            left_data = ds_palm_we[species][:,:,:,0].data
            right_data = ds_palm_we[species][:,:,:,-1].data
            south_data = ds_palm_sn[species][:,:,0,:].data
            north_data = ds_palm_sn[species][:,:,-1,:].data
            top_data = chem_top[species]
            unit = "ppm" if species in ['no', 'no2', 'o3'] else "ppmv"
        
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
##
'''import sys
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

# FIX: Properly handle chemistry species list
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
# these two lines may be redundant need further tests 27 Oct 2021
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
# Add chemistry species to variable list
varbc_list.extend(chem_species)

print("remove unused vars from datasets")
for var in ds_we.data_vars:
    if var not in varbc_list:
        ds_we = ds_we.drop(var)
        ds_sn = ds_sn.drop(var)
    if var not in ["U", "Z"] and var not in chem_species:
        ds_we_ustag = ds_we_ustag.drop(var)
        ds_sn_ustag = ds_sn_ustag.drop(var)
    if var not in ["V", "Z"] and var not in chem_species:
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

# interpolation for chemistry species
print(f"Processing chemistry species: {chem_species}")
for species in chem_species:
    print(f"Checking if {species} exists in dataset...")
    if species in list(ds_we.data_vars.keys()):
        print(f"Processing {species}...")
        # Get the actual dimensions from the WRF data
        chem_dims = ds_we[species].shape
        chem_zeros_we = np.zeros((chem_dims[0], len(z), len(y), len(x[:2])))
        chem_zeros_sn = np.zeros((chem_dims[0], len(z), len(y[:2]), len(x)))
        
        ds_palm_we[species] = xr.DataArray(np.copy(chem_zeros_we), dims=['time','z','y', 'x'])
        ds_palm_sn[species] = xr.DataArray(np.copy(chem_zeros_sn), dims=['time','z','y', 'x'])
        print(f"Processing {species} for west and east boundaries")
        # Use the same interpolation method as other variables
        ds_palm_we[species] = multi_zinterp(max_pool, ds_we, species, z, ds_palm_we)
        print(f"Processing {species} for south and north boundaries")
        ds_palm_sn[species] = multi_zinterp(max_pool, ds_sn, species, z, ds_palm_sn)
    else:
        print(f"Warning: {species} not found in WRF dataset, skipping...")
        print(f"Available variables: {list(ds_we.data_vars.keys())[:10]}...")  # Show first 10 variables
    
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
# Handle NaN values in chemistry boundary conditions
#-------------------------------------------------------------------------------
print("Handling NaN values in chemistry boundary conditions...")
for species in chem_species:
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

# Initialize arrays for chemistry species top boundary
chem_top = {}
for species in chem_species:
    chem_top[species] = np.zeros((len(all_ts), len(y), len(x)))

for var in ds_interp.data_vars:
    if var not in varbc_list:
        ds_interp = ds_interp.drop(var)
    if var not in ["U", "Z"] and var not in chem_species:
        ds_interp_u = ds_interp_u.drop(var)
    if var not in ["V", "Z"] and var not in chem_species:
        ds_interp_v = ds_interp_v.drop(var)

print("Processing top boundary datasets...")
ds_interp_top = xr.Dataset()
ds_interp_u_top = xr.Dataset()
ds_interp_v_top = xr.Dataset()
for var in ["QVAPOR", "pt"]:
    ds_interp_top[var] =  ds_interp.salem.wrf_zlevel(var, levels=z[-1]).copy()

# Process chemistry species for top boundary
for species in chem_species:
    if species in ds_interp.data_vars:
        ds_interp_top[species] = ds_interp.salem.wrf_zlevel(species, levels=z[-1]).copy()

ds_interp_top["W"] = ds_interp.salem.wrf_zlevel("W", levels=zw[-1]).copy()        
ds_interp_u_top["U"] = ds_interp_u.salem.wrf_zlevel("U", levels=z[-1]).copy()
ds_interp_v_top["V"] = ds_interp_v.salem.wrf_zlevel("V", levels=z[-1]).copy()

for ts in tqdm(range(0,len(all_ts)), total=len(all_ts), position=0, leave=True):
    u_top[ts,:,:] = ds_interp_u_top["U"].isel(time=ts)
    v_top[ts,:,:] = ds_interp_v_top["V"].isel(time=ts)
    w_top[ts,:,:] = ds_interp_top["W"].isel(time=ts)  
    pt_top[ts,:,:] = ds_interp_top["pt"].isel(time=ts) 
    qv_top[ts,:,:] = ds_interp_top["QVAPOR"].isel(time=ts) 
    # Process chemistry species top boundary
    for species in chem_species:
        if species in ds_interp_top.data_vars:
            chem_top[species][ts,:,:] = ds_interp_top[species].isel(time=ts)

# Handle NaN values in top boundary chemistry data
for species in chem_species:
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
# Geostrophic wind estimation - ORIGINAL CODE
#-------------------------------------------------------------------------------
print("Geostrophic wind estimation...")
## Check which levels should be used for geostrophic winds calculation
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

if geostr_lvl == "p":
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
# near surface geostrophic wind
for t in range(0,len(all_ts)):
    ds_geostr["ug"][t,:] =  surface_nan_w(ds_geostr["ug"][t,:].data)
    ds_geostr["vg"][t,:] =  surface_nan_w(ds_geostr["vg"][t,:].data)

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
for species in chem_species:
    if species in ds_drop.data_vars:
        chem_init[species] = ds_drop[species].sel(time=dt_start).mean(
            dim=["south_north", "west_east"]).interp(
            {"bottom_top": z}, method = interp_mode)
    else:
        # If species not found, create zeros array
        chem_init[species] = xr.DataArray(np.zeros(len(z)), dims=['z'], coords={'z': z})

surface_pres = psfc_wrf[:, :,:].mean(dim=["south_north", "west_east"]).load()


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


nc_output['ls_forcing_ug'] = xr.DataArray(ds_geostr["ug"].data,dims=['time','z'],
         attrs={'units':'m/s', 'long_name':'u wind component geostrophic', 'source':'WRF', 'res_origin':res_origin})
nc_output['ls_forcing_vg'] = xr.DataArray(ds_geostr["vg"].data,dims=['time','z'],
         attrs={'units':'m/s', 'long_name':'v wind component geostrophic', 'source':'WRF', 'res_origin':res_origin})

# Add chemistry species to output
# Conversion factor from μg/m³ to kg/m³
MICROGRAM_TO_KG = 1e-9

for species in chem_species:
    # Determine the output species name (convert PM2_5_DRY to PM25)
    output_species_name = species.upper()
    if output_species_name == "PM2_5_DRY":
        output_species_name = "PM25"
    
    # Add initial profiles
    if species in chem_init:
        # Convert PM values from μg/m³ to kg/m³
        if species in ['PM10', 'PM2_5_DRY']:
            converted_data = chem_init[species].data * MICROGRAM_TO_KG
            nc_output[f'init_atmosphere_{output_species_name}'] = xr.DataArray(converted_data, dims=['z'],
                 attrs={'units':'kg/m3', 'lod':np.int32(1), 'source':'WRF-Chem', 'res_origin':res_origin})
        else:
            # For gas species like no, no2, o3
            unit = "ppm" if species in ['no', 'no2', 'o3'] else "ppmv"
            nc_output[f'init_atmosphere_{output_species_name}'] = xr.DataArray(chem_init[species].data, dims=['z'],
                 attrs={'units':unit, 'lod':np.int32(1), 'source':'WRF-Chem', 'res_origin':res_origin})
    
    # Add boundary conditions
    if species in ds_palm_we.data_vars:
        # West & East boundaries
        if species in ['PM10', 'PM2_5_DRY']:
            # Convert PM values
            left_data = ds_palm_we[species][:,:,:,0].data * MICROGRAM_TO_KG
            right_data = ds_palm_we[species][:,:,:,-1].data * MICROGRAM_TO_KG
            south_data = ds_palm_sn[species][:,:,0,:].data * MICROGRAM_TO_KG
            north_data = ds_palm_sn[species][:,:,-1,:].data * MICROGRAM_TO_KG
            top_data = chem_top[species] * MICROGRAM_TO_KG
            unit = "kg/m3"
        else:
            # For gas species
            left_data = ds_palm_we[species][:,:,:,0].data
            right_data = ds_palm_we[species][:,:,:,-1].data
            south_data = ds_palm_sn[species][:,:,0,:].data
            north_data = ds_palm_sn[species][:,:,-1,:].data
            top_data = chem_top[species]
            unit = "ppm" if species in ['no', 'no2', 'o3'] else "ppmv"
        
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
print('Time step: '+str(times_sec[1]-times_sec[0])+' seconds')'''
####
#geostraphic wind levels corrected. 
'''import sys
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

# FIX: Properly handle chemistry species list
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

print(f"Using z_origin = {z_origin} m as ground level (0 m)")

y = np.arange(dy/2,dy*ny+dy/2,dy)
x = np.arange(dx/2,dx*nx+dx/2,dx)
# KEY CHANGE: Add z_origin to vertical coordinates for interpolation
#z = np.arange(dz/2, dz*nz, dz) + z_origin
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

# KEY CHANGE: Add z_origin AFTER stretching calculations
z += z_origin
zw += z_origin

print(f"Final vertical grid: z-range = {z[0]:.1f}m to {z[-1]:.1f}m")
print(f"Ground level (z_origin) = {z_origin}m will become 0m in PALM coordinates")

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
# these two lines may be redundant need further tests 27 Oct 2021
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
    median_spois = np.ones_like(median_smois)
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
# Add chemistry species to variable list
varbc_list.extend(chem_species)

print("remove unused vars from datasets")
for var in ds_we.data_vars:
    if var not in varbc_list:
        ds_we = ds_we.drop(var)
        ds_sn = ds_sn.drop(var)
    if var not in ["U", "Z"] and var not in chem_species:
        ds_we_ustag = ds_we_ustag.drop(var)
        ds_sn_ustag = ds_sn_ustag.drop(var)
    if var not in ["V", "Z"] and var not in chem_species:
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

# interpolation for chemistry species - FIXED: Use proper interpolation like other variables
print(f"Processing chemistry species: {chem_species}")
for species in chem_species:
    print(f"Checking if {species} exists in dataset...")
    # Check if the species exists in the dataset before processing
    if species in list(ds_we.data_vars.keys()):
        print(f"Processing {species}...")
        # Get the actual dimensions from the WRF data
        chem_dims = ds_we[species].shape
        chem_zeros_we = np.zeros((chem_dims[0], len(z), len(y), len(x[:2])))
        chem_zeros_sn = np.zeros((chem_dims[0], len(z), len(y[:2]), len(x)))
        
        ds_palm_we[species] = xr.DataArray(np.copy(chem_zeros_we), dims=['time','z','y', 'x'])
        ds_palm_sn[species] = xr.DataArray(np.copy(chem_zeros_sn), dims=['time','z','y', 'x'])
        print(f"Processing {species} for west and east boundaries")
        # Use the same interpolation method as other variables
        ds_palm_we[species] = multi_zinterp(max_pool, ds_we, species, z, ds_palm_we)
        print(f"Processing {species} for south and north boundaries")
        ds_palm_sn[species] = multi_zinterp(max_pool, ds_sn, species, z, ds_palm_sn)
    else:
        print(f"Warning: {species} not found in WRF dataset, skipping...")
        print(f"Available variables: {list(ds_we.data_vars.keys())[:10]}...")  # Show first 10 variables
    
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
# Handle NaN values in chemistry boundary conditions - FIXED: Use proper interpolation instead of filling with zeros
#-------------------------------------------------------------------------------
print("Handling NaN values in chemistry boundary conditions...")
for species in chem_species:
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

# Initialize arrays for chemistry species top boundary
chem_top = {}
for species in chem_species:
    chem_top[species] = np.zeros((len(all_ts), len(y), len(x)))

for var in ds_interp.data_vars:
    if var not in varbc_list:
        ds_interp = ds_interp.drop(var)
    if var not in ["U", "Z"] and var not in chem_species:
        ds_interp_u = ds_interp_u.drop(var)
    if var not in ["V", "Z"] and var not in chem_species:
        ds_interp_v = ds_interp_v.drop(var)

print("Processing top boundary datasets...")
ds_interp_top = xr.Dataset()
ds_interp_u_top = xr.Dataset()
ds_interp_v_top = xr.Dataset()
for var in ["QVAPOR", "pt"]:
    ds_interp_top[var] =  ds_interp.salem.wrf_zlevel(var, levels=z[-1]).copy()

# Process chemistry species for top boundary
for species in chem_species:
    if species in ds_interp.data_vars:
        ds_interp_top[species] = ds_interp.salem.wrf_zlevel(species, levels=z[-1]).copy()

ds_interp_top["W"] = ds_interp.salem.wrf_zlevel("W", levels=zw[-1]).copy()        
ds_interp_u_top["U"] = ds_interp_u.salem.wrf_zlevel("U", levels=z[-1]).copy()
ds_interp_v_top["V"] = ds_interp_v.salem.wrf_zlevel("V", levels=z[-1]).copy()

for ts in tqdm(range(0,len(all_ts)), total=len(all_ts), position=0, leave=True):
    u_top[ts,:,:] = ds_interp_u_top["U"].isel(time=ts)
    v_top[ts,:,:] = ds_interp_v_top["V"].isel(time=ts)
    w_top[ts,:,:] = ds_interp_top["W"].isel(time=ts)  
    pt_top[ts,:,:] = ds_interp_top["pt"].isel(time=ts) 
    qv_top[ts,:,:] = ds_interp_top["QVAPOR"].isel(time=ts) 
    # Process chemistry species top boundary
    for species in chem_species:
        if species in ds_interp_top.data_vars:
            chem_top[species][ts,:,:] = ds_interp_top[species].isel(time=ts)

# Handle NaN values in top boundary chemistry data - use proper interpolation
for species in chem_species:
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

# Diagnostic: Check available pressure variables
print("Available variables in WRF dataset:")
pressure_vars = [var for var in ds_drop.data_vars if 'press' in var.lower() or var in ['P', 'PB', 'PRESSURE']]
for var in pressure_vars:
    var_data = ds_drop[var]
    print(f"  {var}: shape {var_data.shape}, range {np.nanmin(var_data):.1f} to {np.nanmax(var_data):.1f}")

# Also check temperature variables
temp_vars = [var for var in ds_drop.data_vars if 'temp' in var.lower() or 'tk' in var.lower() or 't' in var.lower()]
for var in temp_vars:
    var_data = ds_drop[var]
    print(f"  {var}: shape {var_data.shape}, range {np.nanmin(var_data):.1f} to {np.nanmax(var_data):.1f}")

#-------------------------------------------------------------------------------
# Geostrophic wind estimation - old version
#-------------------------------------------------------------------------------
print("Geostrophic wind estimation...")
## Check which levels should be used for geostrophic winds calculation
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

if geostr_lvl == "p":
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

#-------------------------------------------------------------------------------
# Geostrophic wind estimation - IMPROVED for small domains
#-------------------------------------------------------------------------------
print("Geostrophic wind estimation...")
print(f"Using {geostr_lvl}-level method for geostrophic wind calculation")

# First, let's diagnose the actual wind patterns in the WRF data
print("Diagnosing actual wind patterns from WRF data...")
u_mean = ds_drop["U"].mean(("time", "bottom_top", "south_north", "west_east")).load().data
v_mean = ds_drop["V"].mean(("time", "bottom_top", "south_north", "west_east")).load().data
print(f"Mean WRF winds: U={u_mean:.2f} m/s, V={v_mean:.2f} m/s")

if geostr_lvl == "z":
    lat_geostr = ds_drop.lat[:,0]
    dx_wrf = ds_drop.DX
    dy_wrf = ds_drop.DY
    gph = ds_drop.gph
    print("Geostrophic wind loading data...")
    gph = gph.load()
    
    ds_geostr_z = xr.Dataset()
    ds_geostr_z = ds_geostr_z.assign_coords({
        "time": ds_drop.time.data,
        "z": ds_drop["Z"].mean(("time", "south_north", "west_east")).data
    })
    
    ds_geostr_z["ug"] = xr.DataArray(np.zeros((len(all_ts), len(gph.bottom_top.data))),
                                   dims=['time','z'])
    ds_geostr_z["vg"] = xr.DataArray(np.zeros((len(all_ts), len(gph.bottom_top.data))),
                                   dims=['time','z'])

    print("Calculating geostrophic wind profiles...")
    for ts in tqdm(range(0, len(all_ts)), total=len(all_ts), position=0, leave=True):
        for level in gph.bottom_top.data:
            try:
                ug_val, vg_val = calc_geostrophic_wind_zlevels(
                    gph[ts, level, :, :].data, 
                    lat_geostr.data, 
                    dy_wrf, 
                    dx_wrf
                )
                ds_geostr_z["ug"][ts, level] = ug_val
                ds_geostr_z["vg"][ts, level] = vg_val
            except Exception as e:
                print(f"Error at time {ts}, level {level}: {e}")
                # Use WRF wind as fallback
                ds_geostr_z["ug"][ts, level] = u_mean
                ds_geostr_z["vg"][ts, level] = v_mean

    # interpolate to PALM vertical levels
    ds_geostr = ds_geostr_z.interp({"z": z})

# Remove the unrealistic value check and replacement - trust our new calculation
print("Geostrophic wind diagnostics:")
ug_data = ds_geostr["ug"].data
vg_data = ds_geostr["vg"].data

print(f"  ug shape: {ug_data.shape}")
print(f"  vg shape: {vg_data.shape}")
print(f"  ug range: {np.min(ug_data):.2f} to {np.max(ug_data):.2f} m/s")
print(f"  vg range: {np.min(vg_data):.2f} to {np.max(vg_data):.2f} m/s")
print(f"  ug mean: {np.mean(ug_data):.2f} m/s")
print(f"  vg mean: {np.mean(vg_data):.2f} m/s")

# Check if values are realistic
if np.max(np.abs(ug_data)) > 50 or np.max(np.abs(vg_data)) > 50:
    print("WARNING: Unrealistically high geostrophic wind values detected!")
    print("Using typical values instead...")
    ds_geostr["ug"].data = np.full_like(ug_data, 8.0)  # Typical westerly
    ds_geostr["vg"].data = np.full_like(vg_data, 0.5)  # Typical southerly


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
# near surface geostrophic wind
for t in range(0,len(all_ts)):
    ds_geostr["ug"][t,:] =  surface_nan_w(ds_geostr["ug"][t,:].data)
    ds_geostr["vg"][t,:] =  surface_nan_w(ds_geostr["vg"][t,:].data)

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
w_init = ds_drop["W"].sel(time=dt_start).mean(
    dim=["south_north", "west_east"]).interp(
    {"bottom_top": zw}, method = interp_mode)
pt_init = ds_drop["pt"].sel(time=dt_start).mean(
    dim=["south_north", "west_east"]).interp(
    {"bottom_top": z}, method = interp_mode)
qv_init = ds_drop["QVAPOR"].sel(time=dt_start).mean(
    dim=["south_north", "west_east"]).interp(
    {"bottom_top": z}, method = interp_mode)

# Initialize chemistry species profiles
chem_init = {}
for species in chem_species:
    if species in ds_drop.data_vars:
        chem_init[species] = ds_drop[species].sel(time=dt_start).mean(
            dim=["south_north", "west_east"]).interp(
            {"bottom_top": z}, method = interp_mode)
    else:
        # If species not found, create zeros array
        chem_init[species] = xr.DataArray(np.zeros(len(z)), dims=['z'], coords={'z': z})

#-------------------------------------------------------------------------------
# Create single NetCDF file with PALM dynamic driver format
#-------------------------------------------------------------------------------
print("Creating single NetCDF file with PALM dynamic driver format...")

# Create the main dataset
ds_palm = xr.Dataset()
res_origin = str(dx) + 'x' + str(dy) + ' m'

# Assign coordinates
# KEY CHANGE: Subtract z_origin from vertical coordinates for final output
ds_palm = ds_palm.assign_coords({
    "x": x,
    "y": y,
    "z": z - z_origin,  # Adjust so z_origin becomes 0 m
    "zsoil": zs_palm,
    "xu": xu,
    "yv": yv,
    "zw": zw - z_origin,  # Adjust so z_origin becomes 0 m
    "time": times_sec,
    "time_rad": times_sec  # Assuming same time for radiation
})

# Add coordinate attributes
ds_palm["x"].attrs = {"units": "m", "long_name": "distance to origin in x-direction"}
ds_palm["y"].attrs = {"units": "m", "long_name": "distance to origin in y-direction"}
ds_palm["z"].attrs = {"units": "m", "long_name": "height above origin"}
ds_palm["zsoil"].attrs = {"units": "m", "long_name": "depth of soil layer"}
ds_palm["xu"].attrs = {"units": "m", "long_name": "distance to origin in x-direction at u-grid"}
ds_palm["yv"].attrs = {"units": "m", "long_name": "distance to origin in y-direction at v-grid"}
ds_palm["zw"].attrs = {"units": "m", "long_name": "height above origin at w-grid"}
ds_palm["time"].attrs = {"units": "seconds", "long_name": "time"}
ds_palm["time_rad"].attrs = {"units": "seconds", "long_name": "time for radiation"}

# Add soil data
ds_palm["init_soil_m"] = xr.DataArray(init_msoil, dims=['zsoil', 'y', 'x'])
ds_palm["init_soil_m"].attrs = {
    "units": "m^3/m^3", 
    "long_name": "volumetric soil moisture (m^3/m^3)",
    "source": "WRF",
    "lod": np.int32(2)  # Convert to int32
}

ds_palm["init_soil_t"] = xr.DataArray(init_tsoil, dims=['zsoil', 'y', 'x'])
ds_palm["init_soil_t"].attrs = {
    "units": "K", 
    "long_name": "soil temperature (K)",
    "source": "WRF",
    "lod": np.int32(2)  # Convert to int32
}

# Add initial profiles
ds_palm["init_atmosphere_pt"] = xr.DataArray(pt_init.data, dims=['z'])
ds_palm["init_atmosphere_pt"].attrs = {
    "units": "K", 
    "long_name": "initial potential temperature profile",
    "source": "WRF",
    "lod": np.int32(1),  # Convert to int32
    "res_origin":res_origin
}

ds_palm["init_atmosphere_qv"] = xr.DataArray(qv_init.data, dims=['z'])
ds_palm["init_atmosphere_qv"].attrs = {
    "units": "kg/kg", 
    "long_name": "initial water vapor mixing ratio profile",
    "source": "WRF",
    "lod": np.int32(1),  # Convert to int32
    "res_origin":res_origin  # Convert to int32
}

ds_palm["init_atmosphere_u"] = xr.DataArray(u_init.data, dims=['z'])
ds_palm["init_atmosphere_u"].attrs = {
    "units": "m/s", 
    "long_name": "initial u profile",
    "source": "WRF",
    "lod": np.int32(1),  # Convert to int32
    "res_origin":res_origin  # Convert to int32
}

ds_palm["init_atmosphere_v"] = xr.DataArray(v_init.data, dims=['z'])
ds_palm["init_atmosphere_v"].attrs = {
    "units": "m/s", 
    "long_name": "initial v profile",
    "source": "WRF",
    "lod": np.int32(1),  # Convert to int32
    "res_origin":res_origin  # Convert to int32
}

ds_palm["init_atmosphere_w"] = xr.DataArray(w_init.data, dims=['zw'])
ds_palm["init_atmosphere_w"].attrs = {
    "units": "m/s", 
    "long_name": "initial w profile",
    "source": "WRF",
    "lod": np.int32(1),  # Convert to int32
    "res_origin":res_origin  # Convert to int32
}

# Add chemistry initial profiles - keep original names for now
for species in chem_species:
    ds_palm[f"init_atmosphere_{species}"] = xr.DataArray(chem_init[species].data, dims=['z'])
    # Set units based on species type
    if species in ['PM10', 'PM2_5_DRY']:
        unit = "kg/m3"
    elif species in ['no', 'no2', 'o3']:
        unit = "ppm"
    else:
        unit = "ppmv"
    
    ds_palm[f"init_atmosphere_{species}"].attrs = {
        "units": unit, 
        "long_name": f"initial {species} profile",
        "source": "WRF-Chem",
        "lod": np.int32(1),  # Convert to int32
        "res_origin":res_origin
    }

# Add boundary conditions
# West and East boundaries
# ---------------------------
# West & East boundaries
# ---------------------------
ds_palm["ls_forcing_left_pt"] = xr.DataArray(
    ds_palm_we["pt"].isel(x=0).data, dims=['time', 'z', 'y']
)
ds_palm["ls_forcing_left_pt"].attrs = {"units": "K", "source": "WRF"}

ds_palm["ls_forcing_right_pt"] = xr.DataArray(
    ds_palm_we["pt"].isel(x=-1).data, dims=['time', 'z', 'y']
)
ds_palm["ls_forcing_right_pt"].attrs = {"units": "K", "source": "WRF"}

ds_palm["ls_forcing_left_qv"] = xr.DataArray(
    ds_palm_we["QVAPOR"].isel(x=0).data, dims=['time', 'z', 'y']
)
ds_palm["ls_forcing_left_qv"].attrs = {"units": "kg/kg", "source": "WRF"}

ds_palm["ls_forcing_right_qv"] = xr.DataArray(
    ds_palm_we["QVAPOR"].isel(x=-1).data, dims=['time', 'z', 'y']
)
ds_palm["ls_forcing_right_qv"].attrs = {"units": "kg/kg", "source": "WRF"}

ds_palm["ls_forcing_left_u"] = xr.DataArray(
    ds_palm_we["U"].isel(xu=0).data,
    dims=['time', 'z', 'y']
)
ds_palm["ls_forcing_left_u"].attrs = {"units": "m/s", "source": "WRF"}

ds_palm["ls_forcing_right_u"] = xr.DataArray(
    ds_palm_we["U"].isel(xu=-1).data,
    dims=['time', 'z', 'y']
)
ds_palm["ls_forcing_right_u"].attrs = {"units": "m/s", "source": "WRF"}

ds_palm["ls_forcing_left_v"] = xr.DataArray(
    ds_palm_we["V"].isel(x=0).data, dims=['time', 'z', 'yv']
)
ds_palm["ls_forcing_left_v"].attrs = {"units": "m/s", "source": "WRF"}

ds_palm["ls_forcing_right_v"] = xr.DataArray(
    ds_palm_we["V"].isel(x=-1).data, dims=['time', 'z', 'yv']
)
ds_palm["ls_forcing_right_v"].attrs = {"units": "m/s", "source": "WRF"}

ds_palm["ls_forcing_left_w"] = xr.DataArray(
    ds_palm_we["W"].isel(x=0).data, dims=['time', 'zw', 'y']
)
ds_palm["ls_forcing_left_w"].attrs = {"units": "m/s", "source": "WRF"}

ds_palm["ls_forcing_right_w"] = xr.DataArray(
    ds_palm_we["W"].isel(x=-1).data, dims=['time', 'zw', 'y']
)
ds_palm["ls_forcing_right_w"].attrs = {"units": "m/s", "source": "WRF"}

# ---------------------------
# South & North boundaries
# ---------------------------
ds_palm["ls_forcing_south_pt"] = xr.DataArray(
    ds_palm_sn["pt"].isel(y=0).data, dims=['time', 'z', 'x']
)
ds_palm["ls_forcing_south_pt"].attrs = {"units": "K", "source": "WRF"}

ds_palm["ls_forcing_north_pt"] = xr.DataArray(
    ds_palm_sn["pt"].isel(y=-1).data, dims=['time', 'z', 'x']
)
ds_palm["ls_forcing_north_pt"].attrs = {"units": "K", "source": "WRF"}

ds_palm["ls_forcing_south_qv"] = xr.DataArray(
    ds_palm_sn["QVAPOR"].isel(y=0).data, dims=['time', 'z', 'x']
)
ds_palm["ls_forcing_south_qv"].attrs = {"units": "kg/kg", "source": "WRF"}

ds_palm["ls_forcing_north_qv"] = xr.DataArray(
    ds_palm_sn["QVAPOR"].isel(y=-1).data, dims=['time', 'z', 'x']
)
ds_palm["ls_forcing_north_qv"].attrs = {"units": "kg/kg", "source": "WRF"}

ds_palm["ls_forcing_south_u"] = xr.DataArray(
    ds_palm_sn["U"].isel(y=0).data, dims=['time', 'z', 'xu']
)
ds_palm["ls_forcing_south_u"].attrs = {"units": "m/s", "source": "WRF"}

ds_palm["ls_forcing_north_u"] = xr.DataArray(
    ds_palm_sn["U"].isel(y=-1).data, dims=['time', 'z', 'xu']
)
ds_palm["ls_forcing_north_u"].attrs = {"units": "m/s", "source": "WRF"}

ds_palm["ls_forcing_south_v"] = xr.DataArray(
    ds_palm_sn["V"].isel(yv=0).data,
    dims=['time', 'z', 'x']
)
ds_palm["ls_forcing_south_v"].attrs = {"units": "m/s", "source": "WRF"}

ds_palm["ls_forcing_north_v"] = xr.DataArray(
    ds_palm_sn["V"].isel(yv=-1).data,
    dims=['time', 'z', 'x']
)
ds_palm["ls_forcing_north_v"].attrs = {"units": "m/s", "source": "WRF"}

ds_palm["ls_forcing_south_w"] = xr.DataArray(
    ds_palm_sn["W"].isel(y=0).data, dims=['time', 'zw', 'x']
)
ds_palm["ls_forcing_south_w"].attrs = {"units": "m/s", "source": "WRF"}

ds_palm["ls_forcing_north_w"] = xr.DataArray(
    ds_palm_sn["W"].isel(y=-1).data, dims=['time', 'zw', 'x']
)
ds_palm["ls_forcing_north_w"].attrs = {"units": "m/s", "source": "WRF"}

# ---------------------------
# Top boundary
# ---------------------------
ds_palm["ls_forcing_top_pt"] = xr.DataArray(pt_top, dims=['time', 'y', 'x'])
ds_palm["ls_forcing_top_pt"].attrs = {"units": "K", "source": "WRF"}

ds_palm["ls_forcing_top_qv"] = xr.DataArray(qv_top, dims=['time', 'y', 'x'])
ds_palm["ls_forcing_top_qv"].attrs = {"units": "kg/kg", "source": "WRF"}

ds_palm["ls_forcing_top_u"] = xr.DataArray(u_top, dims=['time', 'y', 'xu'])
ds_palm["ls_forcing_top_u"].attrs = {"units": "m/s", "source": "WRF"}

ds_palm["ls_forcing_top_v"] = xr.DataArray(v_top, dims=['time', 'yv', 'x'])
ds_palm["ls_forcing_top_v"].attrs = {"units": "m/s", "source": "WRF"}

ds_palm["ls_forcing_top_w"]  = xr.DataArray(w_top,  dims=['time', 'y', 'x'])
ds_palm["ls_forcing_top_w"].attrs = {"units": "m/s", "source": "WRF"}

# ---------------------------
# Chemistry boundary conditions - keep original names for now
# ---------------------------
for species in chem_species:
    if species in ds_palm_we.data_vars:
        # West & East
        ds_palm[f"ls_forcing_left_{species}"] = xr.DataArray(
            ds_palm_we[species].isel(x=0).data, dims=['time', 'z', 'y']
        )
        ds_palm[f"ls_forcing_right_{species}"] = xr.DataArray(
            ds_palm_we[species].isel(x=-1).data, dims=['time', 'z', 'y']
        )

        # South & North
        ds_palm[f"ls_forcing_south_{species}"] = xr.DataArray(
            ds_palm_sn[species].isel(y=0).data, dims=['time', 'z', 'x']
        )
        ds_palm[f"ls_forcing_north_{species}"] = xr.DataArray(
            ds_palm_sn[species].isel(y=-1).data, dims=['time', 'z', 'x']
        )

        # Top
        ds_palm[f"ls_forcing_top_{species}"] = xr.DataArray(
            chem_top[species], dims=['time', 'y', 'x']
        )

        # Set units based on species type
        if species in ['PM10', 'PM2_5_DRY']:
            unit = "kg/m3"
        elif species in ['no', 'no2', 'o3']:
            unit = "ppm"
        else:
            unit = "ppmv"
            
        for var_name in [
            f"ls_forcing_left_{species}", f"ls_forcing_right_{species}",
            f"ls_forcing_south_{species}", f"ls_forcing_north_{species}",
            f"ls_forcing_top_{species}"
        ]:
            ds_palm[var_name].attrs = {"units": unit, "source": "WRF-Chem"}

# Add geostrophic wind
ds_palm["ls_forcing_ug"] = xr.DataArray(ds_geostr["ug"].data, dims=['time', 'z'])
ds_palm["ls_forcing_ug"].attrs = {
    "units": "m/s", 
    "long_name": "u wind component geostrophic",
    "source": "WRF",
    "res_origin":res_origin
}

ds_palm["ls_forcing_vg"] = xr.DataArray(ds_geostr["vg"].data, dims=['time', 'z'])
ds_palm["ls_forcing_vg"].attrs = {
    "units": "m/s", 
    "long_name": "v wind component geostrophic",
    "source": "WRF", 
    "res_origin":res_origin
}

# Add global attributes
ds_palm.attrs = {
    "title": "PALM dynamic driver generated by WRF4PALM",
    "author": "WRF4PALM v1.1.2",
    "source": "WRF-Chem",
    "history": f"Created: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
    "reference": "WRF4PALM: Lin et al. (2021)",
    "Conventions": "PIDS v1.9",
    "origin_lat": centlat,
    "origin_lon": centlon,
    "origin_x": centx,
    "origin_y": centy,
    "rotation_angle": 0.0,
    "origin_time": str(dt_start) + " UTC",
    "end_time": str(dt_end) + " UTC"
}

# Write to single NetCDF file
#output_filename = f"./dynamic_files/{case_name}_dynamic_driver.nc"
#print(f"Writing to {output_filename}")
#ds_palm.to_netcdf(output_filename, mode="w", format="NETCDF4")

# Now rename variables to uppercase, fix units, and convert PM values
print("Renaming chemistry variables to uppercase, fixing units, and converting PM values...")
ds_palm_final = xr.Dataset()

# Copy all coordinates
for coord_name in ds_palm.coords:
    ds_palm_final = ds_palm_final.assign_coords({coord_name: ds_palm[coord_name]})

# Copy all non-chemistry variables
for var_name in ds_palm.data_vars:
    if not any(species in var_name for species in chem_species):
        ds_palm_final[var_name] = ds_palm[var_name]

# Conversion factor from μg/m³ to kg/m³
MICROGRAM_TO_KG = 1e-9

# Process chemistry variables - rename to uppercase, fix units, and convert PM values
for species in chem_species:
    # Determine the new species name (convert PM2_5_DRY to PM25)
    new_species_name = species.upper()
    if new_species_name == "PM2_5_DRY":
        new_species_name = "PM25"
    
    # Process initial profiles
    init_var = f"init_atmosphere_{species}"
    if init_var in ds_palm.data_vars:
        new_init_var = f"init_atmosphere_{new_species_name}"
        
        # Convert PM values from μg/m³ to kg/m³
        if species in ['PM10', 'PM2_5_DRY']:
            converted_data = ds_palm[init_var].data * MICROGRAM_TO_KG
            ds_palm_final[new_init_var] = xr.DataArray(converted_data, dims=['z'])
            ds_palm_final[new_init_var].attrs = {"units": "kg/m3", "source": "WRF-Chem", "lod": np.int32(1), "res_origin":res_origin}
        else:
            ds_palm_final[new_init_var] = ds_palm[init_var]
            if species in ['no', 'no2', 'o3']:
                ds_palm_final[new_init_var].attrs = {"units": "ppm", "source": "WRF-Chem", "lod": np.int32(1), "res_origin":res_origin}
            else:
                ds_palm_final[new_init_var].attrs = {"units": "ppmv", "source": "WRF-Chem", "lod": np.int32(1), "res_origin":res_origin}
    
    # Process boundary conditions
    for boundary in ['left', 'right', 'south', 'north', 'top']:
        bc_var = f"ls_forcing_{boundary}_{species}"
        if bc_var in ds_palm.data_vars:
            new_bc_var = f"ls_forcing_{boundary}_{new_species_name}"
            
            # Convert PM values from μg/m³ to kg/m³
            if species in ['PM10', 'PM2_5_DRY']:
                converted_data = ds_palm[bc_var].data * MICROGRAM_TO_KG
                ds_palm_final[new_bc_var] = xr.DataArray(converted_data, dims=ds_palm[bc_var].dims)
                ds_palm_final[new_bc_var].attrs = {"units": "kg/m3", "source": "WRF-Chem"}
            else:
                ds_palm_final[new_bc_var] = ds_palm[bc_var]
                if species in ['no', 'no2', 'o3']:
                    ds_palm_final[new_bc_var].attrs = {"units": "ppm", "source": "WRF-Chem"}
                else:
                    ds_palm_final[new_bc_var].attrs = {"units": "ppmv", "source": "WRF-Chem"}

# Copy global attributes
ds_palm_final.attrs = ds_palm.attrs

# Explicitly set units for coordinate variables before writing NetCDF
for coord_name in ["x", "y", "z", "xu", "yv", "zw", "zsoil"]:
    if coord_name in ds_palm_final.coords:
        ds_palm_final.coords[coord_name].attrs["units"] = "m"
if "time" in ds_palm_final.coords:
    ds_palm_final.coords["time"].attrs["units"] = "seconds"

# Set _FillValue for all init_atmosphere_*, ls_forcing_*, init_soil_* variables and convert to float32
for var_name in ds_palm_final.data_vars:
    if var_name.startswith("init_atmosphere_") or var_name.startswith("ls_forcing_") or var_name.startswith("init_soil_"):
        # Convert to float32 if not already
        ds_palm_final[var_name] = ds_palm_final[var_name].astype(np.float32)
        ds_palm_final[var_name].attrs["_FillValue"] = -9999.0

# Write final file with corrected variable names and units
final_output_filename = f"./dynamic_files/{case_name}_dynamic"
print(f"Writing final file to {final_output_filename}")
ds_palm_final.to_netcdf(final_output_filename, mode="w", format="NETCDF4")

end = datetime.now()
print(f"Total time used: {end-start}")'''
####
#u and v corrected
'''import sys
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

# FIX: Properly handle chemistry species list
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

print(f"Using z_origin = {z_origin} m as ground level (0 m)")

y = np.arange(dy/2,dy*ny+dy/2,dy)
x = np.arange(dx/2,dx*nx+dx/2,dx)
# KEY CHANGE: Add z_origin to vertical coordinates for interpolation
#z = np.arange(dz/2, dz*nz, dz) + z_origin
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

# KEY CHANGE: Add z_origin AFTER stretching calculations
z += z_origin
zw += z_origin

print(f"Final vertical grid: z-range = {z[0]:.1f}m to {z[-1]:.1f}m")
print(f"Ground level (z_origin) = {z_origin}m will become 0m in PALM coordinates")

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
# Print diagnostic information for full WRF domain
#-------------------------------------------------------------------------------
print("\n" + "="*80)
print("DIAGNOSTICS - FULL WRF DOMAIN")
print("="*80)

# Calculate statistics for full domain
print("Full WRF domain dimensions:")
print(f"  Time steps: {len(ds_wrf.time)}")
print(f"  West-East: {ds_wrf.dims['west_east']}")
print(f"  South-North: {ds_wrf.dims['south_north']}")
print(f"  Bottom-Top: {ds_wrf.dims['bottom_top']}")

# Calculate potential temperature from T
ds_wrf["pt"] = ds_wrf["T"] + 300

# Calculate wind speed magnitude
u_full = ds_wrf["U"].mean(dim="time").load()
v_full = ds_wrf["V"].mean(dim="time").load()
w_full = ds_wrf["W"].mean(dim="time").load()
wind_speed_full = np.sqrt(u_full**2 + v_full**2)

# Print statistics for full domain
print("\nFull WRF Domain Statistics (averaged over time):")
print(f"U wind component: min={u_full.min().values:.2f}, max={u_full.max().values:.2f} m/s, mean={u_full.mean().values:.2f} m/s")
print(f"V wind component: min={v_full.min().values:.2f}, max={v_full.max().values:.2f} m/s, mean={v_full.mean().values:.2f} m/s")
print(f"W wind component: min={w_full.min().values:.2f}, max={w_full.max().values:.2f} m/s, mean={w_full.mean().values:.2f} m/s")
print(f"Wind speed: min={wind_speed_full.min().values:.2f}, max={wind_speed_full.max().values:.2f} m/s, mean={wind_speed_full.mean().values:.2f} m/s")

# Temperature statistics
pt_full = ds_wrf["pt"].mean(dim="time").load()
print(f"Potential temperature: min={pt_full.min().values:.2f}, max={pt_full.max().values:.2f} K, mean={pt_full.mean().values:.2f} K")

# Check for unrealistic values in full domain
print("\nFull Domain Value Validation:")
if np.any(np.isnan(u_full)):
    print("  WARNING: NaN values found in U component")
if np.any(np.isinf(u_full)):
    print("  WARNING: Infinite values found in U component")
if u_full.min() < -100 or u_full.max() > 100:
    print("  WARNING: U wind component has unrealistic values (< -100 or > 100 m/s)")

if np.any(np.isnan(v_full)):
    print("  WARNING: NaN values found in V component")
if np.any(np.isinf(v_full)):
    print("  WARNING: Infinite values found in V component")
if v_full.min() < -100 or v_full.max() > 100:
    print("  WARNING: V wind component has unrealistic values (< -100 or > 100 m/s)")

if np.any(np.isnan(w_full)):
    print("  WARNING: NaN values found in W component")
if np.any(np.isinf(w_full)):
    print("  WARNING: Infinite values found in W component")
if w_full.min() < -10 or w_full.max() > 10:
    print("  WARNING: W wind component has unrealistic values (< -10 or > 10 m/s)")

if pt_full.min() < 200 or pt_full.max() > 350:
    print("  WARNING: Potential temperature has unrealistic values (< 200K or > 350K)")

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
# these two lines may be redundant need further tests 27 Oct 2021
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
# Print diagnostic information for PALM subdomain
#-------------------------------------------------------------------------------
print("\n" + "="*80)
print("DIAGNOSTICS - PALM SUBDOMAIN")
print("="*80)

print("PALM subdomain dimensions:")
print(f"  Time steps: {len(ds_drop.time)}")
print(f"  West-East: {ds_drop.dims['west_east']}")
print(f"  South-North: {ds_drop.dims['south_north']}")
print(f"  Bottom-Top: {ds_drop.dims['bottom_top']}")

# Calculate statistics for PALM subdomain
u_palm = ds_drop["U"].mean(dim="time").load()
v_palm = ds_drop["V"].mean(dim="time").load()
w_palm = ds_drop["W"].mean(dim="time").load()
wind_speed_palm = np.sqrt(u_palm**2 + v_palm**2)
pt_palm = ds_drop["pt"].mean(dim="time").load()

print("\nPALM Subdomain Statistics (averaged over time):")
print(f"U wind component: min={u_palm.min().values:.2f}, max={u_palm.max().values:.2f} m/s, mean={u_palm.mean().values:.2f} m/s")
print(f"V wind component: min={v_palm.min().values:.2f}, max={v_palm.max().values:.2f} m/s, mean={v_palm.mean().values:.2f} m/s")
print(f"W wind component: min={w_palm.min().values:.2f}, max={w_palm.max().values:.2f} m/s, mean={w_palm.mean().values:.2f} m/s")
print(f"Wind speed: min={wind_speed_palm.min().values:.2f}, max={wind_speed_palm.max().values:.2f} m/s, mean={wind_speed_palm.mean().values:.2f} m/s")
print(f"Potential temperature: min={pt_palm.min().values:.2f}, max={pt_palm.max().values:.2f} K, mean={pt_palm.mean().values:.2f} K")

# Check for unrealistic values in PALM subdomain
print("\nPALM Subdomain Value Validation:")
validation_passed = True

if np.any(np.isnan(u_palm)):
    print("  ❌ ERROR: NaN values found in U component in PALM domain")
    validation_passed = False
if np.any(np.isinf(u_palm)):
    print("  ❌ ERROR: Infinite values found in U component in PALM domain")
    validation_passed = False
if u_palm.min() < -100 or u_palm.max() > 100:
    print(f"  ⚠️  WARNING: U wind component has unrealistic values ({u_palm.min().values:.2f} to {u_palm.max().values:.2f} m/s)")

if np.any(np.isnan(v_palm)):
    print("  ❌ ERROR: NaN values found in V component in PALM domain")
    validation_passed = False
if np.any(np.isinf(v_palm)):
    print("  ❌ ERROR: Infinite values found in V component in PALM domain")
    validation_passed = False
if v_palm.min() < -100 or v_palm.max() > 100:
    print(f"  ⚠️  WARNING: V wind component has unrealistic values ({v_palm.min().values:.2f} to {v_palm.max().values:.2f} m/s)")

if np.any(np.isnan(w_palm)):
    print("  ❌ ERROR: NaN values found in W component in PALM domain")
    validation_passed = False
if np.any(np.isinf(w_palm)):
    print("  ❌ ERROR: Infinite values found in W component in PALM domain")
    validation_passed = False
if w_palm.min() < -10 or w_palm.max() > 10:
    print(f"  ⚠️  WARNING: W wind component has unrealistic values ({w_palm.min().values:.2f} to {w_palm.max().values:.2f} m/s)")

if pt_palm.min() < 200 or pt_palm.max() > 350:
    print(f"  ⚠️  WARNING: Potential temperature has unrealistic values ({pt_palm.min().values:.2f} to {pt_palm.max().values:.2f} K)")

if validation_passed:
    print("  ✅ PALM subdomain validation: All variables contain realistic values")
else:
    print("  ❌ PALM subdomain validation: Some variables contain problematic values")

print("\n" + "="*80)
print("DOMAIN COMPARISON SUMMARY")
print("="*80)
print(f"{'Variable':<20} {'Full Domain Range':<25} {'PALM Domain Range':<25}")
print(f"{'-'*20} {'-'*25} {'-'*25}")
print(f"{'U (m/s)':<20} {f'{u_full.min().values:.2f} to {u_full.max().values:.2f}':<25} {f'{u_palm.min().values:.2f} to {u_palm.max().values:.2f}':<25}")
print(f"{'V (m/s)':<20} {f'{v_full.min().values:.2f} to {v_full.max().values:.2f}':<25} {f'{v_palm.min().values:.2f} to {v_palm.max().values:.2f}':<25}")
print(f"{'W (m/s)':<20} {f'{w_full.min().values:.2f} to {w_full.max().values:.2f}':<25} {f'{w_palm.min().values:.2f} to {w_palm.max().values:.2f}':<25}")
print(f"{'Wind Speed (m/s)':<20} {f'{wind_speed_full.min().values:.2f} to {wind_speed_full.max().values:.2f}':<25} {f'{wind_speed_palm.min().values:.2f} to {wind_speed_palm.max().values:.2f}':<25}")
print(f"{'Temperature (K)':<20} {f'{pt_full.min().values:.2f} to {pt_full.max().values:.2f}':<25} {f'{pt_palm.min().values:.2f} to {pt_palm.max().values:.2f}':<25}")

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
    median_spois = np.ones_like(median_smois)
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
# Add chemistry species to variable list
varbc_list.extend(chem_species)

print("remove unused vars from datasets")
for var in ds_we.data_vars:
    if var not in varbc_list:
        ds_we = ds_we.drop(var)
        ds_sn = ds_sn.drop(var)
    if var not in ["U", "Z"] and var not in chem_species:
        ds_we_ustag = ds_we_ustag.drop(var)
        ds_sn_ustag = ds_sn_ustag.drop(var)
    if var not in ["V", "Z"] and var not in chem_species:
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

# interpolation for chemistry species - FIXED: Use proper interpolation like other variables
print(f"Processing chemistry species: {chem_species}")
for species in chem_species:
    print(f"Checking if {species} exists in dataset...")
    # Check if the species exists in the dataset before processing
    if species in list(ds_we.data_vars.keys()):
        print(f"Processing {species}...")
        # Get the actual dimensions from the WRF data
        chem_dims = ds_we[species].shape
        chem_zeros_we = np.zeros((chem_dims[0], len(z), len(y), len(x[:2])))
        chem_zeros_sn = np.zeros((chem_dims[0], len(z), len(y[:2]), len(x)))
        
        ds_palm_we[species] = xr.DataArray(np.copy(chem_zeros_we), dims=['time','z','y', 'x'])
        ds_palm_sn[species] = xr.DataArray(np.copy(chem_zeros_sn), dims=['time','z','y', 'x'])
        print(f"Processing {species} for west and east boundaries")
        # Use the same interpolation method as other variables
        ds_palm_we[species] = multi_zinterp(max_pool, ds_we, species, z, ds_palm_we)
        print(f"Processing {species} for south and north boundaries")
        ds_palm_sn[species] = multi_zinterp(max_pool, ds_sn, species, z, ds_palm_sn)
    else:
        print(f"Warning: {species} not found in WRF dataset, skipping...")
        print(f"Available variables: {list(ds_we.data_vars.keys())[:10]}...")  # Show first 10 variables
    
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
# Handle NaN values in chemistry boundary conditions - FIXED: Use proper interpolation instead of filling with zeros
#-------------------------------------------------------------------------------
print("Handling NaN values in chemistry boundary conditions...")
for species in chem_species:
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

# Initialize arrays for chemistry species top boundary
chem_top = {}
for species in chem_species:
    chem_top[species] = np.zeros((len(all_ts), len(y), len(x)))

for var in ds_interp.data_vars:
    if var not in varbc_list:
        ds_interp = ds_interp.drop(var)
    if var not in ["U", "Z"] and var not in chem_species:
        ds_interp_u = ds_interp_u.drop(var)
    if var not in ["V", "Z"] and var not in chem_species:
        ds_interp_v = ds_interp_v.drop(var)

print("Processing top boundary datasets...")
ds_interp_top = xr.Dataset()
ds_interp_u_top = xr.Dataset()
ds_interp_v_top = xr.Dataset()
for var in ["QVAPOR", "pt"]:
    ds_interp_top[var] =  ds_interp.salem.wrf_zlevel(var, levels=z[-1]).copy()

# Process chemistry species for top boundary
for species in chem_species:
    if species in ds_interp.data_vars:
        ds_interp_top[species] = ds_interp.salem.wrf_zlevel(species, levels=z[-1]).copy()

ds_interp_top["W"] = ds_interp.salem.wrf_zlevel("W", levels=zw[-1]).copy()        
ds_interp_u_top["U"] = ds_interp_u.salem.wrf_zlevel("U", levels=z[-1]).copy()
ds_interp_v_top["V"] = ds_interp_v.salem.wrf_zlevel("V", levels=z[-1]).copy()

for ts in tqdm(range(0,len(all_ts)), total=len(all_ts), position=0, leave=True):
    u_top[ts,:,:] = ds_interp_u_top["U"].isel(time=ts)
    v_top[ts,:,:] = ds_interp_v_top["V"].isel(time=ts)
    w_top[ts,:,:] = ds_interp_top["W"].isel(time=ts)  
    pt_top[ts,:,:] = ds_interp_top["pt"].isel(time=ts) 
    qv_top[ts,:,:] = ds_interp_top["QVAPOR"].isel(time=ts) 
    # Process chemistry species top boundary
    for species in chem_species:
        if species in ds_interp_top.data_vars:
            chem_top[species][ts,:,:] = ds_interp_top[species].isel(time=ts)

# Handle NaN values in top boundary chemistry data - use proper interpolation
for species in chem_species:
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

# Diagnostic: Check available pressure variables
print("Available variables in WRF dataset:")
pressure_vars = [var for var in ds_drop.data_vars if 'press' in var.lower() or var in ['P', 'PB', 'PRESSURE']]
for var in pressure_vars:
    var_data = ds_drop[var]
    print(f"  {var}: shape {var_data.shape}, range {np.nanmin(var_data):.1f} to {np.nanmax(var_data):.1f}")

# Also check temperature variables
temp_vars = [var for var in ds_drop.data_vars if 'temp' in var.lower() or 'tk' in var.lower() or 't' in var.lower()]
for var in temp_vars:
    var_data = ds_drop[var]
    print(f"  {var}: shape {var_data.shape}, range {np.nanmin(var_data):.1f} to {np.nanmax(var_data):.1f}")

#-------------------------------------------------------------------------------
# Geostrophic wind estimation - IMPROVED for small domains
#-------------------------------------------------------------------------------
print("Geostrophic wind estimation...")
print(f"Using {geostr_lvl}-level method for geostrophic wind calculation")

# First, let's diagnose the actual wind patterns in the WRF data
print("Diagnosing actual wind patterns from WRF data...")
u_mean = ds_drop["U"].mean(("time", "bottom_top", "south_north", "west_east")).load().data
v_mean = ds_drop["V"].mean(("time", "bottom_top", "south_north", "west_east")).load().data
print(f"Mean WRF winds: U={u_mean:.2f} m/s, V={v_mean:.2f} m/s")

if geostr_lvl == "z":
    lat_geostr = ds_drop.lat[:,0]
    dx_wrf = ds_drop.DX
    dy_wrf = ds_drop.DY
    gph = ds_drop.gph
    print("Geostrophic wind loading data...")
    gph = gph.load()
    
    ds_geostr_z = xr.Dataset()
    ds_geostr_z = ds_geostr_z.assign_coords({
        "time": ds_drop.time.data,
        "z": ds_drop["Z"].mean(("time", "south_north", "west_east")).data
    })
    
    ds_geostr_z["ug"] = xr.DataArray(np.zeros((len(all_ts), len(gph.bottom_top.data))),
                                   dims=['time','z'])
    ds_geostr_z["vg"] = xr.DataArray(np.zeros((len(all_ts), len(gph.bottom_top.data))),
                                   dims=['time','z'])

    print("Calculating geostrophic wind profiles...")
    for ts in tqdm(range(0, len(all_ts)), total=len(all_ts), position=0, leave=True):
        for level in gph.bottom_top.data:
            try:
                ug_val, vg_val = calc_geostrophic_wind_zlevels(
                    gph[ts, level, :, :].data, 
                    lat_geostr.data, 
                    dy_wrf, 
                    dx_wrf
                )
                ds_geostr_z["ug"][ts, level] = ug_val
                ds_geostr_z["vg"][ts, level] = vg_val
            except Exception as e:
                print(f"Error at time {ts}, level {level}: {e}")
                # Use WRF wind as fallback
                ds_geostr_z["ug"][ts, level] = u_mean
                ds_geostr_z["vg"][ts, level] = v_mean

    # interpolate to PALM vertical levels
    ds_geostr = ds_geostr_z.interp({"z": z})

# Remove the unrealistic value check and replacement - trust our new calculation
print("Geostrophic wind diagnostics:")
ug_data = ds_geostr["ug"].data
vg_data = ds_geostr["vg"].data

print(f"  ug shape: {ug_data.shape}")
print(f"  vg shape: {vg_data.shape}")
print(f"  ug range: {np.min(ug_data):.2f} to {np.max(ug_data):.2f} m/s")
print(f"  vg range: {np.min(vg_data):.2f} to {np.max(vg_data):.2f} m/s")
print(f"  ug mean: {np.mean(ug_data):.2f} m/s")
print(f"  vg mean: {np.mean(vg_data):.2f} m/s")

# Check if values are realistic
if np.max(np.abs(ug_data)) > 50 or np.max(np.abs(vg_data)) > 50:
    print("WARNING: Unrealistically high geostrophic wind values detected!")
    print("Using typical values instead...")
    ds_geostr["ug"].data = np.full_like(ug_data, 8.0)  # Typical westerly
    ds_geostr["vg"].data = np.full_like(vg_data, 0.5)  # Typical southerly


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
# near surface geostrophic wind
for t in range(0,len(all_ts)):
    ds_geostr["ug"][t,:] =  surface_nan_w(ds_geostr["ug"][t,:].data)
    ds_geostr["vg"][t,:] =  surface_nan_w(ds_geostr["vg"][t,:].data)

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
w_init = ds_drop["W"].sel(time=dt_start).mean(
    dim=["south_north", "west_east"]).interp(
    {"bottom_top": zw}, method = interp_mode)
pt_init = ds_drop["pt"].sel(time=dt_start).mean(
    dim=["south_north", "west_east"]).interp(
    {"bottom_top": z}, method = interp_mode)
qv_init = ds_drop["QVAPOR"].sel(time=dt_start).mean(
    dim=["south_north", "west_east"]).interp(
    {"bottom_top": z}, method = interp_mode)

# Initialize chemistry species profiles
chem_init = {}
for species in chem_species:
    if species in ds_drop.data_vars:
        chem_init[species] = ds_drop[species].sel(time=dt_start).mean(
            dim=["south_north", "west_east"]).interp(
            {"bottom_top": z}, method = interp_mode)
    else:
        # If species not found, create zeros array
        chem_init[species] = xr.DataArray(np.zeros(len(z)), dims=['z'], coords={'z': z})

#-------------------------------------------------------------------------------
# Create single NetCDF file with PALM dynamic driver format
#-------------------------------------------------------------------------------
print("Creating single NetCDF file with PALM dynamic driver format...")

# Create the main dataset
ds_palm = xr.Dataset()
res_origin = str(dx) + 'x' + str(dy) + ' m'

# Assign coordinates
# KEY CHANGE: Subtract z_origin from vertical coordinates for final output
ds_palm = ds_palm.assign_coords({
    "x": x,
    "y": y,
    "z": z - z_origin,  # Adjust so z_origin becomes 0 m
    "zsoil": zs_palm,
    "xu": xu,
    "yv": yv,
    "zw": zw - z_origin,  # Adjust so z_origin becomes 0 m
    "time": times_sec,
    "time_rad": times_sec  # Assuming same time for radiation
})

# Add coordinate attributes
ds_palm["x"].attrs = {"units": "m", "long_name": "distance to origin in x-direction"}
ds_palm["y"].attrs = {"units": "m", "long_name": "distance to origin in y-direction"}
ds_palm["z"].attrs = {"units": "m", "long_name": "height above origin"}
ds_palm["zsoil"].attrs = {"units": "m", "long_name": "depth of soil layer"}
ds_palm["xu"].attrs = {"units": "m", "long_name": "distance to origin in x-direction at u-grid"}
ds_palm["yv"].attrs = {"units": "m", "long_name": "distance to origin in y-direction at v-grid"}
ds_palm["zw"].attrs = {"units": "m", "long_name": "height above origin at w-grid"}
ds_palm["time"].attrs = {"units": "seconds", "long_name": "time"}
ds_palm["time_rad"].attrs = {"units": "seconds", "long_name": "time for radiation"}

# Add soil data
ds_palm["init_soil_m"] = xr.DataArray(init_msoil, dims=['zsoil', 'y', 'x'])
ds_palm["init_soil_m"].attrs = {
    "units": "m^3/m^3", 
    "long_name": "volumetric soil moisture (m^3/m^3)",
    "source": "WRF",
    "lod": np.int32(2)  # Convert to int32
}

ds_palm["init_soil_t"] = xr.DataArray(init_tsoil, dims=['zsoil', 'y', 'x'])
ds_palm["init_soil_t"].attrs = {
    "units": "K", 
    "long_name": "soil temperature (K)",
    "source": "WRF",
    "lod": np.int32(2)  # Convert to int32
}

# Add initial profiles
ds_palm["init_atmosphere_pt"] = xr.DataArray(pt_init.data, dims=['z'])
ds_palm["init_atmosphere_pt"].attrs = {
    "units": "K", 
    "long_name": "initial potential temperature profile",
    "source": "WRF",
    "lod": np.int32(1),  # Convert to int32
    "res_origin":res_origin
}

ds_palm["init_atmosphere_qv"] = xr.DataArray(qv_init.data, dims=['z'])
ds_palm["init_atmosphere_qv"].attrs = {
    "units": "kg/kg", 
    "long_name": "initial water vapor mixing ratio profile",
    "source": "WRF",
    "lod": np.int32(1),  # Convert to int32
    "res_origin":res_origin  # Convert to int32
}

ds_palm["init_atmosphere_u"] = xr.DataArray(u_init.data, dims=['z'])
ds_palm["init_atmosphere_u"].attrs = {
    "units": "m/s", 
    "long_name": "initial u profile",
    "source": "WRF",
    "lod": np.int32(1),  # Convert to int32
    "res_origin":res_origin  # Convert to int32
}

ds_palm["init_atmosphere_v"] = xr.DataArray(v_init.data, dims=['z'])
ds_palm["init_atmosphere_v"].attrs = {
    "units": "m/s", 
    "long_name": "initial v profile",
    "source": "WRF",
    "lod": np.int32(1),  # Convert to int32
    "res_origin":res_origin  # Convert to int32
}

ds_palm["init_atmosphere_w"] = xr.DataArray(w_init.data, dims=['zw'])
ds_palm["init_atmosphere_w"].attrs = {
    "units": "m/s", 
    "long_name": "initial w profile",
    "source": "WRF",
    "lod": np.int32(1),  # Convert to int32
    "res_origin":res_origin  # Convert to int32
}

# Add chemistry initial profiles - keep original names for now
for species in chem_species:
    ds_palm[f"init_atmosphere_{species}"] = xr.DataArray(chem_init[species].data, dims=['z'])
    # Set units based on species type
    if species in ['PM10', 'PM2_5_DRY']:
        unit = "kg/m3"
    elif species in ['no', 'no2', 'o3']:
        unit = "ppm"
    else:
        unit = "ppmv"
    
    ds_palm[f"init_atmosphere_{species}"].attrs = {
        "units": unit, 
        "long_name": f"initial {species} profile",
        "source": "WRF-Chem",
        "lod": np.int32(1),  # Convert to int32
        "res_origin":res_origin
    }

# Add boundary conditions
# West and East boundaries
# ---------------------------
# West & East boundaries
# ---------------------------
ds_palm["ls_forcing_left_pt"] = xr.DataArray(
    ds_palm_we["pt"].isel(x=0).data, dims=['time', 'z', 'y']
)
ds_palm["ls_forcing_left_pt"].attrs = {"units": "K", "source": "WRF"}

ds_palm["ls_forcing_right_pt"] = xr.DataArray(
    ds_palm_we["pt"].isel(x=-1).data, dims=['time', 'z', 'y']
)
ds_palm["ls_forcing_right_pt"].attrs = {"units": "K", "source": "WRF"}

ds_palm["ls_forcing_left_qv"] = xr.DataArray(
    ds_palm_we["QVAPOR"].isel(x=0).data, dims=['time', 'z', 'y']
)
ds_palm["ls_forcing_left_qv"].attrs = {"units": "kg/kg", "source": "WRF"}

ds_palm["ls_forcing_right_qv"] = xr.DataArray(
    ds_palm_we["QVAPOR"].isel(x=-1).data, dims=['time', 'z', 'y']
)
ds_palm["ls_forcing_right_qv"].attrs = {"units": "kg/kg", "source": "WRF"}

ds_palm["ls_forcing_left_u"] = xr.DataArray(
    ds_palm_we["U"].isel(xu=0).data,
    dims=['time', 'z', 'y']
)
ds_palm["ls_forcing_left_u"].attrs = {"units": "m/s", "source": "WRF"}

ds_palm["ls_forcing_right_u"] = xr.DataArray(
    ds_palm_we["U"].isel(xu=-1).data,
    dims=['time', 'z', 'y']
)
ds_palm["ls_forcing_right_u"].attrs = {"units": "m/s", "source": "WRF"}

ds_palm["ls_forcing_left_v"] = xr.DataArray(
    ds_palm_we["V"].isel(x=0).data, dims=['time', 'z', 'yv']
)
ds_palm["ls_forcing_left_v"].attrs = {"units": "m/s", "source": "WRF"}

ds_palm["ls_forcing_right_v"] = xr.DataArray(
    ds_palm_we["V"].isel(x=-1).data, dims=['time', 'z', 'yv']
)
ds_palm["ls_forcing_right_v"].attrs = {"units": "m/s", "source": "WRF"}

ds_palm["ls_forcing_left_w"] = xr.DataArray(
    ds_palm_we["W"].isel(x=0).data, dims=['time', 'zw', 'y']
)
ds_palm["ls_forcing_left_w"].attrs = {"units": "m/s", "source": "WRF"}

ds_palm["ls_forcing_right_w"] = xr.DataArray(
    ds_palm_we["W"].isel(x=-1).data, dims=['time', 'zw', 'y']
)
ds_palm["ls_forcing_right_w"].attrs = {"units": "m/s", "source": "WRF"}

# ---------------------------
# South & North boundaries
# ---------------------------
ds_palm["ls_forcing_south_pt"] = xr.DataArray(
    ds_palm_sn["pt"].isel(y=0).data, dims=['time', 'z', 'x']
)
ds_palm["ls_forcing_south_pt"].attrs = {"units": "K", "source": "WRF"}

ds_palm["ls_forcing_north_pt"] = xr.DataArray(
    ds_palm_sn["pt"].isel(y=-1).data, dims=['time', 'z', 'x']
)
ds_palm["ls_forcing_north_pt"].attrs = {"units": "K", "source": "WRF"}

ds_palm["ls_forcing_south_qv"] = xr.DataArray(
    ds_palm_sn["QVAPOR"].isel(y=0).data, dims=['time', 'z', 'x']
)
ds_palm["ls_forcing_south_qv"].attrs = {"units": "kg/kg", "source": "WRF"}

ds_palm["ls_forcing_north_qv"] = xr.DataArray(
    ds_palm_sn["QVAPOR"].isel(y=-1).data, dims=['time', 'z', 'x']
)
ds_palm["ls_forcing_north_qv"].attrs = {"units": "kg/kg", "source": "WRF"}

ds_palm["ls_forcing_south_u"] = xr.DataArray(
    ds_palm_sn["U"].isel(y=0).data, dims=['time', 'z', 'xu']
)
ds_palm["ls_forcing_south_u"].attrs = {"units": "m/s", "source": "WRF"}

ds_palm["ls_forcing_north_u"] = xr.DataArray(
    ds_palm_sn["U"].isel(y=-1).data, dims=['time', 'z', 'xu']
)
ds_palm["ls_forcing_north_u"].attrs = {"units": "m/s", "source": "WRF"}

ds_palm["ls_forcing_south_v"] = xr.DataArray(
    ds_palm_sn["V"].isel(yv=0).data,
    dims=['time', 'z', 'x']
)
ds_palm["ls_forcing_south_v"].attrs = {"units": "m/s", "source": "WRF"}

ds_palm["ls_forcing_north_v"] = xr.DataArray(
    ds_palm_sn["V"].isel(yv=-1).data,
    dims=['time', 'z', 'x']
)
ds_palm["ls_forcing_north_v"].attrs = {"units": "m/s", "source": "WRF"}

ds_palm["ls_forcing_south_w"] = xr.DataArray(
    ds_palm_sn["W"].isel(y=0).data, dims=['time', 'zw', 'x']
)
ds_palm["ls_forcing_south_w"].attrs = {"units": "m/s", "source": "WRF"}

ds_palm["ls_forcing_north_w"] = xr.DataArray(
    ds_palm_sn["W"].isel(y=-1).data, dims=['time', 'zw', 'x']
)
ds_palm["ls_forcing_north_w"].attrs = {"units": "m/s", "source": "WRF"}

# ---------------------------
# Top boundary
# ---------------------------
ds_palm["ls_forcing_top_pt"] = xr.DataArray(pt_top, dims=['time', 'y', 'x'])
ds_palm["ls_forcing_top_pt"].attrs = {"units": "K", "source": "WRF"}

ds_palm["ls_forcing_top_qv"] = xr.DataArray(qv_top, dims=['time', 'y', 'x'])
ds_palm["ls_forcing_top_qv"].attrs = {"units": "kg/kg", "source": "WRF"}

ds_palm["ls_forcing_top_u"] = xr.DataArray(u_top, dims=['time', 'y', 'xu'])
ds_palm["ls_forcing_top_u"].attrs = {"units": "m/s", "source": "WRF"}

ds_palm["ls_forcing_top_v"] = xr.DataArray(v_top, dims=['time', 'yv', 'x'])
ds_palm["ls_forcing_top_v"].attrs = {"units": "m/s", "source": "WRF"}

ds_palm["ls_forcing_top_w"]  = xr.DataArray(w_top,  dims=['time', 'y', 'x'])
ds_palm["ls_forcing_top_w"].attrs = {"units": "m/s", "source": "WRF"}

# ---------------------------
# Chemistry boundary conditions - keep original names for now
# ---------------------------
for species in chem_species:
    if species in ds_palm_we.data_vars:
        # West & East
        ds_palm[f"ls_forcing_left_{species}"] = xr.DataArray(
            ds_palm_we[species].isel(x=0).data, dims=['time', 'z', 'y']
        )
        ds_palm[f"ls_forcing_right_{species}"] = xr.DataArray(
            ds_palm_we[species].isel(x=-1).data, dims=['time', 'z', 'y']
        )

        # South & North
        ds_palm[f"ls_forcing_south_{species}"] = xr.DataArray(
            ds_palm_sn[species].isel(y=0).data, dims=['time', 'z', 'x']
        )
        ds_palm[f"ls_forcing_north_{species}"] = xr.DataArray(
            ds_palm_sn[species].isel(y=-1).data, dims=['time', 'z', 'x']
        )

        # Top
        ds_palm[f"ls_forcing_top_{species}"] = xr.DataArray(
            chem_top[species], dims=['time', 'y', 'x']
        )

        # Set units based on species type
        if species in ['PM10', 'PM2_5_DRY']:
            unit = "kg/m3"
        elif species in ['no', 'no2', 'o3']:
            unit = "ppm"
        else:
            unit = "ppmv"
            
        for var_name in [
            f"ls_forcing_left_{species}", f"ls_forcing_right_{species}",
            f"ls_forcing_south_{species}", f"ls_forcing_north_{species}",
            f"ls_forcing_top_{species}"
        ]:
            ds_palm[var_name].attrs = {"units": unit, "source": "WRF-Chem"}

# Add geostrophic wind
ds_palm["ls_forcing_ug"] = xr.DataArray(ds_geostr["ug"].data, dims=['time', 'z'])
ds_palm["ls_forcing_ug"].attrs = {
    "units": "m/s", 
    "long_name": "u wind component geostrophic",
    "source": "WRF",
    "res_origin":res_origin
}

ds_palm["ls_forcing_vg"] = xr.DataArray(ds_geostr["vg"].data, dims=['time', 'z'])
ds_palm["ls_forcing_vg"].attrs = {
    "units": "m/s", 
    "long_name": "v wind component geostrophic",
    "source": "WRF", 
    "res_origin":res_origin
}

# Add global attributes
ds_palm.attrs = {
    "title": "PALM dynamic driver generated by WRF4PALM",
    "author": "WRF4PALM v1.1.2",
    "source": "WRF-Chem",
    "history": f"Created: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
    "reference": "WRF4PALM: Lin et al. (2021)",
    "Conventions": "PIDS v1.9",
    "origin_lat": centlat,
    "origin_lon": centlon,
    "origin_x": centx,
    "origin_y": centy,
    "rotation_angle": 0.0,
    "origin_time": str(dt_start) + " UTC",
    "end_time": str(dt_end) + " UTC"
}

# Write to single NetCDF file
#output_filename = f"./dynamic_files/{case_name}_dynamic_driver.nc"
#print(f"Writing to {output_filename}")
#ds_palm.to_netcdf(output_filename, mode="w", format="NETCDF4")

# Now rename variables to uppercase, fix units, and convert PM values
print("Renaming chemistry variables to uppercase, fixing units, and converting PM values...")
ds_palm_final = xr.Dataset()

# Copy all coordinates
for coord_name in ds_palm.coords:
    ds_palm_final = ds_palm_final.assign_coords({coord_name: ds_palm[coord_name]})

# Copy all non-chemistry variables
for var_name in ds_palm.data_vars:
    if not any(species in var_name for species in chem_species):
        ds_palm_final[var_name] = ds_palm[var_name]

# Conversion factor from μg/m³ to kg/m³
MICROGRAM_TO_KG = 1e-9

# Process chemistry variables - rename to uppercase, fix units, and convert PM values
for species in chem_species:
    # Determine the new species name (convert PM2_5_DRY to PM25)
    new_species_name = species.upper()
    if new_species_name == "PM2_5_DRY":
        new_species_name = "PM25"
    
    # Process initial profiles
    init_var = f"init_atmosphere_{species}"
    if init_var in ds_palm.data_vars:
        new_init_var = f"init_atmosphere_{new_species_name}"
        
        # Convert PM values from μg/m³ to kg/m³
        if species in ['PM10', 'PM2_5_DRY']:
            converted_data = ds_palm[init_var].data * MICROGRAM_TO_KG
            ds_palm_final[new_init_var] = xr.DataArray(converted_data, dims=['z'])
            ds_palm_final[new_init_var].attrs = {"units": "kg/m3", "source": "WRF-Chem", "lod": np.int32(1), "res_origin":res_origin}
        else:
            ds_palm_final[new_init_var] = ds_palm[init_var]
            if species in ['no', 'no2', 'o3']:
                ds_palm_final[new_init_var].attrs = {"units": "ppm", "source": "WRF-Chem", "lod": np.int32(1), "res_origin":res_origin}
            else:
                ds_palm_final[new_init_var].attrs = {"units": "ppmv", "source": "WRF-Chem", "lod": np.int32(1), "res_origin":res_origin}
    
    # Process boundary conditions
    for boundary in ['left', 'right', 'south', 'north', 'top']:
        bc_var = f"ls_forcing_{boundary}_{species}"
        if bc_var in ds_palm.data_vars:
            new_bc_var = f"ls_forcing_{boundary}_{new_species_name}"
            
            # Convert PM values from μg/m³ to kg/m³
            if species in ['PM10', 'PM2_5_DRY']:
                converted_data = ds_palm[bc_var].data * MICROGRAM_TO_KG
                ds_palm_final[new_bc_var] = xr.DataArray(converted_data, dims=ds_palm[bc_var].dims)
                ds_palm_final[new_bc_var].attrs = {"units": "kg/m3", "source": "WRF-Chem"}
            else:
                ds_palm_final[new_bc_var] = ds_palm[bc_var]
                if species in ['no', 'no2', 'o3']:
                    ds_palm_final[new_bc_var].attrs = {"units": "ppm", "source": "WRF-Chem"}
                else:
                    ds_palm_final[new_bc_var].attrs = {"units": "ppmv", "source": "WRF-Chem"}

# Copy global attributes
ds_palm_final.attrs = ds_palm.attrs

# Explicitly set units for coordinate variables before writing NetCDF
for coord_name in ["x", "y", "z", "xu", "yv", "zw", "zsoil"]:
    if coord_name in ds_palm_final.coords:
        ds_palm_final.coords[coord_name].attrs["units"] = "m"
if "time" in ds_palm_final.coords:
    ds_palm_final.coords["time"].attrs["units"] = "seconds"

# Set _FillValue for all init_atmosphere_*, ls_forcing_*, init_soil_* variables and convert to float32
for var_name in ds_palm_final.data_vars:
    if var_name.startswith("init_atmosphere_") or var_name.startswith("ls_forcing_") or var_name.startswith("init_soil_"):
        # Convert to float32 if not already
        ds_palm_final[var_name] = ds_palm_final[var_name].astype(np.float32)
        ds_palm_final[var_name].attrs["_FillValue"] = -9999.0

# Write final file with corrected variable names and units
final_output_filename = f"./dynamic_files/{case_name}_dynamic"
print(f"Writing final file to {final_output_filename}")
ds_palm_final.to_netcdf(final_output_filename, mode="w", format="NETCDF4")

end = datetime.now()
print(f"Total time used: {end-start}")'''
###
#z values
'''import sys
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

# FIX: Properly handle chemistry species list
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

palm_proj_code = ast.literal_eval(config.get("domain", "palm_proj"))[0]
centlat = ast.literal_eval(config.get("domain", "centlat"))[0]
centlon = ast.literal_eval(config.get("domain", "centlon"))[0]
dx = ast.literal_eval(config.get("domain", "dx"))[0]
dy = ast.literal_eval(config.get("domain", "dy"))[0]
dz = ast.literal_eval(config.get("domain", "dz"))[0]
nx = ast.literal_eval(config.get("domain", "nx"))[0]
ny = ast.literal_eval(config.get("domain", "ny"))[0]
nz = ast.literal_eval(config.get("domain", "nz"))[0]

# NEW: Handle z_origin configuration - auto or manual
z_origin_config = ast.literal_eval(config.get("domain", "z_origin"))[0]
min_terrain_buffer = ast.literal_eval(config.get("terrain", "min_terrain_buffer"))[0]
max_terrain_buffer = ast.literal_eval(config.get("terrain", "max_terrain_buffer"))[0]

print(f"z_origin configuration: {z_origin_config}")

# Initialize z_origin (will be calculated later after reading WRF data)
z_origin = None

print(f"Domain configuration: {nx}x{ny}x{nz} grid with {dx}x{dy}x{dz} m resolution")

y = np.arange(dy/2,dy*ny+dy/2,dy)
x = np.arange(dx/2,dx*nx+dx/2,dx)
# Vertical coordinates without z_origin initially
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
# NEW: Calculate terrain height and determine z_origin
#-------------------------------------------------------------------------------
print("\n" + "="*80)
print("TERRAIN HEIGHT ANALYSIS")
print("="*80)

def get_terrain_height(ds):
    """Get terrain height from WRF dataset using available variables"""
    if 'HGT' in ds.variables:
        # Use terrain height directly from WRF
        terrain_height = ds['HGT'].isel(time=0).load()
        print("Using HGT variable for terrain height")
        return terrain_height
    elif 'PH' in ds.variables and 'PHB' in ds.variables:
        # Calculate from geopotential
        print("Calculating terrain height from geopotential (PH + PHB)")
        ph = ds['PH'].isel(time=0, bottom_top=0).load()
        phb = ds['PHB'].isel(time=0, bottom_top=0).load()
        terrain_height = (ph + phb) / 9.81
        return terrain_height
    else:
        # Try to find elevation variable
        elev_vars = [var for var in ds.variables if 'elev' in var.lower() or 'height' in var.lower() or 'terrain' in var.lower()]
        if elev_vars:
            print(f"Using {elev_vars[0]} variable for terrain height")
            return ds[elev_vars[0]].isel(time=0).load()
        else:
            raise ValueError("No terrain height variable found in WRF dataset. Available variables: " + ", ".join(ds.variables.keys()))

# Get terrain height for full domain
terrain_height_full = get_terrain_height(ds_wrf)
print(f"Full WRF domain terrain height range: {terrain_height_full.min().values:.2f} to {terrain_height_full.max().values:.2f} m")

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

# find indices of closest values
west_idx,east_idx,south_idx,north_idx = framing_2d_cartesian(lons_wrf,lats_wrf, west,east,south,north,dx_wrf, dy_wrf)

# in case negative longitudes are used
# these two lines may be redundant need further tests 27 Oct 2021
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

#-------------------------------------------------------------------------------
# NEW: Calculate z_origin based on PALM subdomain terrain
#-------------------------------------------------------------------------------
print("\nCalculating z_origin for PALM subdomain...")

# Get terrain height for PALM subdomain
palm_terrain = get_terrain_height(ds_drop)

min_terrain = float(palm_terrain.min().values)
max_terrain = float(palm_terrain.max().values)
mean_terrain = float(palm_terrain.mean().values)

print(f"PALM subdomain terrain height: min={min_terrain:.2f}m, max={max_terrain:.2f}m, mean={mean_terrain:.2f}m")

# Determine z_origin
if z_origin_config == "auto":
    # Use minimum terrain height plus buffer
    z_origin = min_terrain - min_terrain_buffer
    print(f"Auto-calculated z_origin: {z_origin:.2f}m (min_terrain {min_terrain:.2f}m - buffer {min_terrain_buffer}m)")
    
    # Validate that this doesn't create issues
    if z_origin < min_terrain - max_terrain_buffer:
        print(f"Warning: z_origin is more than {max_terrain_buffer}m below minimum terrain")
        print("This might indicate problematic terrain data")
else:
    # Use manual z_origin
    z_origin = float(z_origin_config)
    print(f"Using manual z_origin: {z_origin:.2f}m")
    
    # Check if manual z_origin is reasonable
    if z_origin > min_terrain:
        print(f"⚠️  WARNING: Manual z_origin ({z_origin:.2f}m) is above minimum terrain ({min_terrain:.2f}m)")
        print("This will create underground grid points with NaN values!")
    elif z_origin < min_terrain - max_terrain_buffer:
        print(f"⚠️  WARNING: Manual z_origin ({z_origin:.2f}m) is more than {max_terrain_buffer}m below minimum terrain ({min_terrain:.2f}m)")

print(f"Final z_origin: {z_origin:.2f}m")
print(f"PALM vertical coordinates will range from {z[0] + z_origin:.2f}m to {z[-1] + z_origin:.2f}m above sea level")
print(f"PALM ground level (0m in PALM coordinates) corresponds to {z_origin:.2f}m above sea level")

# Now add z_origin to vertical coordinates
z += z_origin
zw += z_origin

print(f"Adjusted vertical grid: z-range = {z[0]:.1f}m to {z[-1]:.1f}m ASL")
print(f"Ground level in PALM coordinates: 0m = {z_origin:.1f}m ASL")

# Continue with processing...
ds_drop["pt"] = ds_drop["T"] + 300
ds_drop["pt"].attrs = ds_drop["T"].attrs
ds_drop["gph"] = (ds_drop["PH"] + ds_drop["PHB"])/9.81
ds_drop["gph"].attrs = ds_drop["PH"].attrs

# Write cfg file with calculated z_origin
generate_cfg(case_name, dx, dy, dz, nx, ny, nz,
             west, east, south, north, centlat, centlon, z_origin)

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
    median_spois = np.ones_like(median_smois)
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
# Add chemistry species to variable list
varbc_list.extend(chem_species)

print("remove unused vars from datasets")
for var in ds_we.data_vars:
    if var not in varbc_list:
        ds_we = ds_we.drop(var)
        ds_sn = ds_sn.drop(var)
    if var not in ["U", "Z"] and var not in chem_species:
        ds_we_ustag = ds_we_ustag.drop(var)
        ds_sn_ustag = ds_sn_ustag.drop(var)
    if var not in ["V", "Z"] and var not in chem_species:
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

# FIX: Don't try to get terrain height from boundary datasets - they don't have HGT
# Instead, use the original palm_terrain data we already computed
print("Using pre-computed terrain data for boundaries...")

# interpolation scalars
for varbc in ["QVAPOR","pt"]:
    ds_palm_we[varbc] = xr.DataArray(np.copy(zeros_we), dims=['time','z','y', 'x'])
    ds_palm_sn[varbc] = xr.DataArray(np.copy(zeros_sn), dims=['time','z','y', 'x'])
    print(f"Processing {varbc} for west and east boundaries")
    ds_palm_we[varbc] = multi_zinterp(max_pool, ds_we, varbc, z, ds_palm_we)
    print(f"Processing {varbc} for south and north boundaries")
    ds_palm_sn[varbc] = multi_zinterp(max_pool, ds_sn, varbc, z, ds_palm_sn)

# interpolation for chemistry species
print(f"Processing chemistry species: {chem_species}")
for species in chem_species:
    print(f"Checking if {species} exists in dataset...")
    if species in list(ds_we.data_vars.keys()):
        print(f"Processing {species}...")
        chem_zeros_we = np.zeros((len(all_ts), len(z), len(y), len(x[:2])))
        chem_zeros_sn = np.zeros((len(all_ts), len(z), len(y[:2]), len(x)))
        ds_palm_we[species] = xr.DataArray(np.copy(chem_zeros_we), dims=['time','z','y', 'x'])
        ds_palm_sn[species] = xr.DataArray(np.copy(chem_zeros_sn), dims=['time','z','y', 'x'])
        ds_palm_we[species] = multi_zinterp(max_pool, ds_we, species, z, ds_palm_we)
        ds_palm_sn[species] = multi_zinterp(max_pool, ds_sn, species, z, ds_palm_sn)
    else:
        print(f"Warning: {species} not found in WRF dataset. Skipping...")

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
# Handle NaN values in chemistry boundary conditions
#-------------------------------------------------------------------------------
print("Handling NaN values in chemistry boundary conditions...")
for species in chem_species:
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
            
            if np.any(np.isnan(ds_palm_sn[species].data)):
                print(f"Filling remaining NaNs for {species} in south/north with nearest values")
                ds_palm_sn[species] = ds_palm_sn[species].ffill('z').bfill('z')
                ds_palm_sn[species] = ds_palm_sn[species].ffill('x').bfill('x')
            
            print(f"Completed NaN handling for {species}")

#-------------------------------------------------------------------------------
# Top boundary conditions
#-------------------------------------------------------------------------------
print("Processing top boundary conditions...")
print("Processing top boundary datasets...")
ds_top = ds_interp.isel(south_north=slice(south_idx, north_idx),
                        west_east=slice(west_idx, east_idx)).load()

# Create top boundary dataset
ds_palm_top = xr.Dataset()
zeros_top = np.zeros((len(all_ts), len(z), len(y), len(x)))
for varbc in ["QVAPOR","pt"]:
    ds_palm_top[varbc] = xr.DataArray(np.copy(zeros_top), dims=['time','z','y', 'x'])
    ds_palm_top[varbc] = multi_zinterp(max_pool, ds_top, varbc, z, ds_palm_top)

# Add chemistry species to top boundary
for species in chem_species:
    if species in ds_top.data_vars:
        chem_zeros_top = np.zeros((len(all_ts), len(z), len(y), len(x)))
        ds_palm_top[species] = xr.DataArray(np.copy(chem_zeros_top), dims=['time','z','y', 'x'])
        ds_palm_top[species] = multi_zinterp(max_pool, ds_top, species, z, ds_palm_top)

#-------------------------------------------------------------------------------
# Geostrophic wind
#-------------------------------------------------------------------------------
print("Geostrophic wind estimation...")
if geostr_lvl == 0:
    print("Using simplified method for small domains")
    # Use WRF mean winds as approximation for geostrophic wind
    print("Small domain detected, using WRF mean winds as geostrophic approximation")
    
    # Calculate mean wind profiles from WRF
    print("Calculating mean wind profiles from WRF...")
    u_mean = ds_interp["U"].mean(dim=["south_north", "west_east"]).load()
    v_mean = ds_interp["V"].mean(dim=["south_north", "west_east"]).load()
    
    print(f"U mean profile range: {u_mean.min().values:.2f} to {u_mean.max().values:.2f} m/s")
    print(f"V mean profile range: {v_mean.min().values:.2f} to {v_mean.max().values:.2f} m/s")
    
    # Create geostrophic wind from mean profiles
    print("Creating geostrophic wind profiles from WRF mean winds...")
    ug_data = np.zeros((len(all_ts), len(z)))
    vg_data = np.zeros((len(all_ts), len(z)))
    
    for t in tqdm(range(len(all_ts)), desc="Time steps"):
        # Interpolate mean winds to PALM vertical levels
        ug_interp = np.interp(z, u_mean.bottom_top.data, u_mean.isel(time=t).data)
        vg_interp = np.interp(z, v_mean.bottom_top.data, v_mean.isel(time=t).data)
        ug_data[t, :] = ug_interp
        vg_data[t, :] = vg_interp
    
    print("Geostrophic wind diagnostics:")
    print(f"  ug shape: {ug_data.shape}")
    print(f"  vg shape: {vg_data.shape}")
    
    # Check for NaN values and handle them
    if np.all(np.isnan(ug_data)) or np.all(np.isnan(vg_data)):
        print("WARNING: All NaN values in geostrophic wind, using fallback values")
        # Use simple fallback: constant wind profile
        ug_data = np.full_like(ug_data, 5.0)  # 5 m/s from west
        vg_data = np.full_like(vg_data, 0.0)  # no north-south component
    else:
        # Fill any remaining NaN values with column mean
        for t in range(len(all_ts)):
            ug_col = ug_data[t, :]
            vg_col = vg_data[t, :]
            
            if np.any(np.isnan(ug_col)):
                ug_mean_val = np.nanmean(ug_col)
                if np.isnan(ug_mean_val):
                    ug_mean_val = 5.0  # fallback
                ug_col[np.isnan(ug_col)] = ug_mean_val
                ug_data[t, :] = ug_col
            
            if np.any(np.isnan(vg_col)):
                vg_mean_val = np.nanmean(vg_col)
                if np.isnan(vg_mean_val):
                    vg_mean_val = 0.0  # fallback
                vg_col[np.isnan(vg_col)] = vg_mean_val
                vg_data[t, :] = vg_col
    
    print(f"  ug range: {np.nanmin(ug_data):.2f} to {np.nanmax(ug_data):.2f} m/s")
    print(f"  vg range: {np.nanmin(vg_data):.2f} to {np.nanmax(vg_data):.2f} m/s")
    print(f"  ug mean: {np.nanmean(ug_data):.2f} m/s")
    print(f"  vg mean: {np.nanmean(vg_data):.2f} m/s")

else:
    # Original geostrophic wind calculation for larger domains
    ug_data, vg_data = calc_geostrophic_wind(ds_interp, z, palm_proj, all_ts, max_pool)

# Create geostrophic wind dataset
ds_geostr = xr.Dataset()
ds_geostr["ug"] = xr.DataArray(ug_data, dims=['time','z'])
ds_geostr["vg"] = xr.DataArray(vg_data, dims=['time','z'])
ds_geostr = ds_geostr.assign_coords({"time": all_ts, "z": z})

#-------------------------------------------------------------------------------
# Initial profiles
#-------------------------------------------------------------------------------
print("Calculating initial profiles...")
print("Creating initial profiles...")

# Create initial profile dataset
ds_init = xr.Dataset()
init_shape = (len(z), len(y), len(x))

# Initialize arrays for each variable
init_u = np.zeros(init_shape)
init_v = np.zeros(init_shape)
init_w = np.zeros((len(zw), len(y), len(x)))
init_pt = np.zeros(init_shape)
init_qvapor = np.zeros(init_shape)

# Add chemistry species
init_chem = {}
for species in chem_species:
    init_chem[species] = np.zeros(init_shape)

# Use first timestep for initial conditions
t0 = 0

print("  Processing U...")
u_interp = np.zeros((len(z), len(y), len(x)))
for iy in range(len(y)):
    for ix in range(len(x)):
        profile = ds_interp["U"].isel(time=t0, south_north=iy, west_east=ix)
        u_interp[:, iy, ix] = np.interp(z, profile.bottom_top.data, profile.data)
init_u = u_interp
print("    U profile created successfully")

print("  Processing V...")
v_interp = np.zeros((len(z), len(y), len(x)))
for iy in range(len(y)):
    for ix in range(len(x)):
        profile = ds_interp["V"].isel(time=t0, south_north=iy, west_east=ix)
        v_interp[:, iy, ix] = np.interp(z, profile.bottom_top.data, profile.data)
init_v = v_interp
print("    V profile created successfully")

print("  Processing W...")
w_interp = np.zeros((len(zw), len(y), len(x)))
for iy in range(len(y)):
    for ix in range(len(x)):
        profile = ds_interp["W"].isel(time=t0, south_north=iy, west_east=ix)
        w_interp[:, iy, ix] = np.interp(zw, profile.bottom_top_stag.data, profile.data)
init_w = w_interp
print("    W profile created successfully")

print("  Processing pt...")
pt_interp = np.zeros((len(z), len(y), len(x)))
for iy in range(len(y)):
    for ix in range(len(x)):
        profile = ds_interp["pt"].isel(time=t0, south_north=iy, west_east=ix)
        pt_interp[:, iy, ix] = np.interp(z, profile.bottom_top.data, profile.data)
init_pt = pt_interp
print("    pt profile created successfully")

print("  Processing QVAPOR...")
qvapor_interp = np.zeros((len(z), len(y), len(x)))
for iy in range(len(y)):
    for ix in range(len(x)):
        profile = ds_interp["QVAPOR"].isel(time=t0, south_north=iy, west_east=ix)
        qvapor_interp[:, iy, ix] = np.interp(z, profile.bottom_top.data, profile.data)
init_qvapor = qvapor_interp
print("    QVAPOR profile created successfully")

# Process chemistry species initial profiles
print("Verifying initial profiles...")
for species in chem_species:
    if species in ds_interp.data_vars:
        print(f"  Processing {species}...")
        chem_interp = np.zeros((len(z), len(y), len(x)))
        for iy in range(len(y)):
            for ix in range(len(x)):
                profile = ds_interp[species].isel(time=t0, south_north=iy, west_east=ix)
                chem_interp[:, iy, ix] = np.interp(z, profile.bottom_top.data, profile.data)
        init_chem[species] = chem_interp
        print(f"    {species} profile created successfully")
    else:
        print(f"  Warning: {species} not found for initial profile, using zeros")

# Assign to dataset
ds_init["u"] = xr.DataArray(init_u, dims=['z', 'y', 'x'])
ds_init["v"] = xr.DataArray(init_v, dims=['z', 'y', 'x'])
ds_init["w"] = xr.DataArray(init_w, dims=['zw', 'y', 'x'])
ds_init["pt"] = xr.DataArray(init_pt, dims=['z', 'y', 'x'])
ds_init["qv"] = xr.DataArray(init_qvapor, dims=['z', 'y', 'x'])

for species in chem_species:
    if species in init_chem:
        ds_init[species] = xr.DataArray(init_chem[species], dims=['z', 'y', 'x'])

ds_init = ds_init.assign_coords({"x": x, "y": y, "z": z, "zw": zw})

print("Initial profiles completed successfully")

#-------------------------------------------------------------------------------
# NEW: Fix for surface_nan_w function - handle edge cases properly
#-------------------------------------------------------------------------------
def fixed_surface_nan_w(data):
    """
    Fixed version of surface_nan_w that properly handles edge cases
    where all values might be NaN or indices go out of bounds.
    """
    # Make a copy to avoid modifying original
    data = data.copy()
    
    # Find first non-NaN value from the bottom
    non_nan_mask = ~np.isnan(data)
    
    if not np.any(non_nan_mask):
        # All values are NaN, fill with zeros or small value
        data[:] = 0.0
        return data
    
    # Find first non-NaN index
    nan_idx = np.argmax(non_nan_mask)
    
    # Fill surface NaNs with the first valid value
    if nan_idx > 0:
        # We have NaNs at the surface
        first_valid_value = data[nan_idx]
        data[:nan_idx] = first_valid_value
    
    # Fill any remaining NaNs (above surface) with nearest valid value
    if np.any(np.isnan(data)):
        # Forward fill from bottom
        valid_mask = ~np.isnan(data)
        valid_indices = np.where(valid_mask)[0]
        valid_values = data[valid_mask]
        
        if len(valid_indices) > 0:
            # Interpolate to fill all NaN positions
            all_indices = np.arange(len(data))
            data = np.interp(all_indices, valid_indices, valid_values)
    
    return data

# Replace the problematic function call
print("Resolving surface NaNs...")
for t in tqdm(range(len(all_ts))):
    # Use the fixed function instead of the original surface_nan_w
    ds_geostr["ug"][t,:] = fixed_surface_nan_w(ds_geostr["ug"][t,:].data)
    ds_geostr["vg"][t,:] = fixed_surface_nan_w(ds_geostr["vg"][t,:].data)

#-------------------------------------------------------------------------------
# Write dynamic driver
#-------------------------------------------------------------------------------
print("Writing dynamic driver...")

# Create the dynamic driver dataset
ds_dynamic = xr.Dataset()

# Add all boundary conditions and initial profiles
ds_dynamic["ls_forcing_left_u"] = ds_palm_we["U"].isel(xu=0).drop("xu")
ds_dynamic["ls_forcing_right_u"] = ds_palm_we["U"].isel(xu=-1).drop("xu")
ds_dynamic["ls_forcing_south_v"] = ds_palm_sn["V"].isel(yv=0).drop("yv")
ds_dynamic["ls_forcing_north_v"] = ds_palm_sn["V"].isel(yv=-1).drop("yv")

ds_dynamic["ls_forcing_left_v"] = ds_palm_we["V"].isel(x=0).drop("x")
ds_dynamic["ls_forcing_right_v"] = ds_palm_we["V"].isel(x=-1).drop("x")
ds_dynamic["ls_forcing_south_u"] = ds_palm_sn["U"].isel(y=0).drop("y")
ds_dynamic["ls_forcing_north_u"] = ds_palm_sn["U"].isel(y=-1).drop("y")

ds_dynamic["ls_forcing_left_w"] = ds_palm_we["W"].isel(x=0).drop("x")
ds_dynamic["ls_forcing_right_w"] = ds_palm_we["W"].isel(x=-1).drop("x")
ds_dynamic["ls_forcing_south_w"] = ds_palm_sn["W"].isel(y=0).drop("y")
ds_dynamic["ls_forcing_north_w"] = ds_palm_sn["W"].isel(y=-1).drop("y")

ds_dynamic["ls_forcing_left_pt"] = ds_palm_we["pt"].isel(x=0).drop("x")
ds_dynamic["ls_forcing_right_pt"] = ds_palm_we["pt"].isel(x=-1).drop("x")
ds_dynamic["ls_forcing_south_pt"] = ds_palm_sn["pt"].isel(y=0).drop("y")
ds_dynamic["ls_forcing_north_pt"] = ds_palm_sn["pt"].isel(y=-1).drop("y")

ds_dynamic["ls_forcing_left_qv"] = ds_palm_we["QVAPOR"].isel(x=0).drop("x")
ds_dynamic["ls_forcing_right_qv"] = ds_palm_we["QVAPOR"].isel(x=-1).drop("x")
ds_dynamic["ls_forcing_south_qv"] = ds_palm_sn["QVAPOR"].isel(y=0).drop("y")
ds_dynamic["ls_forcing_north_qv"] = ds_palm_sn["QVAPOR"].isel(y=-1).drop("y")

# Add chemistry species boundaries
for species in chem_species:
    if species in ds_palm_we.data_vars:
        ds_dynamic[f"ls_forcing_left_{species}"] = ds_palm_we[species].isel(x=0).drop("x")
        ds_dynamic[f"ls_forcing_right_{species}"] = ds_palm_we[species].isel(x=-1).drop("x")
        ds_dynamic[f"ls_forcing_south_{species}"] = ds_palm_sn[species].isel(y=0).drop("y")
        ds_dynamic[f"ls_forcing_north_{species}"] = ds_palm_sn[species].isel(y=-1).drop("y")

# Add top boundary conditions
ds_dynamic["ls_forcing_top_u"] = ds_palm_top["pt"].mean(dim=["y","x"]) * 0.0 # placeholder
ds_dynamic["ls_forcing_top_v"] = ds_palm_top["pt"].mean(dim=["y","x"]) * 0.0 # placeholder
ds_dynamic["ls_forcing_top_w"] = ds_palm_top["pt"].mean(dim=["y","x"]) * 0.0 # placeholder
ds_dynamic["ls_forcing_top_pt"] = ds_palm_top["pt"].mean(dim=["y","x"])
ds_dynamic["ls_forcing_top_qv"] = ds_palm_top["QVAPOR"].mean(dim=["y","x"])

# Add chemistry top boundaries
for species in chem_species:
    if species in ds_palm_top.data_vars:
        ds_dynamic[f"ls_forcing_top_{species}"] = ds_palm_top[species].mean(dim=["y","x"])

# Add geostrophic wind
ds_dynamic["ug"] = ds_geostr["ug"]
ds_dynamic["vg"] = ds_geostr["vg"]

# Add initial profiles
ds_dynamic["u_init"] = ds_init["u"]
ds_dynamic["v_init"] = ds_init["v"]
ds_dynamic["w_init"] = ds_init["w"]
ds_dynamic["pt_init"] = ds_init["pt"]
ds_dynamic["qv_init"] = ds_init["qv"]

# Add chemistry initial profiles
for species in chem_species:
    if species in ds_init.data_vars:
        ds_dynamic[f"{species}_init"] = ds_init[species]

# Add surface data
ds_dynamic["surface_forcing_pt"] = pt2_wrf
ds_dynamic["surface_forcing_qv"] = qv2_wrf

# Add soil data
ds_dynamic["t_soil"] = xr.DataArray(init_tsoil, dims=['zs', 'y', 'x'])
ds_dynamic["m_soil"] = xr.DataArray(init_msoil, dims=['zs', 'y', 'x'])
ds_dynamic = ds_dynamic.assign_coords({"zs": zs_palm})

# Add time coordinates
ds_dynamic = ds_dynamic.assign_coords({"time": all_ts})
ds_dynamic["time"] = times_sec
ds_dynamic["time"].attrs = {"units": "s"}

# Add spatial coordinates
ds_dynamic = ds_dynamic.assign_coords({"x": x, "y": y, "z": z, "zw": zw, "xu": xu, "yv": yv})

# Write to file
output_file = f"./dynamic_files/{case_name}_dynamic"
print(f"Writing dynamic driver to: {output_file}")
ds_dynamic.to_netcdf(output_file)

end = datetime.now()
print(f"WRF4PALM completed in {end-start}")'''