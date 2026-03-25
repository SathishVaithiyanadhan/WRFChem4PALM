# WRFChem4PALM

WRFChem4PALM is an extension of WRF4PALM that handles meteorology, radiation, chemistry, and aerosols for generating PALM dynamic driver files from WRF-Chem output. This tool processes WRF-Chem data to create boundary conditions, initial profiles, and time-dependent forcing for PALM large-eddy simulations.


## What's New

### Version 2.0 (current)
- **Aerosol Processing**: Full integration of SALSA aerosol scheme from Project B, enabling processing of up to 7 aerosol species with 8 size bins
- **WRF-Chem Support**: Complete support for all WRF-Chem aerosol variables including mass concentrations and number distributions
- **Translation Table**: Automatic mapping between PALM aerosol species and WRF-Chem variables (SO4→so4, OC→oc/asoa/bsoa, BC→bc, SS→cl/na, NH→nh4, NO→no3, DU→co3/ca/oin)
- **Upwind Method**: Advanced upwind location method for initial aerosol profiles based on wind direction at each vertical level
- **Multi-species Processing**: Ability to process up to 140+ chemistry and aerosol species simultaneously
- **Enhanced Output**: Added aerosol-specific global attributes including bin sizes, reglim, species order, and nf2a factor
- **Improved Metadata**: Comprehensive global attributes with author attribution and institution information
- **Geostrophic Wind**: Support for geostrophic wind calculation from geopotential height (z) or pressure levels (p)
- **Radiation Processing**: Full support for shortwave and longwave radiation data with spatial averaging

### Version 1.1 (Previous)
- Multiple WRF output files allowed (list or glob patterns)
- Migration from wrf-python to salem for reduced RAM usage and faster computation
- xarray implementation for improved memory management and lazy loading
- Multiprocessing support to reduce computation time (configurable `max_pool`)
- User-friendly namelist configuration instead of script editing
- Surface variables (U10, V10, T2, Q2) for surface NaN solver
- Automatic WRF projection detection for domain location
- Geostrophic wind estimation using geopotential height

## Features

### Meteorology
- Wind components (U, V, W) at all grid points and boundaries
- Potential temperature (PT) and water vapor mixing ratio (QVAPOR)
- Surface pressure and 10m wind components
- Soil temperature and moisture (multi-layer) with vertical interpolation
- Vertical grid stretching support (dz_stretch_factor, dz_stretch_level, dz_max)
- Support for both staggered and unstaggered grid coordinates

### Chemistry
- **Gas phase species**: NO, NO2, O3, HNO3, HO2, HO, NH3, SO2, CO, H2SO4 (sulf)
- **Aggregated species**: 
  - RH (reactive hydrocarbons) from 13 component species
  - RO2 (peroxy radicals) from 15 component species
  - RCHO (aldehydes) from 7 component species
  - OCSV (semi-volatile organic compounds) from 6 component species
  - OCNV (non-volatile organic compounds) from 4 component species
- **Particulate matter**: PM10 and PM2.5 (converted from µg/m³ to kg/m³)
- **Traffic emission support**: NO_traffic and NO2_traffic (copied from base species)

### Aerosols (SALSA Scheme)
- **Supported PALM species**: SO4 (sulfate), OC (organic carbon), BC (black carbon), SS (sea salt), NH (ammonium), NO (nitrate), DU (dust)
- **Automatic mapping from WRF-Chem variables**:
  - `SO4` → `so4_a01`, `so4_a02`, `so4_a03`, `so4_a04`
  - `OC` → `oc_a01-04`, `asoaX_a01-04`, `asoa1-4_a01-04`, `bsoaX_a01-04`, `bsoa1-4_a01-04` (20+ variables)
  - `BC` → `bc_a01-04`
  - `SS` → `cl_a01-04`, `na_a01-04`
  - `NH` → `nh4_a01-04`
  - `NO` → `no3_a01-04`
  - `DU` → `co3_a01-04`, `ca_a01-04`, `oin_a01-04`
- **Bin structure**: Configurable subranges (default: 1 bin in first subrange, 7 bins in second = 8 total bins)
- **Number concentration**: Full support for 4 WRF-Chem bins (`num_a01` through `num_a04`) with overlap ratio mapping
- **Mass fractions**: Separate calculations for soluble (Mode A) and insoluble (Mode B) components
- **Upwind initialization**: Initial profiles calculated using upwind location based on wind direction

### Radiation
- Shortwave incoming radiation (SWDOWN) - W/m²
- Longwave incoming radiation (GLW) - W/m²
- Diffuse shortwave radiation (SWDDIF) - W/m²
- Spatial averaging over PALM domain with configurable smoothing distance
- Time interpolation to match PALM simulation timesteps

### Data Processing
- Horizontal interpolation (linear or nearest) from WRF to PALM grid
- Vertical interpolation of all variables to PALM vertical levels
- Surface NaN resolution using logarithmic (wind) and linear (scalars) interpolation
- Boundary condition extraction for all four lateral boundaries and top
- Mass conservation adjustment for wind fields
- Traffic variable handling (duplicate base species with suffix)

## Installation

### Conda Environment Setup

# Create the environment from the provided YAML file
conda env create -f wrf4palm_env.yml

# Activate the environment
conda activate wrf4palm

##### users don't have to edit the main script, and only need to edit the namelist file to provide their input (for examples please see namelist.wrf4palm).

#### case
In the `case` section, users need to provide their case name and the maximum number of CPUs they want to use in WRF4PALM (here the number is 4).

```
[case]
case_name = "wrf4palm_test", # specify your case name here
max_pool = 4,                # specify the maximum number of CPUs to use
```

#### domain
In the `domain` section, users need to provide PALM domain configuration (dx, dy, dz, nx, ny, nz, and z_origin), the latitude and longitude at PALM domain centre, and the projection of PALM domain. The projection of PALM domain and centre lat/lon are used to locate PALM domain in the WRF domain. The projection of PALM domain should be identical to the projection of PALM static driver, if the user has one. If users do not have the projection information, they can leave the field empty as `palm_proj = "",such that WRF4PALM v1.1 will use the projetion of WRF directly.


```
[domain]
palm_proj = "EPSG:2193",    # projection of PALM
centlat   = -35.7853,       # latitude of domain centre
centlon   = 174.1,          # longitude of domain centre
nx        = 200,            # number of grid points along x-axis
ny        = 200,            # number of grid points along y-axis
nz        = 120,            # number of grid points along z-axis
dx        = 50.0,           # number of grid points along x-axis
dy        = 50.0,           # number of grid points along y-axis
dz        = 10.0,           # number of grid points along z-axis
z_origin  = 508.0,            # elevated mean grid position (elevated terrain)
```

#### stretch
In the `stretch` section, users can define vertically stretched grid spacing. The parameters are identical to those in PALM. If no stretching is required, leave dz_stretch_factor=1.0,

```
[stretch]
dz_stretch_factor = 1.0,        # stretch factor for a vertically stretched grid
                                # set this to 1.0 if no stretching required
dz_stretch_level = 1200.0,      # Height level above which the grid is to be stretched vertically (in m)

dz_max = 30.0,                  # allowed maximum vertical grid spacing (in m)
```

#### wrf
WRF4PALM users must provide their own WRF output. Users must specify the directory (`wrf_path`) to access WRF netcdf output files, and WRF output filenames. WRF4PALM v1.1 allows users to provide one or multiple WRF files. Users can either provide a list of filenames, e.g.:  
`wrf_output = "wrfout_d04_2020-12-25_12-00-00", "wrfout_d04_2020-12-26_12-00-00"`
or a string glob in the form:
`wrf_output = "wrfout_d04_2020-12-*", `

Users also need to specify the interpolation mode (`interp_mode`) to interpolate WRF output onto PALM grid. Both `"linear"` and `"nearest"` are allowed, while we recommend using `"linear"`.

The start and end datetime of PALM simulation must be provided. The PALM dynamic driver update frequency is controlled by `dynamic_ts` (unit: seconds), e.g. `dynamic_ts = 3600.0,` means the boudnary conditions will be updated every hour.

```
[wrf]
wrf_path = "./wrf_output/",
wrf_output = "wrfout_d04_2020-12-25_12-00-00", "wrfout_d04_2020-12-26_12-00-00",

interp_mode = "linear",

start_year = 2020,
start_month = 12,
start_day = 25,
start_hour = 13,

end_year = 2020,
end_month = 12,
end_day = 26,
end_hour = 10,

dynamic_ts = 3600.0,         # PALM dynamic driver update frequency (seconds)

```

**Note**: leading zeros are not permitted in the datetime configuration. For example, if the `start_month` is January, then the namelist should have `start_month = 1,` instead of `start_month = 01,`.

#### soil
In the `soil` section, users need to config the soil layers (`dz_soil`). In case when soil moisture output in WRF is all zeros (due to WRF's parameterisation), a dummy value can be chosen (e.g. `msoil = 0.3,`).

```
[soil]
# layers for soil temperature and moisture calculation
# this shall be changed depending on different cases

dz_soil = 0.01, 0.02, 0.04, 0.06, 0.14, 0.26, 0.54, 1.86,
msoil = 0.3,         # dummy value in case soil moisture from WRF output is 0.0
```

#### Chemistry
In the `chemistry` section, users need to config the initial and the boundary conditions for the species 

```
[chemistry] 
species = ["PM10", "PM2_5_DRY"], # chemical species to include from WRF-Chem - "no", "no2", "o3", "PM10", "PM2_5_DRY"

```
#### Radiation
In the `radiation` section, users need to config the radiation from WRFChem 

```
[radiation] 
radiation_from_wrf = True,
radiation_smoothing_distance = 10000.0,  # Distance (in meters) around domain center to average radiation data

```
#### Aerosol
In the `Aerosol` section, users need to config the initial and the boundary conditions for the aerosol and the mass fractions 

```
[aerosol]
aerosol_wrfchem = True,                     # Enable aerosol processing
listspec = ["SO4", "OC", "BC", "SS", "NH", "NO", "DU"],  # Aerosol chemical components
nbin = [1, 7],                              # Number of bins in each subrange [bins in subrange 1, bins in subrange 2]
reglim = [3.9e-8, 5.0e-8, 2.5e-6],         # Bin limits (m) [3.9e-8, 5.0e-8, 2.5e-6]   [2.5E-9, 1.5E-8, 1.0E-6] [3.0e-9, 1.0e-8, 2.5e-6] [d_min, d_split, d_max] in meters
wrfchem_bin_limits = [3.9e-8, 1.56e-7, 6.25e-7, 2.5e-6, 1.0e-5],  # WRF-Chem bin limits
nf2a = 1.0,    

```
### One line command
Once the namelist is ready, users can run WRF4PALM using the one line command:
```
python run_config_wrf4palm.py [your namelist]
Eg. python run_config_wrf4palm.py  Augs_Bourges_Platz.wrf4palm
```



If the execution is successful, the dynamic driver will be ready in `dynamic_files` with the `case_name` and start timestamp user specified. A cfg reference file will also be stored in `cfg_files` which contains domain configuration and soil temperature and moisture information. An example dynamic driver and an example cfg file are provided in `dynamic_files` and `cfg_files`, respectively.

## Quick compare WRF & PALM

In order for users to quickly check the quality of the dynamic driver generated by WRF4PALM, we provide a quick comparison script. Five variables are allowed (can be in uppercase or lowercase):
- U
- V
- W
- PT
- QV

Three plot types are provided:
1. **zcross**: vertical cross sections of west/east/south/north boundaries for the user specified variable and timestamp
```
python3 quick_compare.py [your namelist] zcross [variable name]
```
then the script will ask for the timestamp:
```
Please enter the timestamp (yyyy-mm-dd-hh):
```
Once the timestamp is given, the script will return a comparison plot.

2. **pr**: vertical profiless of west/east/south/north boundaries for the user specified variable and timestamp
```
python3 quick_compare.py [your namelist] pr [variable name]
```
then the script will ask for the timestamp:
```
Please enter the timestamp (yyyy-mm-dd-hh):
```
Once the timestamp is given, the script will return a comparison plot.  
Note that the vertical profiles are horizontally averaged and hence the comparison only gives a approximate reference regarding the performance of WRF4PALM.

2. **ts**: time series of west/east/south/north boundaries for the user specified variable and altitude
```
python3 quick_compare.py [your namelist] ts [variable name]
```
then the script will ask for the altitude:
```
Please enter the vertical level in m:
```
Once the vertical level is given, the script will return a comparison plot.  
Note that the time series are horizontally averaged and hence the comparison only gives a approximate reference regarding the performance of WRF4PALM.

## Remark
- [`Surface_NaN_Solver.pdf`](https://github.com/dongqi-DQ/WRF4PALM/blob/v1.1/Surface_NaN_Solver.pdf) provides a short documentation explaining how the surface nans are resolved.
- The WRF4PALM v1.1 python environment is available in [`wrf4palm_env.yml`](https://github.com/dongqi-DQ/WRF4PALM/blob/v1.1/wrf4palm_env.yml).

# Note  
- We noticed that PALM uses a water temperature of 283 K as default, which may lead to a stable layer over water bodies (if there are any in the PALM simulation domain). We recommend users to modify the water temperature using the static driver.
- We may release a static driver generator using global data set from Google earth engine and SST from ERA5 (date TBC).
- Geostrophic winds are only an estimation while the accuracy of the estimation still needs further discussion and investigation. This problem is the same in INIFOR.
- We encourage WRF4PALM users to use the GitHub **Issue** system if they encountered any issues or problems using WRF4PALM such that communications and trouble shooting will be easier.

--------------------------------------------------------------------------------------------
### End of README
--------------------------------------------------------------------------------------------

Development of WRFChem4PALM is based on WRF4PALM (https://github.com/dongqi-DQ/WRF4PALM) with the additional Chemistry, Radiation and Aerosol features.

A full documentation is still under construction, if you have any queries please contact the author or open a new issue.

--------------------------------------------------------------------------------------------
**Contact: Dongqi Lin (dongqi.lin@pg.canterbury.ac.nz)
Sathish Kumar Vaithiyanadhan (sathishvaithiyanadhan@gmail.com) -- Chemistry, Radiation and Aerosol part**

