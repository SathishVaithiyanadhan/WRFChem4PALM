import numpy as np
import xarray as xr
from typing import Dict, List, Tuple, Optional, Union
import warnings

class WRFChemAerosolProcessor:
    """
    Process WRF-Chem bin-resolved aerosols to PALM two-mode representation.
    """
    
    # Translation table mapping SALSA composition names to WRF-Chem variable prefixes
    TRANSLATION_TABLE = {
        "SO4": ["so4"],                                         # Sulfate
        "NO": ["no3"],                                           # Nitrate
        "NH": ["nh4"],                                           # Ammonium
        "BC": ["bc"],                                            # Black Carbon
        "OC": ["oc", "asoaX", "asoa1", "asoa2", "asoa3", "asoa4", 
               "bsoaX", "bsoa1", "bsoa2", "bsoa3", "bsoa4"],    # Organic Carbon (fresh + aged)
        "SS": ["cl", "na"],                                      # Sea Salt (chloride + sodium)
        "DU": ["co3", "ca", "oin"]                               # Dust (carbonate + calcium + other inorganics)
    }
    
    # SOLUBLE species (hygroscopic, CCN-active)
    # These go to Mode A in PALM
    SOLUBLE_SPECIES = [
        'so4',      
        'no3',    
        'nh4',      
        'na',      
        'cl',       
        'asoaX',  
        'asoa1',   
        'asoa2',    
        'asoa3',   
        'asoa4',   
        'bsoaX',   
        'bsoa1',   
        'bsoa2',   
        'bsoa3',    
        'bsoa4',    
    ]
    
    # INSOLUBLE species (hydrophobic, need aging)
    INSOLUBLE_SPECIES = [
        'bc',       
        'oin',      
        'ca',      
        'co3',      
        'oc',     
    ]
    
    def __init__(self, 
                 listspec: List[str],
                 nbin: List[int],
                 reglim: List[float],
                 wrfchem_bin_limits: List[float] = None,
                 nf2a: float = 1.0,
                 number_species: List[str] = None,
                 process_cloud_aerosols: bool = False,
                 process_condensable_vapors: bool = False,
                 oc_solubility_factor: float = 0.5):
        """
        Initialize aerosol processor with SALSA bin structure.
        """
        self.listspec = listspec
        self.nbin = nbin
        self.reglim = np.array(reglim)
        self.wrfchem_bin_limits = np.array(wrfchem_bin_limits) if wrfchem_bin_limits is not None else None
        self.nf2a = nf2a
        self.process_cloud_aerosols = process_cloud_aerosols
        self.process_condensable_vapors = process_condensable_vapors
        self.oc_solubility_factor = oc_solubility_factor
        
        # Calculate bin boundaries and geometric mean diameters using SALSA method
        self._calculate_bins()
        
        # Calculate overlap with WRF-Chem bins if limits are provided
        if self.wrfchem_bin_limits is not None:
            self.overlap_ratio = self._calculate_bin_overlap()
        else:
            self.overlap_ratio = None
        
        # Handle number species
        self.number_species = number_species if number_species is not None else []
        
        # these are the composition indices
        self.composition_species = listspec.copy()
        
        # Build mapping from WRF-Chem prefixes to listspec names
        self.prefix_to_listspec = {}
        for salsa_name in self.listspec:
            if salsa_name in self.TRANSLATION_TABLE:
                for prefix in self.TRANSLATION_TABLE[salsa_name]:
                    self.prefix_to_listspec[prefix] = salsa_name
            else:
                self.prefix_to_listspec[salsa_name.lower()] = salsa_name
        
        # Expand listspec to actual WRF-Chem variables using translation table
        # This is used for scanning available variables, NOT for composition index
        self._expanded_species_for_scanning = self._expand_species_list_for_scanning()
        # Keep expanded_species for backward compatibility with main script
        self.expanded_species = self._expanded_species_for_scanning
        
        # Available species
        self.available_species = {
            'number': [],      # List of bin indices for number conc
            'mass': []         # List of base species names 
        }
        
        # Track solubility classification for each species
        self.species_solubility = {}  # Will be filled during scanning
        
        # Flag to indicate if scanning has been done
        self._scanned = False
        
        print("\n" + "="*60)
        print("AEROSOL PROCESSOR INITIALIZED")
        print("="*60)
        print(f"Number of bins: {self.nbins} (subrange1: {nbin[0]}, subrange2: {nbin[1]})")
        print("Bin limits (μm):", [f"{lim*1e6:.3f}" for lim in self.bin_limits])
        print("Geometric mean diameters (μm):", [f"{d*1e6:.3f}" for d in self.dmid])
        if self.wrfchem_bin_limits is not None:
            print("WRF-Chem bin limits (μm):", [f"{lim*1e6:.3f}" for lim in self.wrfchem_bin_limits])
        print(f"SALSA composition list: {listspec}")
        print(f"Composition indices: {len(listspec)} species")
        print(f"Expanded to {len(self._expanded_species_for_scanning)} WRF-Chem variables for scanning")
        print(f"Number species requested: {len(number_species) if number_species else 0}")
        print(f"OC solubility factor: {oc_solubility_factor}")
        print("\nSolubility classification:")
        print(f"  SOLUBLE species (Mode A): {self.SOLUBLE_SPECIES}")
        print(f"  INSOLUBLE species (Mode B): {self.INSOLUBLE_SPECIES}")
        print(f"Process cloud aerosols: {process_cloud_aerosols}")
        print(f"Process condensable vapors: {process_condensable_vapors}")
        print("="*60)
    
    def _calculate_bins(self):
        """
        Calculate bin boundaries and geometric mean diameters using SALSA method.
        
        - Two subranges: fine (d_min to d_split) and coarse (d_split to d_max)
        - nbin[0] bins in subrange 1, nbin[1] bins in subrange 2
        """
        d_min, d_split, d_max = self.reglim
        nbin1, nbin2 = self.nbin
        nbins_total = nbin1 + nbin2
        
        # Initialize arrays
        vlolim = np.zeros(nbins_total)  # Lower volume limits
        vhilim = np.zeros(nbins_total)  # Upper volume limits
        dmid = np.zeros(nbins_total)
        bin_limits = np.zeros(nbins_total + 1)  # +1 for upper boundary of last bin
        
        # Subrange 1: d_min to d_split with nbin1 bins
        ratio_d1 = d_split / d_min
        for b in range(nbin1):
            # Calculate diameter boundaries
            d_low = d_min * ratio_d1**(b / nbin1)
            d_high = d_min * ratio_d1**((b + 1) / nbin1)
            
            # Convert to volumes for internal calculations (optional)
            vlolim[b] = np.pi / 6.0 * d_low**3
            vhilim[b] = np.pi / 6.0 * d_high**3
            
            # Geometric mean diameter
            dmid[b] = np.sqrt(d_low * d_high)
            
            # Store bin limits (lower boundaries)
            bin_limits[b] = d_low
        
        # Subrange 2: d_split to d_max with nbin2 bins
        ratio_d2 = d_max / d_split
        for b in range(nbin2):
            idx = nbin1 + b
            d_low = d_split * ratio_d2**(b / nbin2)
            d_high = d_split * ratio_d2**((b + 1) / nbin2)
            
            # Convert to volumes for internal calculations (optional)
            vlolim[idx] = np.pi / 6.0 * d_low**3
            vhilim[idx] = np.pi / 6.0 * d_high**3
            
            # Geometric mean diameter
            dmid[idx] = np.sqrt(d_low * d_high)
            
            # Store bin limits (lower boundaries)
            bin_limits[idx] = d_low
        
        # Add the upper boundary of the last bin
        bin_limits[-1] = d_max
        
        self.vlolim = vlolim
        self.vhilim = vhilim
        self.dmid = dmid
        self.bin_limits = bin_limits
        self.nbins = nbins_total
    
    def _calculate_bin_overlap(self):
        """
        Calculate overlap between PALM bins and WRF-Chem bins.
    
        """
        if self.wrfchem_bin_limits is None:
            return None
        
        n_wrfchem_bins = len(self.wrfchem_bin_limits) - 1
        overlap_ratio = np.zeros((self.nbins, n_wrfchem_bins))
        
        # Convert to nanometers for integer range comparison
        palm_limits_nm = self.bin_limits * 1e9
        wrfchem_limits_nm = self.wrfchem_bin_limits * 1e9
        
        for pbin in range(self.nbins):
            palm_range = range(
                int(palm_limits_nm[pbin]), 
                int(palm_limits_nm[pbin + 1])
            )
            
            for wbin in range(n_wrfchem_bins):
                wrf_range = range(
                    int(wrfchem_limits_nm[wbin]) + 1,
                    int(wrfchem_limits_nm[wbin + 1])
                )
                
                # Calculate overlap
                overlap = len(set(palm_range) & set(wrf_range))
                if overlap > 0:
                    overlap_ratio[pbin, wbin] = overlap / len(wrf_range)
        
        return overlap_ratio
    
    def _expand_species_list_for_scanning(self):
        """
        Expand SALSA composition names to actual WRF-Chem variable prefixes
        using the translation table. This is used ONLY for scanning available
        variables, NOT for composition indices.
        """
        expanded = []
        
        for salsa_name in self.listspec:
            if salsa_name in self.TRANSLATION_TABLE:
                wrf_prefixes = self.TRANSLATION_TABLE[salsa_name]
                for prefix in wrf_prefixes:
                    expanded.append(prefix)
            else:
                # If not in translation table, use as is
                expanded.append(salsa_name.lower())
        
        # Remove duplicates while preserving order
        seen = set()
        expanded = [x for x in expanded if not (x in seen or seen.add(x))]
        
        return expanded
    
    def get_species_solubility(self, species_name: str) -> str:
        """
        Determine if a species is soluble or insoluble.
        
        Returns:
            'soluble', 'insoluble', or 'unknown'
        """
        if species_name in self.SOLUBLE_SPECIES:
            return 'soluble'
        elif species_name in self.INSOLUBLE_SPECIES:
            return 'insoluble'
        elif species_name == 'oc':
            return 'mixed'
        else:
            return 'unknown'
    
    def scan_available_species(self, ds: xr.Dataset):
        """
        Scan the dataset to find all available aerosol species.

        """
        if self._scanned:
            return
            
        print("\nScanning for available aerosol species...")
        
        # Reset available species
        self.available_species = {
            'number': [],
            'mass': []
        }
        self.species_solubility = {}
        
        # Check for number concentration bins
        for bin_idx in range(self.nbins):
            bin_name = f"a0{bin_idx+1}" if bin_idx < 9 else f"a{bin_idx+1}"
            num_var = f"num_{bin_name}"
            if num_var in ds.data_vars:
                self.available_species['number'].append(bin_idx)
                print(f"  Found number concentration: {num_var}")
        
        # Check for mass species from expanded list (for scanning only)
        soluble_found = []
        insoluble_found = []
        mixed_found = []
        unknown_found = []
        
        for base_name in self._expanded_species_for_scanning:
            found = False
            for bin_idx in range(self.nbins):
                bin_name = f"a0{bin_idx+1}" if bin_idx < 9 else f"a{bin_idx+1}"
                mass_var = f"{base_name}_{bin_name}"
                if mass_var in ds.data_vars:
                    if not found:
                        self.available_species['mass'].append(base_name)
                        found = True
                        
                        # Classify by solubility
                        solubility = self.get_species_solubility(base_name)
                        self.species_solubility[base_name] = solubility
                        
                        if solubility == 'soluble':
                            soluble_found.append(base_name)
                        elif solubility == 'insoluble':
                            insoluble_found.append(base_name)
                        elif solubility == 'mixed':
                            mixed_found.append(base_name)
                        else:
                            unknown_found.append(base_name)
                        
                    print(f"  Found mass: {mass_var}")
        
        print(f"\n  Available species summary:")
        print(f"    Number bins: {len(self.available_species['number'])}")
        print(f"    Mass species: {len(self.available_species['mass'])}")
        print(f"    Soluble species: {soluble_found}")
        print(f"    Insoluble species: {insoluble_found}")
        print(f"    Mixed species (OC): {mixed_found}")
        print(f"    Unknown species: {unknown_found}")
        
        self._scanned = True
    
    def extract_aerosol_data(self, ds: xr.Dataset, time_idx: int) -> Dict:
        """
        Extract all aerosol data from WRF-Chem dataset.
        
        """
        aerosol_data = {
            'number': {},      # bin -> number concentration (#/kg-dryair)
            'mass': {}         # species -> bin -> mass concentration (ug/kg-dryair)
        }
        
        # Get grid dimensions
        nz = len(ds.bottom_top)
        ny = len(ds.south_north)
        nx = len(ds.west_east)
        
        # Extract number concentration for each bin
        for bin_idx in self.available_species['number']:
            bin_name = f"a0{bin_idx+1}" if bin_idx < 9 else f"a{bin_idx+1}"
            num_var = f"num_{bin_name}"
            
            if num_var in ds.data_vars:
                aerosol_data['number'][bin_idx] = ds[num_var].isel(time=time_idx).values
            else:
                aerosol_data['number'][bin_idx] = np.zeros((nz, ny, nx))
        
        # Extract mass concentration for each species and bin
        for species in self.available_species['mass']:
            aerosol_data['mass'][species] = {}
            for bin_idx in range(self.nbins):
                bin_name = f"a0{bin_idx+1}" if bin_idx < 9 else f"a{bin_idx+1}"
                mass_var = f"{species}_{bin_name}"
                
                if mass_var in ds.data_vars:
                    aerosol_data['mass'][species][bin_idx] = ds[mass_var].isel(time=time_idx).values
                else:
                    # If variable doesn't exist, create zeros
                    aerosol_data['mass'][species][bin_idx] = np.zeros((nz, ny, nx))
        
        return aerosol_data
    
    def calculate_total_number(self, aerosol_data: Dict, 
                                air_density: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Calculate total number concentration in #/m3.
        
        Conversion: #/kg-dryair * air_density (kg/m3) = #/m3
        
        """
        if not aerosol_data['number']:
            # If no number data, return zeros
            sample = next(iter(aerosol_data['mass'].values())) if aerosol_data['mass'] else None
            if sample is None:
                return np.array([])
            sample_data = next(iter(sample.values()))
            return np.zeros_like(sample_data)
        
        # Get dimensions from first number field
        first_num = next(iter(aerosol_data['number'].values()))
        total_num = np.zeros_like(first_num)
        
        # Sum over all bins
        for bin_idx, num_conc in aerosol_data['number'].items():
            total_num += num_conc
        
        if air_density is not None:
            # Convert from #/kg-dryair to #/m3
            total_num = total_num * air_density
        else:
            warnings.warn("Air density not provided. Returning number in #/kg-dryair")
        
        return total_num
    
    def calculate_number_per_bin(self, aerosol_data: Dict,
                                   air_density: Optional[np.ndarray] = None,
                                   use_overlap: bool = False) -> np.ndarray:
        """
        Calculate number concentration per bin in #/m3.
        """
        if not aerosol_data['number']:
            return np.array([])
        
        # Get dimensions
        first_num = next(iter(aerosol_data['number'].values()))
        nz, ny, nx = first_num.shape
        
        # Initialize array for all bins
        num_per_bin = np.zeros((nz, ny, nx, self.nbins))
        
        if use_overlap and self.overlap_ratio is not None:
            # Use overlap mapping to distribute WRF-Chem bins to PALM bins
            for pbin in range(self.nbins):
                for wbin, num_conc in aerosol_data['number'].items():
                    if self.overlap_ratio[pbin, wbin] > 0:
                        if air_density is not None:
                            contribution = num_conc * air_density * self.overlap_ratio[pbin, wbin]
                        else:
                            contribution = num_conc * self.overlap_ratio[pbin, wbin]
                        num_per_bin[:, :, :, pbin] += contribution
        else:
            # Simple direct mapping (assumes bins align)
            for bin_idx, num_conc in aerosol_data['number'].items():
                if bin_idx < self.nbins:
                    if air_density is not None:
                        num_per_bin[:, :, :, bin_idx] = num_conc * air_density
                    else:
                        num_per_bin[:, :, :, bin_idx] = num_conc
        
        return num_per_bin
    
    def calculate_mass_fractions(self, aerosol_data: Dict,
                                   air_density: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Calculate mass fractions for PALM modes A (soluble) and B (insoluble).
        
        Mass fraction = species_mass / total_mass (dimensionless)
        
        IMPORTANT: 
            Mode A = SOLUBLE components (hygroscopic, CCN-active)
            Mode B = INSOLUBLE components (hydrophobic, need aging)
            
        MODIFIED: All species in listspec are set as soluble (Mode A),
                  Mode B (insoluble) remains zero.
        
        The output arrays use the original listspec order for composition_index.
        """
        n_species = len(self.listspec)  # Use listspec for composition dimension
        if n_species == 0:
            return None, None
        
        # Get dimensions
        if aerosol_data['number']:
            sample = next(iter(aerosol_data['number'].values()))
        else:
            sample = next(iter(next(iter(aerosol_data['mass'].values())).values()))
        nz, ny, nx = sample.shape
        
        # Initialize mass fraction arrays with listspec dimension
        mass_fracs_a = np.zeros((nz, ny, nx, n_species))  # SOLUBLE components
        mass_fracs_b = np.zeros((nz, ny, nx, n_species))  # INSOLUBLE components (set to zero)
        
        # Calculate total mass for soluble mode (all species contribute to Mode A)
        total_mass_soluble = np.zeros((nz, ny, nx))
        
        # Step 1: For each listspec species, find all WRF-Chem variables that map to it
        # and sum their masses
        species_total_mass = {}  # listspec_name -> total mass array
        
        for salsa_name in self.listspec:
            species_total_mass[salsa_name] = np.zeros((nz, ny, nx))
            
            # Get all WRF-Chem prefixes for this SALSA name
            if salsa_name in self.TRANSLATION_TABLE:
                wrf_prefixes = self.TRANSLATION_TABLE[salsa_name]
            else:
                wrf_prefixes = [salsa_name.lower()]
            
            # Sum masses from all matching WRF-Chem variables
            for prefix in wrf_prefixes:
                if prefix in aerosol_data['mass']:
                    for bin_idx, mass_conc in aerosol_data['mass'][prefix].items():
                        species_total_mass[salsa_name] += mass_conc
        
        # Step 2: Calculate total mass for soluble mode (ALL species contribute)
        for sp_idx, salsa_name in enumerate(self.listspec):
            species_mass = species_total_mass[salsa_name]
            total_mass_soluble += species_mass
        
        # Avoid division by zero
        total_mass_soluble = np.maximum(total_mass_soluble, 1e-30)
        
        # Step 3: Calculate mass fractions for each listspec species (all to Mode A)
        for sp_idx, salsa_name in enumerate(self.listspec):
            species_mass = species_total_mass[salsa_name]
            
            # ALL mass goes to soluble mode (Mode A)
            mass_fracs_a[:, :, :, sp_idx] = species_mass / total_mass_soluble
            # Mode B remains zero (no insoluble contribution)
        
        return mass_fracs_a, mass_fracs_b
    
    def calculate_air_density(self, ds: xr.Dataset, time_idx: int) -> np.ndarray:
        """
        Calculate air density from WRF-Chem variables.
        
        Using: ρ = p / (R * T)
        where:
            p = pressure (Pa)
            R = gas constant for dry air (287 J/kg/K)
            T = temperature (K)
        """
        R = 287.0  # J/kg/K
        
        # Get pressure and temperature
        if 'P' in ds and 'PB' in ds:
            # Perturbation + base state pressure
            p = (ds['P'].isel(time=time_idx) + ds['PB'].isel(time=time_idx)).values
        else:
            raise ValueError("Cannot calculate pressure: P and PB not found")
        
        if 'T' in ds and 'TK' in ds:
            # T is perturbation potential temperature, TK is temperature
            t = ds['TK'].isel(time=time_idx).values
        elif 'T' in ds:
            # Need to convert potential temperature to temperature
            # T = (theta + 300) * (p/100000)^(R/cp)
            cp = 1004.0  # J/kg/K
            theta = ds['T'].isel(time=time_idx).values + 300
            t = theta * (p / 100000.0) ** (R / cp)
        else:
            # Try to use temperature from other variables
            if 'theta' in ds:
                theta = ds['theta'].isel(time=time_idx).values
                t = theta * (p / 100000.0) ** (R / cp)
            else:
                raise ValueError("Cannot calculate temperature")
        
        # Calculate density
        rho = p / (R * t)
        
        return rho
    
    def _aggregate_mass_fractions_to_listspec(self, mf_a_wrf, mf_b_wrf, available_species_list):
        """
        Aggregate mass fractions from expanded WRF-Chem species (20) to listspec species (7).
        """
        nz, ny, nx, n_expanded = mf_a_wrf.shape
        n_listspec = len(self.listspec)
        
        # Initialize aggregated arrays
        mf_a_agg = np.zeros((nz, ny, nx, n_listspec))
        mf_b_agg = np.zeros((nz, ny, nx, n_listspec))
        
        # For each expanded species, add its contribution to the appropriate listspec index
        for exp_idx, exp_species in enumerate(available_species_list):
            if exp_species in self.prefix_to_listspec:
                listspec_name = self.prefix_to_listspec[exp_species]
                listspec_idx = self.listspec.index(listspec_name)
                
                # Add contributions
                mf_a_agg[:, :, :, listspec_idx] += mf_a_wrf[:, :, :, exp_idx]
                mf_b_agg[:, :, :, listspec_idx] += mf_b_wrf[:, :, :, exp_idx]
            else:
                print(f"Warning: Expanded species '{exp_species}' not found in prefix mapping")
        
        return mf_a_agg, mf_b_agg
    
    def prepare_palm_aerosol_variables(self, ds_wrf: xr.Dataset,
                                        ds_interp: xr.Dataset,
                                        time_indices: List[int],
                                        z_levels: np.ndarray,
                                        y_coords: np.ndarray,
                                        x_coords: np.ndarray) -> Dict:
        """
        Prepare all aerosol variables for PALM dynamic file.
        """
        n_times = len(time_indices)
        n_z = len(z_levels)
        n_y = len(y_coords)
        n_x = len(x_coords)
        
        # First, scan the ORIGINAL WRF dataset to find aerosol species
        self.scan_available_species(ds_wrf)
        
        n_species = len(self.listspec)  # Use listspec for composition dimension
        
        # Initialize output arrays with PALM grid dimensions
        palm_vars = {
            # Size-resolved number concentration (#/m3) with bin dimension
            'aerosol_num': np.zeros((n_times, n_z, n_y, n_x, self.nbins)),
            
            # Geometric mean diameters for reference (m)
            'dmid': self.dmid,
            
            # Species names from listspec (for composition_index)
            'species_names': self.listspec.copy()
        }
        
        # Initialize mass fraction arrays if species exist
        if n_species > 0:
            palm_vars['mass_fracs_a'] = np.zeros((n_times, n_z, n_y, n_x, n_species))
            palm_vars['mass_fracs_b'] = np.zeros((n_times, n_z, n_y, n_x, n_species))
        else:
            palm_vars['mass_fracs_a'] = None
            palm_vars['mass_fracs_b'] = None
        
        print("\nPreparing PALM aerosol variables...")
        print(f"  Processing {n_times} timesteps")
        print(f"  Composition species: {self.listspec}")
        print(f"  Found {len(self.available_species['number'])} number concentration bins")
        
        if n_species == 0 and len(self.available_species['number']) == 0:
            print("  WARNING: No aerosol species found in the dataset!")
            print("  Available variables in ds_wrf:", list(ds_wrf.data_vars.keys())[:20])
            return palm_vars
        
        # Get WRF vertical levels
        if 'Z' in ds_wrf:
            wrf_z = ds_wrf['Z'].isel(time=0, west_east=0, south_north=0).values
        else:
            # Use index-based levels
            wrf_z = np.arange(len(ds_wrf.bottom_top))
            print(f"  Warning: Z not found, using index-based levels")
        
        # Create mapping from PALM grid to WRF grid indices
        # This is a simplified approach - assumes regular grid spacing
        palm_y_indices = np.linspace(0, ds_wrf.dims['south_north'] - 1, n_y).astype(int)
        palm_x_indices = np.linspace(0, ds_wrf.dims['west_east'] - 1, n_x).astype(int)
        
        # Get the list of available species in the order they appear
        available_species_list = self.available_species['mass']
        
        # For each time step
        for t_idx, time_idx in enumerate(time_indices):
            print(f"  Processing time index {time_idx} ({t_idx+1}/{n_times})")
            
            # Extract aerosol data from ORIGINAL WRF dataset
            aerosol_data = self.extract_aerosol_data(ds_wrf, time_idx)
            
            # Calculate air density for unit conversion
            try:
                air_density = self.calculate_air_density(ds_wrf, time_idx)
            except Exception as e:
                print(f"    Warning: Could not calculate air density: {e}")
                print("    Using placeholder values - numbers will be in #/kg-dryair")
                air_density = None
            
            #-------------------------------------------------------------------
            # Process size-resolved number concentration
            #-------------------------------------------------------------------
            # Calculate number per bin on WRF grid with overlap mapping if available
            num_per_bin_wrf = self.calculate_number_per_bin(
                aerosol_data, air_density, use_overlap=True
            )
            
            if num_per_bin_wrf.size > 0:
                # For each bin, interpolate vertically to PALM levels
                for bin_idx in range(self.nbins):
                    for y_idx, wy in enumerate(palm_y_indices):
                        for x_idx, wx in enumerate(palm_x_indices):
                            # Extract vertical profile from WRF grid for this bin
                            wrf_profile = num_per_bin_wrf[:, wy, wx, bin_idx]
                            
                            # Handle NaN values
                            valid_mask = ~np.isnan(wrf_profile)
                            if np.any(valid_mask):
                                valid_z = wrf_z[valid_mask]
                                valid_vals = wrf_profile[valid_mask]
                                
                                # Interpolate to PALM levels
                                palm_profile = np.interp(z_levels, valid_z, valid_vals,
                                                        left=valid_vals[0], right=valid_vals[-1])
                            else:
                                palm_profile = np.zeros_like(z_levels)
                            
                            palm_vars['aerosol_num'][t_idx, :, y_idx, x_idx, bin_idx] = palm_profile
            
            #-------------------------------------------------------------------
            # Process mass fractions (soluble/insoluble)
            #-------------------------------------------------------------------
            if n_species > 0:
                # Calculate mass fractions - this NOW returns 7 species (listspec size)
                mf_a_wrf, mf_b_wrf = self.calculate_mass_fractions(aerosol_data, air_density)
                
                if mf_a_wrf is not None:
                    # mf_a_wrf already has shape [z, y, x, 7]
                    # Directly interpolate to PALM vertical levels
                    for sp_idx in range(n_species):
                        for y_idx, wy in enumerate(palm_y_indices):
                            for x_idx, wx in enumerate(palm_x_indices):
                                # Extract vertical profiles for this species
                                mf_a_profile = mf_a_wrf[:, wy, wx, sp_idx]
                                mf_b_profile = mf_b_wrf[:, wy, wx, sp_idx]
                                
                                # Handle NaN values for mode A (soluble)
                                valid_mask_a = ~np.isnan(mf_a_profile)
                                if np.any(valid_mask_a):
                                    valid_z_a = wrf_z[valid_mask_a]
                                    valid_vals_a = mf_a_profile[valid_mask_a]
                                    palm_mf_a = np.interp(z_levels, valid_z_a, valid_vals_a,
                                                        left=valid_vals_a[0], right=valid_vals_a[-1])
                                else:
                                    palm_mf_a = np.zeros_like(z_levels)
                                
                                # Handle NaN values for mode B (insoluble)
                                valid_mask_b = ~np.isnan(mf_b_profile)
                                if np.any(valid_mask_b):
                                    valid_z_b = wrf_z[valid_mask_b]
                                    valid_vals_b = mf_b_profile[valid_mask_b]
                                    palm_mf_b = np.interp(z_levels, valid_z_b, valid_vals_b,
                                                        left=valid_vals_b[0], right=valid_vals_b[-1])
                                else:
                                    palm_mf_b = np.zeros_like(z_levels)
                                
                                palm_vars['mass_fracs_a'][t_idx, :, y_idx, x_idx, sp_idx] = palm_mf_a
                                palm_vars['mass_fracs_b'][t_idx, :, y_idx, x_idx, sp_idx] = palm_mf_b
        
        # Print statistics
        print("\nAerosol statistics:")
        print(f"  Number concentration range: {palm_vars['aerosol_num'].min():.2e} - {palm_vars['aerosol_num'].max():.2e} #/m3")
        if palm_vars['mass_fracs_a'] is not None:
            print(f"  Soluble mass fractions (Mode A) range: {palm_vars['mass_fracs_a'].min():.2e} - {palm_vars['mass_fracs_a'].max():.2e}")
            print(f"  Insoluble mass fractions (Mode B) range: {palm_vars['mass_fracs_b'].min():.2e} - {palm_vars['mass_fracs_b'].max():.2e}")
        
        return palm_vars
    
    def get_species_names(self) -> List[str]:
        """Get list of aerosol species names for output (from listspec)."""
        return self.listspec.copy()


def setup_aerosol_processing(config: Dict) -> Tuple[bool, Optional[WRFChemAerosolProcessor]]:
    """
    Setup aerosol processing based on config.
    """
    # Check if aerosol processing is enabled
    has_aerosols = config.get('process_aerosols', False)
    
    if not has_aerosols:
        return False, None
    
    # Get SALSA parameters (all from config, no defaults)
    listspec = config.get('listspec')
    nbin = config.get('nbin')
    reglim = config.get('reglim')
    wrfchem_bin_limits = config.get('wrfchem_bin_limits', None)
    nf2a = config.get('nf2a', 1.0)
    
    # Validate required parameters
    if listspec is None:
        raise ValueError("listspec must be provided in config when aerosol processing is enabled")
    if nbin is None:
        raise ValueError("nbin must be provided in config when aerosol processing is enabled")
    if reglim is None:
        raise ValueError("reglim must be provided in config when aerosol processing is enabled")
    
    # Get species lists
    number_species = config.get('number_species', [])
    
    # Get optional processing flags
    process_cloud = config.get('process_cloud_aerosols', False)
    process_vapors = config.get('process_condensable_vapors', False)
    
    # Get OC solubility factor
    oc_solubility = config.get('oc_solubility_factor', 0.5)
    
    # Initialize processor with SALSA parameters
    processor = WRFChemAerosolProcessor(
        listspec=listspec,
        nbin=nbin,
        reglim=reglim,
        wrfchem_bin_limits=wrfchem_bin_limits,
        nf2a=nf2a,
        number_species=number_species,
        process_cloud_aerosols=process_cloud,
        process_condensable_vapors=process_vapors,
        oc_solubility_factor=oc_solubility
    )
    
    return True, processor