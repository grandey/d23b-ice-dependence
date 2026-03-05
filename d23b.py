"""
d23b:
    Functions that support the analysis contained in the d23b-ice-dependence repository.

Author:
    Benjamin S. Grandey, 2023-2026.
"""


from functools import cache
import itertools
import json
import math
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path
import pyvinecopulib as pv
from scipy import stats
import seaborn as sns
import statsmodels.formula.api as smf
from watermark import watermark
import xarray as xr


# Matplotlib settings
plt.rcParams['figure.titlesize'] = 'x-large'  # suptitle
plt.rcParams['figure.titleweight'] = 'bold'  # suptitle
plt.rcParams['axes.titlesize'] = 'x-large'
plt.rcParams['axes.titleweight'] = 'bold'
plt.rcParams['axes.labelsize'] = 'large'
plt.rcParams['axes.labelweight'] = 'bold'
plt.rcParams['xtick.labelsize'] = 'large'
plt.rcParams['ytick.labelsize'] = 'large'
plt.rcParams['savefig.dpi'] = 300

# Seaborn style
SNS_STYLE = 'whitegrid'  # default seaborn style to use
sns.set_style(SNS_STYLE)


# Constants
IN_BASE = Path.cwd() / 'data'  # base directory of input data
COMPONENTS = ['EAIS', 'WAIS', 'GrIS']  # ice sheet components of sea level, ordered according to vine copula
CONVERT_GT_M = 1. / 362.5 / 1e3  # ice mass above floatation to SLE: 362.5 Gt ~ 1 mm SLE (Goelzer et al, 2020)
S20_EXP_DF = pd.DataFrame(  # Seroussi et al. (2020) experiments, from Table 1 of Seroussi et al. (2020)
    [['exp01', 'NorESM1-M', 'RCP8.5', 'Open', 'Medium', 'No'],
     ['exp02', 'MIROC-ESM-CHEM', 'RCP8.5', 'Open', 'Medium', 'No'],
     #['exp03', 'NorESM1-M', 'RCP2.6', 'Open', 'Medium', 'No'],  # use only RCP8.5 simulations
     ['exp04', 'CCSM4', 'RCP8.5', 'Open', 'Medium', 'No'],
     ['exp05', 'NorESM1-M', 'RCP8.5', 'Standard', 'Medium', 'No'],
     ['exp06', 'MIROC-ESM-CHEM', 'RCP8.5', 'Standard', 'Medium', 'No'],
     #['exp07', 'NorESM1-M', 'RCP2.6', 'Standard', 'Medium', 'No'],
     ['exp08', 'CCSM4', 'RCP8.5', 'Standard', 'Medium', 'No'],
     ['exp09', 'NorESM1-M', 'RCP8.5', 'Standard', 'High', 'No'],
     ['exp10', 'NorESM1-M', 'RCP8.5', 'Standard', 'Low', 'No'],
     ['exp11', 'CCSM4', 'RCP8.5', 'Open', 'Medium', 'Yes'],
     ['exp12', 'CCSM4', 'RCP8.5', 'Standard', 'Medium', 'Yes'],
     ['exp13', 'NorESM1-M', 'RCP8.5', 'Standard', 'PIGL', 'No'],
     ['expA1', 'HadGEM2-ES', 'RCP8.5', 'Open', 'Medium', 'No'],
     ['expA2', 'CSIRO-MK3', 'RCP8.5', 'Open', 'Medium', 'No'],
     ['expA3', 'IPSL-CM5A-MR', 'RCP8.5', 'Open', 'Medium', 'No'],
     #['expA4', 'IPSL-CM5A-MR', 'RCP2.6', 'Open', 'Medium', 'No'],
     ['expA5', 'HadGEM2-ES', 'RCP8.5', 'Standard', 'Medium', 'No'],
     ['expA6', 'CSIRO-MK3', 'RCP8.5', 'Standard', 'Medium', 'No'],
     ['expA7', 'IPSL-CM5A-MR', 'RCP8.5', 'Standard', 'Medium', 'No'],
     #['expA8', 'IPSL-CM5A-MR', 'RCP2.6', 'Standard', 'Medium', 'No']
     ],
    columns=['Experiment', 'ESM', 'Scenario', 'Ocean forcing', 'Ocean sensitivity', 'Ice shelf fracture'])
S20_EXP_DF.set_index(S20_EXP_DF['Experiment'].str.strip('exp').values, inplace=True)
P21_EXP_DF = pd.DataFrame(  # Payne et al. (2020) experiments, from https://doi.org/10.5281/zenodo.4498331 README.txt
    [['expB1', 'CNRM-CM6-1', 'ssp585,', 'Standard'],
     #['expB2', 'CNRM-CM6-1', 'ssp126,', 'Standard'],  # use only SSP5-8.5 simulations
     ['expB3', 'UKESM1-0-LL', 'ssp585,', 'Standard'],
     ['expB4', 'CESM2', 'ssp585,', 'Standard'],
     ['expB5', 'CNRM-ESM2-1', 'ssp585,', 'Standard'],
     ['expB6', 'CNRM-CM6-1', 'ssp585,', 'Open'],
     #['expB7', 'CNRM-CM6-1', 'ssp126,', 'Open'],
     ['expB8', 'UKESM1-0-LL', 'ssp585,', 'Open'],
     ['expB9', 'CESM2', 'ssp585,', 'Open'],
     ['expB10', 'CNRM-ESM2-1', 'ssp585,', 'Open']],
    columns=['Experiment', 'ESM', 'Scenario', 'Ocean forcing'])
P21_EXP_DF.set_index(P21_EXP_DF['Experiment'].str.strip('exp').values, inplace=True)
WORKFLOW_LABELS = {'wf_1e': 'Workflow 1e corr.',  # labels of "workflows" used for the correlation structures
                   'wf_4': 'Workflow 4 corr.',
                   'wf_2e': 'Workflow 2e corr.',
                   'wf_3e': 'Workflow 3e corr.',
                   'S20+P21+L23': 'Combined ensemble corr.',
                   'S20+P21': 'ISMIP6 ensemble corr.',
                   'L23': 'L23 ensemble corr.',
                   '0': 'Independence',  # idealized independence
                   '1': 'Perfect correlation',  # idealized perfect dependence
                   '10': 'Antarctic correlation',  # perfect dependence & independence
                   '01': f'{COMPONENTS[1]}–{COMPONENTS[2]} perfect corr.',  # independence & perfect dependence
                   }
WORKFLOW_NOTES = {'wf_1e': '$\\bf{Workflow\ 1e}$\n(shared dependence on GSAT;\nEdwards et al., 2021)',
                  'wf_4': '$\\bf{Workflow\ 4}$\n(structured expert judgment;\nBamber et al., 2019)',
                  'wf_2e': '$\\bf{Workflow\ 2e}$\n(basal melt emulator;\nLevermann et al., 2020)',
                  'wf_3e': '$\\bf{Workflow\ 3e}$\n(perturbed parameter;\nDeConto et al., 2021)',
                  'S20+P21+L23': ('$\\bf{Combined\ ensemble}$\n(Seroussi et al., 2020;\n'
                                  'Payne et al., 2021; Li et al., 2023)'),
                  'S20+P21': '$\\bf{ISMIP6\ ensemble}$\n(Seroussi et al., 2020;\nPayne et al., 2021)',
                  'L23': '$\\bf{L23\ ensemble}$\n(Li. et al., 2023)',
                  '0': '$\\bf{Independence}$\n(idealized)',
                  '1': '$\\bf{Perfect\ correlation}$\n(idealized)',
                  '10': '$\\bf{Antarctic\ correlation}$\n(idealized)',
                  }  # WORKFLOW_NOTES is used by fig_dependence_table()
WORKFLOW_COLORS = {'wf_1e': 'darkblue',  # colors used by ax_total_vs_time(), ax_sum_vs_gris_fingerprint()
                   'wf_4': 'darkgreen',
                   'wf_2e': 'tomato',
                   'wf_3e': 'darkred',
                   'S20+P21+L23': 'purple',
                   'S20+P21': 'blue',
                   'L23': 'red',
                   '0': 'lightslategrey',
                   '1': 'brown',
                   '10': 'darkorange',
                   '01': 'peru',
                   }
TAU_REG = r'$\tau$'  # tau (regular font)
TAU_BOLD = r'$\bf{\tau}$'  # tau (bold font)
FIG_DIR = Path.cwd() / 'figs_d23b'  # directory in which to save figures
F_NUM = itertools.count(1)  # main figures counter
S_NUM = itertools.count(1)  # supplementary figures counter
O_NUM = itertools.count(1)  # other figures counter


def get_watermark():
    """Return watermark string, including versions of dependencies."""
    packages = ('matplotlib,numpy,pandas,pyvinecopulib,scipy,seaborn,xarray')
    return watermark(machine=True, conda=True, python=True, packages=packages)


@cache
def read_ar6_samples(workflow='wf_1e', component='EAIS', scenario='ssp585', year=2100):
    """
    Return samples from the AR6 GMSLR projections for a specified workflow, component, scenario, and year.

    Parameters
    ----------
    workflow : str
        AR6 workflow.  Options are 'wf_1e' (default), 'wf_2e', 'wf_3e', or 'wf_4'.
    component : str
        Component of GMSLR. Options are 'EAIS' (East Antarctic Ice Sheet, default),
        'WAIS' (West Antarctic Ice Sheet), 'GrIS' (Greenland Ice Sheet), and 'GMSLR' (total GMSLR).
        Note, for wf_1e and wf_2e, 'PEN' (Antarctic peninsula) is also included in 'WAIS'.
    scenario : str
        Options are 'ssp126' and 'ssp585' (default).
    year : int
        Year. Default is 2100.

    Returns
    -------
    samples_da : xarray DataArray
        DataArray containing different samples of specified component of GMSLR, in metres.
    """
    # Identify input file, based on component and workflow
    if component == 'GMSLR':  # total GMSLR
        in_dir = IN_BASE / 'ar6' / 'global' / 'full_sample_workflows' / workflow / scenario
        in_fn = in_dir / 'total-workflow.nc'
    elif component == 'GrIS':  # Greenland
        in_dir = IN_BASE / 'ar6' / 'global' / 'full_sample_components'
        if workflow in ['wf_1e', 'wf_2e', 'wf_3e']:
            gris_source = 'ipccar6-ismipemu'
        elif workflow == 'wf_4':
            gris_source = 'ipccar6-bamber'
        in_fn = in_dir / f'icesheets-{gris_source}icesheet-{scenario}_GIS_globalsl.nc'
    elif component in ['EAIS', 'WAIS', 'PEN']:  # Antarctica
        in_dir = IN_BASE / 'ar6' / 'global' / 'full_sample_components'
        if workflow == 'wf_1e':
            ais_source = 'ipccar6-ismipemu'
        elif workflow == 'wf_2e':
            ais_source = 'ipccar6-larmip'
        elif workflow == 'wf_3e':
            ais_source = 'dp20-'
        elif workflow == 'wf_4':
            ais_source = 'ipccar6-bamber'
        in_fn = in_dir / f'icesheets-{ais_source}icesheet-{scenario}_{component}_globalsl.nc'
    else:
        raise ValueError(f'Unrecognised parameter value: component={component}')
    # Does input file exist?
    if not in_fn.exists():
        raise FileNotFoundError(in_fn)
    # Read data
    samples_da = xr.open_dataset(in_fn)['sea_level_change'].squeeze().drop_vars('locations').sel(years=year)
    # Change units from mm to m
    samples_da = samples_da / 1000.
    samples_da.attrs['units'] = 'm'
    # For wf_1e, also include PEN in WAIS (implicitly preserving dependence structure of samples)
    if workflow in ['wf_1e', 'wf_2e'] and component == 'WAIS':
        samples_da += read_ar6_samples(workflow=workflow, component='PEN', scenario=scenario, year=year)
        print(f'read_ar6_samples({workflow}, {component}, {scenario}, {year}): including PEN in WAIS')
    # Return result (without sorting/ordering)
    return samples_da


@cache
def read_ism_ensemble_data(ensemble='S20+P21+L23', ref_year=2015, target_year=2100, fully_crossed=True):
    """
    Read Antarctic ISM ensemble data from Seroussi et al. (2020), Payne et al. (2021), and Li et al. (2023).

    This function uses data from https://doi.org/10.5281/zenodo.3940766, https://doi.org/10.5281/zenodo.4498331,
    and https://doi.org/10.5281/zenodo.7380180.

    Parameters
    ----------
    ensemble : str
        Ensemble to read. Options are 'S20' (Seroussi et al.), 'P21' (Payne et al.), 'L23' (Li et al.), and
        combinations such as 'S20+P21+L23' (default).
    ref_year : int
        Reference year. Default is 2015 (which is the start year for the P21 data).
    target_year : int
        Target year for difference. Default is 2100.
    fully_crossed : bool
        If True, only return data for ISMs that include every ESM within a given ensemble (S20, P21, or L23).
        Default is True.
        Note: a combined ensemble (e.g. S20+P21+L23) will not be fully crossed.

    Returns
    -------
    ism_df : pandas DataFrame
        A DataFrame containing EAIS and WAIS components (in m), Ensemble (S20, P21, or L23), Exp, ESM, and ISM.

    Notes
    -----
    For convenience, a 'GrIS' column is included, populated with zeros. This enables fitting of a vine copula.
    """
    # DataFrame to hold data
    ism_df = pd.DataFrame(columns=['EAIS', 'WAIS', 'Ensemble', 'Exp', 'ESM', 'ISM'])
    # If combined ensemble, call recursively
    if '+' in ensemble:
        for ens in ensemble.split('+'):
            temp_df = read_ism_ensemble_data(ensemble=ens, ref_year=ref_year, target_year=target_year,
                                             fully_crossed=fully_crossed)
            if ism_df.empty:  # avoid FutureWarning about concatenating an empty DataFrame
                ism_df = temp_df
            else:
                ism_df = pd.concat([ism_df, temp_df], ignore_index=True)
    # Seroussi et al data
    elif ensemble == 'S20':
        # Location of S20 data
        in_dir = IN_BASE / 'ComputedScalarsPaper'
        # Loop over experiments
        for exp, exp_ser in S20_EXP_DF.iterrows():
            print(f'Reading {ensemble} exp{exp} data.')
            # Loop over available input files
            in_fns = sorted(in_dir.glob(f'*/*/exp{exp}/computed_ivaf_minus_ctrl_proj_AIS_*_exp{exp}.nc'))
            for in_fn in in_fns:
                # Create dictionary to hold data for this input file, including experiment info
                ais_dict = {'Ensemble': ensemble, 'Exp': exp, 'ESM': exp_ser['ESM']}
                # Get ice sheet model info
                ism_info = '_'.join(str(in_fn).split('/')[-4:-2])  # institute and model name
                ism_info += '_' + exp_ser['Ocean forcing']  # include ocean forcing protocol info
                if exp_ser['Ocean sensitivity'] != 'Medium':  # include ocean sensitivity info if not medium
                    ism_info += '_' + exp_ser['Ocean sensitivity']
                if exp_ser['Ice shelf fracture'] == 'Yes':  # if ice shelf fracture included, indicate this
                    ism_info += '_Fracture'
                ais_dict['ISM'] = ism_info
                # Read DataSet
                in_ds = xr.load_dataset(in_fn, decode_times=False)
                # Calculate SLE for target year relative to reference year for EAIS and WAIS; invert sign
                eais_da = in_ds[f'ivaf_region_{2}']
                wais_da = in_ds[f'ivaf_region_{1}'] + in_ds[f'ivaf_region_{3}']  # include peninsula in WAIS
                try:
                    ice_density = float(in_ds['rhoi']) / 1e12  # Gt / m3
                except KeyError:
                    ice_density = 910 / 1e12  # Gt / m3
                    print(f'No ice density found in {in_fn.name}. Using {ice_density} Gt / m3.')
                for region_name, in_da in [('EAIS', eais_da), ('WAIS', wais_da)]:
                    if ref_year == 2015:
                        ais_dict[region_name] = -1. * float(in_da.sel(time=target_year)) * ice_density * CONVERT_GT_M
                    else:
                        ais_dict[region_name] = -1. * float(in_da.sel(time=target_year) -
                                                            in_da.sel(time=ref_year)) * ice_density * CONVERT_GT_M
                # Append to DataFrame
                ism_df.loc[len(ism_df)] = ais_dict
    # Payne et al. data
    elif ensemble == 'P21':
        # Location of P21 data
        in_dir = IN_BASE / 'CMIP5_CMIP6_Scalars_Paper' / 'AIS' / 'Ice'
        # Loop over experiments
        for exp, exp_ser in P21_EXP_DF.iterrows():
            print(f'Reading {ensemble} exp{exp} data.')
            # Loop over available input files
            in_fns = sorted(in_dir.glob(f'computed_limnsw_minus_ctrl_proj_AIS_*_exp{exp}.nc'))
            for in_fn in in_fns:
                # Create dictionary to hold data for this input file, including experiment info
                ais_dict = {'Ensemble': ensemble, 'Exp': exp, 'ESM': exp_ser['ESM']}
                # Get ice sheet model info
                ism_info = in_fn.name.split('AIS_')[-1].split('_exp')[0]
                ism_info += '_' + exp_ser['Ocean forcing']  # include ocean forcing protocol info
                ais_dict['ISM'] = ism_info
                # Read DataSet
                in_ds = xr.load_dataset(in_fn, decode_times=False)
                # Calculate SLE for target year relative to reference year for EAIS and WAIS; invert sign
                eais_da = in_ds[f'limnsw_region_{2}']
                wais_da = in_ds[f'limnsw_region_{1}'] + in_ds[f'limnsw_region_{3}']  # include peninsula in WAIS
                for region_name, in_da in [('EAIS', eais_da), ('WAIS', wais_da)]:
                    if ref_year == 2015:
                        ais_dict[region_name] = -1. * float(in_da.sel(time=target_year)) * CONVERT_GT_M
                    else:
                        ais_dict[region_name] = -1. * float(in_da.sel(time=target_year) -
                                                            in_da.sel(time=ref_year)) * CONVERT_GT_M
                # Append to DataFrame
                ism_df.loc[len(ism_df)] = ais_dict
    # Li et al. data
    elif ensemble == 'L23':
        # Loop over experiments
        exp_list = ['CMIP6_BC_1850-2100', 'CMIP6_BC_1850-2100_NO_MICI']
        for exp in exp_list:
            print(f'Reading {ensemble} {exp} data.')
            # Loop over available input files for different ESMs
            in_dir = IN_BASE / exp
            in_fns = sorted(in_dir.glob('*/fort.22'))
            for in_fn in in_fns:
                # Create dictionary to hold data for this input file, including model info
                if exp == 'CMIP6_BC_1850-2100':
                    ism_info = 'L23_MICI'
                elif exp == 'CMIP6_BC_1850-2100_NO_MICI':
                    ism_info = 'L23_NO_MICI'
                else:
                    print(f'Unknown experiment {exp}')
                    ism_info = None
                ais_dict = {'Ensemble': ensemble, 'Exp': exp, 'ESM': str(in_fn).split('/')[-2], 'ISM': ism_info}
                # Read data
                try:
                    in_df = pd.read_fwf(in_fn, skiprows=1, index_col='time')
                except ValueError:
                    in_df = pd.read_fwf(in_fn, skiprows=2, index_col='time')
                # Get SLE for target year relative to reference year for WAIS and EAIS; invert sign
                for region_name, in_varname in [ ('EAIS', 'eofe(m)'),  # sea-level equivalent change in ice sheet
                                                 ('WAIS', 'eofw(m)')]:
                    ais_dict[region_name] = -1. * (in_df.loc[target_year][in_varname] - in_df.loc[ref_year][in_varname])
                # Append to DataFrame
                ism_df.loc[len(ism_df)] = ais_dict
    # Include GrIS column for convenience, enabling fitting of vine copula below, with GrIS independent of EAIS and WAIS
    ism_df['GrIS'] = 0.
    # Ensure fully crossed within each separate ensemble
    if fully_crossed and '+' not in ensemble:
        n_before = len(ism_df)  # number of rows before filtering
        n_esm = ism_df['ESM'].nunique()   # number of ESMs in total
        n_esm_per_ism_ser = ism_df.groupby('ISM')['ESM'].nunique()  # number of ESMs per ISM
        valid_ism_ind = n_esm_per_ism_ser[n_esm_per_ism_ser == n_esm].index  # ISMs with every ESM
        ism_df = ism_df[ism_df['ISM'].isin(valid_ism_ind)]
        n_after = len(ism_df)
        print(f'{ensemble}: removed {n_before - n_after} rows to ensure fully crossed.')
        print(f'{ensemble}: now {ism_df["ESM"].nunique()} ESMs x {ism_df["ISM"].nunique()} ISM configurations.')
    # Return result
    return ism_df


@cache
def read_gauge_info(gauge='TANJONG_PAGAR'):
    """
    Read name, ID, latitude, and longitude of tide gauge, using location_list.lst.
    (https://doi.org/10.5281/zenodo.6382554).

    Parameters
    ----------
    gauge : int or str
        ID or name of gauge. Default is 'TANJONG_PAGAR' (equivalent to 1746).

    Returns
    ----------
    gauge_info : dict
        Dictionary containing gauge_name, gauge_id, lat, lon.
    """
    # Read input file into DataFrame
    in_fn = IN_BASE / 'location_list.lst'
    in_df = pd.read_csv(in_fn, sep='\t', names=['gauge_name', 'gauge_id', 'lat', 'lon'])
    # Get data for gauge of interest
    try:
        if type(gauge) == str:
            df = in_df[in_df.gauge_name == gauge]
        else:
            df = in_df[in_df.gauge_id == gauge]
        gauge_info = dict()
        for c in ['gauge_name', 'gauge_id', 'lat', 'lon']:
            gauge_info[c] = df[c].values[0]
    except IndexError:
        raise ValueError(f"gauge='{gauge}' not found.")
    return gauge_info


@cache
def read_gauge_grd(gauge='TANJONG_PAGAR'):
    """
    Read GRD fingerprints near a tide gauge location, using fingerprints from the FACTS module data
    (https://doi.org/10.5281/zenodo.7478192).

    Parameters
    ----------
    gauge : int or str
        ID or name of gauge. Default is 'TANJONG_PAGAR' (equivalent to 1746).

    Returns
    ----------
    gauge_grd : dict
        Dictionary containing gauge_name, gauge_id, lat, lon (from get_gauge_info), alongside
        lat_grd, lon_grd (nearest GRD data location used) and EAIS, WAIS, GrIS weights.
    """
    # Get gauge info, including location
    gauge_grd = read_gauge_info(gauge)
    lat, lon = gauge_grd['lat'], gauge_grd['lon']
    # Express longitude as +ve (deg E), for consistency with fingerprint data
    if lon < 0.:
        lon += 360
    # Get GRD fingerprint for each ice sheet, using nearest location with GRD data
    in_dir = IN_BASE / 'grd_fingerprints_data' / 'FPRINT'
    for component in ['EAIS', 'WAIS', 'GIS']:  # FACTS uses 'GIS' instead of 'GrIS'
        in_fn = in_dir / f'fprint_{component.lower()}.nc'
        try:
            in_da = xr.open_dataset(in_fn)['fp'].sel(lat=lat, lon=lon, method='nearest', tolerance=1.)
        except KeyError:
            raise ValueError(f"For gauge='{gauge}', suitable GRD data not found within lat-lon tolerance.")
        if component == 'EAIS':  # record location used for GRD data - do this only once
            gauge_grd['lat_grd'] = float(in_da['lat'].data)
            gauge_grd['lon_grd'] = float(in_da['lon'].data)
        gauge_grd[component] = float(in_da.data)*1000  # save fingerprint to dictionary
        if component == 'GIS':  # duplicate GIS as GrIS
            gauge_grd['GrIS'] = float(in_da.data)*1000
    return gauge_grd


@cache
def get_grd_df(gauges=('REYKJAVIK', 'DUBLIN', 'TANJONG_PAGAR')):
    """
    Return DataFrame of GRD fingerprints for the specified gauge locations.

    Parameters
    ----------
    gauges : tuple
        Gauge locations. Default is ['REYKJAVIK', 'DUBLIN', 'TANJONG_PAGAR'].

    Returns
    -------
    grd_df : pd.DataFrame
        DataFrame containing GRD fingerprints.
    """
    grd_df = pd.DataFrame(columns=COMPONENTS)
    for gauge in gauges:
        gauge_grd = read_gauge_grd(gauge=gauge)
        grd_df.loc[gauge] = gauge_grd
    return grd_df


@cache
def get_component_qf(workflow='wf_1e', component='EAIS', scenario='ssp585', year=2100, plot=False):
    """
    Return quantile function corresponding to a component of GMSLR.

    Parameters
    ----------
    workflow : str
        AR6 workflow (e.g. 'wf_1e', default), p-box bound ('lower', 'upper', 'outer'), or fusion (e.g. 'fusion_1e').
        Note, 'wf_2e' is unsupported.
    component : str
        Component of GMSLR. Options are 'EAIS' (East Antarctic Ice Sheet, default),
        'WAIS' (West Antarctic Ice Sheet), 'GrIS' (Greenland Ice Sheet), and 'GMSLR' (total GMSLR).
        Note, for wf_1e, 'PEN' (Antarctic peninsula) is also included in 'WAIS'.
    scenario : str
        Options are 'ssp126' and 'ssp585' (default).
    year : int
        Year. Default is 2100.
    plot : Bool
        Plot the result? Default is False.

    Returns
    -------
    qf_da : xarray DataArray
        DataArray of sea level quantiles in metres for different probability levels.

    Notes
    -----
    1. Following the AR6 projections, the quantile function will contain 20,000 samples.
    2. This function is based on https://github.com/grandey/d23a-fusion.
    """
    # Case 1: single workflow, corresponding to one of the alternative AR6 projections
    if workflow in ['wf_1e', 'wf_3e', 'wf_4']:
        # Read AR6 samples
        samples_da = read_ar6_samples(workflow=workflow, component=component, scenario=scenario, year=year)
        # Transform samples to quantile function
        qf_da = samples_da.sortby(samples_da)  # sort
        qf_da = qf_da.assign_coords(samples=np.linspace(0., 1., len(qf_da)))  # uniformly distributed probabilities
        qf_da = qf_da.rename({'samples': 'p'})  # rename coordinate to p (probability).
    # Case 2: lower or upper bound of low-confidence p-box
    elif workflow in ['lower', 'upper']:
        wf_list = ['wf_1e', 'wf_3e', 'wf_4']
        # Get quantile function data for each of these workflows
        qf_da_list = []  # list to hold quantile functions
        for wf in wf_list:
            qf_da_list.append(get_component_qf(workflow=wf, component=component, scenario=scenario, year=year))
        concat_da = xr.concat(qf_da_list, 'wf')  # concatenate the quantile functions along new dimension
        # Find lower or upper bound
        if workflow == 'lower':
            qf_da = concat_da.min(dim='wf')
        else:
            qf_da = concat_da.max(dim='wf')
    # Case 3: outer bound of p-box
    elif workflow == 'outer':
        # Get data for lower and upper bounds
        lower_da = get_component_qf(workflow='lower', component=component, scenario=scenario, year=year)
        upper_da = get_component_qf(workflow='upper', component=component, scenario=scenario, year=year)
        # Derive outer bound; note, although median is undefined, the qf with 20,000 samples does not contain p=0.5
        qf_da = xr.concat([lower_da.sel(p=slice(0, 0.5)),  # lower bound below median
                           upper_da.sel(p=slice(0.5000001, 1))],  # upper bound above median
                          dim='p')
    # Case 4: fusion distribution
    elif 'fusion' in workflow:
        # Get data for preferred workflow and outer bound of p-box
        wf = f'wf_{workflow.split("_")[-1]}'
        pref_da = get_component_qf(workflow=wf, component=component, scenario=scenario, year=year)
        outer_da = get_component_qf(workflow='outer', component=component, scenario=scenario, year=year)
        # Triangular weighting function, with weights depending on probability p
        w_da = get_fusion_weights()
        # Derive fusion distribution; rely on automatic broadcasting/alignment; no need to correct median (see Case 3)
        qf_da = w_da * pref_da + (1 - w_da) * outer_da
    else:
        raise ValueError(f'Unrecognised parameter value: workflow={workflow}')
    # Plot?
    if plot:
        if 'wf' in workflow:
            linestyle = ':'
        elif 'fusion' in workflow:
            linestyle = '-'
        else:
            linestyle = '--'
        qf_da.plot(y='p', label=workflow, alpha=0.5, linestyle=linestyle)
    # Return result
    return qf_da


@cache
def get_fusion_weights():
    """
    Return trapezoidal weighting function for fusion.

    Returns
    -------
    w_da : xarray DataArray
        DataArray of weights for preferred workflow, with weights depending on probability

    Notes
    -----
    This function follows https://github.com/grandey/d23a-fusion.
    """
    # Get a quantile function corresponding to a projection of total sea level, using default parameters
    w_da = get_component_qf(workflow='wf_1e', component='EAIS', scenario='ssp585', year=2100
                            ).copy()  # use as template for w_da, with data to be updated
    # Update data to follow trapezoidal weighting function, with weights depending on probability
    da1 = w_da.sel(p=slice(0, 0.1699999))
    da1[:] = da1.p / 0.17
    da2 = w_da.sel(p=slice(0.17, 0.83))
    da2[:] = 1.
    da3 = w_da.sel(p=slice(0.8300001, 1))
    da3[:] = (1 - da3.p) / 0.17
    w_da = xr.concat([da1, da2, da3], dim='p')
    # Rename
    w_da = w_da.rename('weights')
    return w_da


@cache
def get_ism_corr_df(ensemble='S20+P21+L23', ref_year=2015, target_year=2100):
    """
    Return DataFrame of EAIS−WAIS correlation (Pearson's r, Kendall's tau) and partial correlation (controlling for
    ESM/ISM) using ISM ensemble data.

    Parameters
    ----------
    ensemble : str
        Ensemble to include. Default is 'S20+P21+L23' (default).
    ref_year : int
        Reference year. Default is 2015 (which is the start year for the P21 data).
    target_year : int
        Target year for difference. Default is 2100.

    Returns
    -------
    ism_corr_df : pandas.DataFrame

    Note
    ----
    Partial correlation is estimated using a residual-based approximation.
    The effects of the controlling categorical variable (ESM or ISM) are first removed using linear regression.
    """
    # Get data for combined ISM ensemble
    ism_df = read_ism_ensemble_data(ensemble=ensemble, ref_year=ref_year, target_year=target_year,
                                    fully_crossed=True).dropna()
    # Create DataFrame to store correlation data
    ism_corr_df = pd.DataFrame(columns=["Description", "Control", "Pearson's r", "Kendall's 𝜏"])
    # Loop over rows (control)
    for i, control in enumerate([None, 'ISM', 'ESM']):
        # If no control, use EAIS and WAIS data
        if control is None:
            ism_corr_df.loc[i, 'Description'] = 'Correlation between EAIS and WAIS'
            ism_corr_df.loc[i, 'Control'] = 'None'
            eais = ism_df['EAIS']
            wais = ism_df['WAIS']
        # Control for control variable using linear regression and use residuals to estimate partial correlation
        else:
            eais = smf.ols(f'EAIS ~ C({control})', data=ism_df).fit().resid
            wais = smf.ols(f'WAIS ~ C({control})', data=ism_df).fit().resid
            if np.var(eais) < 1e-6 or np.var(wais) < 1e-6:
                print(f'Caution: when controlling {control}, residual variance is very small')
            if control == 'ISM':
                ism_corr_df.loc[i, 'Description'] = 'Partial correlation due to climate uncertainty'
                ism_corr_df.loc[i, 'Control'] = 'Ice sheet model configuration'
            elif control == 'ESM':
                ism_corr_df.loc[i, 'Description'] = 'Partial correlation due to process uncertainty'
                ism_corr_df.loc[i, 'Control'] = 'Earth system model'
        # Calculate correlation, using both Pearson's r and Kendall's tau
        r, _ = stats.pearsonr(eais, wais)
        ism_corr_df.loc[i, "Pearson's r"] = r.round(2)
        tau, _ = stats.kendalltau(eais, wais)
        ism_corr_df.loc[i, "Kendall's 𝜏"] = tau.round(2)
    return ism_corr_df


@cache
def quantify_bivariate_dependence(cop_workflow='wf_1e', year=2100, components=('EAIS', 'WAIS')):
    """
    Quantify dependence between two ice sheet components by fitting a bivariate copula, calculating Kendall's tau,
    and calculating Pearson's r using the SSP5-8.5 data for a given workflow/ensemble and year.

    Parameters
    ----------
    cop_workflow : str
        AR6 workflow (e.g. 'wf_1e', default), ice sheet model ensemble (e.g. 'P21+L23'), or idealized dependence
        (e.g. '1') to use.
    year : int
        Year. Default is 2100.
    components : tuple of str
        Two ice sheet components. Default is ('EAIS', 'WAIS').

    Returns
    -------
    bicop : pv.Bicop
        Fitted bivariate copula (limited to single-parameter families).
    tau : float
        Kendall's tau (calculated using the sample).
    r : float
        Pearson's r (calculated using the sample).
    """
    # Check that two and only two components have been specified
    if len(components) != 2:
        raise ValueError(f'Unrecognized argument value: components={components}. Length should be 2.')
    # Case 1: specify idealized dependence by specifying the copula
    if cop_workflow in ('0', '1', '10', '01'):
        if cop_workflow == '1':  # perfect dependence
            bicop = pv.Bicop(family=pv.BicopFamily.gaussian, parameters=np.array([[1.0]]))
            tau, r = 1., 1.
        elif cop_workflow == '10' and components == tuple(COMPONENTS[:2]):  # perfect dep. between 1st & 2nd components
            bicop = pv.Bicop(family=pv.BicopFamily.gaussian, parameters=np.array([[1.0]]))
            tau, r = 1., 1.
        elif cop_workflow == '01' and components == tuple(COMPONENTS[1:]):  # perfect dep. between 2nd & 3rd components
            bicop = pv.Bicop(family=pv.BicopFamily.gaussian, parameters=np.array([[1.0]]))
            tau, r = 1., 1.
        else:  # independence
            bicop = pv.Bicop(family=pv.BicopFamily.indep)
            tau, r = 0., 0.
    # Case 2: quantify dependence by fitting copula to samples
    else:
        # Read samples
        samples_list = []
        for component in components:
            if 'wf' in cop_workflow:  # if workflow, read samples DataArray and extract data
                samples = read_ar6_samples(workflow=cop_workflow, component=component, scenario='ssp585',
                                           year=year).data
            else:  # if ISM ensemble, read samples DataFrame and extract data
                samples = read_ism_ensemble_data(ensemble=cop_workflow, ref_year=2015,
                                                 target_year=year)[component].values
            samples_list.append(samples)
        # Fit copula (limited to single-parameter families)
        x_n2 = np.stack(samples_list, axis=1)
        u_n2 = pv.to_pseudo_obs(x_n2)
        controls = pv.FitControlsBicop(family_set=[pv.BicopFamily.indep, pv.BicopFamily.joe, pv.BicopFamily.gumbel,
                                                   pv.BicopFamily.gaussian, pv.BicopFamily.frank,
                                                   pv.BicopFamily.clayton])
        bicop = pv.Bicop.from_data(u_n2, controls=controls)  # fit
        # Calculate Kendall's tau and Pearson's r
        tau, _ = stats.kendalltau(samples_list[0], samples_list[1])
        r, _ = stats.pearsonr(samples_list[0], samples_list[1])
    # Return result
    return bicop, tau, r


@cache
def quantify_trivariate_dependence(cop_workflow='wf_1e'):
    """
    Quantify dependence between the three ice sheet components by fitting a vine copula to the year-2100 SSP5-8.5 data.

    Parameters
    ----------
    cop_workflow : str or tuple
        AR6 workflow (e.g. 'wf_1e', default) or ISM ensemble (e.g. 'P21+L23') to which to fit copula,
        or idealized case (e.g. '10', (pv.BicopFamily.gaussian, 0.5)).

    Returns
    -------
    tricop : pv.Vinecop
        Fitted vine copula (limited to single-parameter families).

    Notes
    -----
    The structure is specified according to the order of COMPONENTS.
    """
    # Case 1: idealized dependence by specifying a truncated vine copula
    if cop_workflow in ('1', '0', '10', '01') or type(cop_workflow) == tuple:
        if cop_workflow == '1':  # perfect dependence
            bicops = [pv.Bicop(family=pv.BicopFamily.gaussian, parameters=np.array([[1.0]])), ] * 2
        elif cop_workflow == '0':  # independence
            bicops = [pv.Bicop(family=pv.BicopFamily.indep), ] * 2
        elif cop_workflow == '10':  # perfect dep. between 1st & 2nd components
            bicops = [pv.Bicop(family=pv.BicopFamily.gaussian, parameters=np.array([[1.0]])),
                      pv.Bicop(family=pv.BicopFamily.indep)]
        elif cop_workflow == '01':  # perfect dep. between 2nd & 3rd components
            bicops = [pv.Bicop(family=pv.BicopFamily.indep),
                      pv.Bicop(family=pv.BicopFamily.gaussian, parameters=np.array([[1.0]]))]
        elif type(cop_workflow) == tuple:  # tuple of pair copula family and tau
            family, tau = cop_workflow
            parameters = pv.Bicop(family=family).tau_to_parameters(tau)
            bicop = pv.Bicop(family=family, parameters=parameters)
            bicops = [bicop, ] * 2
        structure = pv.DVineStructure(order=(1, 2, 3), trunc_lvl=1)
        tricop = pv.Vinecop.from_structure(structure=structure, pair_copulas=[bicops])
    # Case 2: quantify dependence by fitting vine copula to samples
    else:
        # Read samples
        samples_list = []
        for component in COMPONENTS:
            if 'wf' in cop_workflow:  # if workflow, read samples DataArray and extract data
                samples = read_ar6_samples(workflow=cop_workflow, component=component,
                                           scenario='ssp585', year=2100).data
            else:  # if ISM ensemble, read samples DataFrame and extract data
                samples = read_ism_ensemble_data(ensemble=cop_workflow,
                                                 ref_year=2015, target_year=2100)[component].values
            samples_list.append(samples)
        # Fit vine copula (limited to single-parameter families)
        x_n3 = np.stack(samples_list, axis=1)
        u_n3 = pv.to_pseudo_obs(x_n3)
        structure = pv.DVineStructure(order=(1, 2, 3))  # specify order, rather than selecting as part of fit
        controls = pv.FitControlsVinecop(family_set=[pv.BicopFamily.indep, pv.BicopFamily.joe, pv.BicopFamily.gumbel,
                                                     pv.BicopFamily.gaussian, pv.BicopFamily.frank,
                                                     pv.BicopFamily.clayton])
        tricop = pv.Vinecop.from_data(u_n3, structure=structure, controls=controls)  # fit
    # Return result
    return tricop


@cache
def sample_trivariate_copula(cop_workflow='wf_1e', n_samples=1000000, plot=False):
    """
    Sample a vine copula returned by quantify_trivariate_dependence().

    Parameters
    ----------
    cop_workflow : str or tuple
        AR6 workflow (e.g. 'wf_1e', default), ISM ensemble (e.g. 'P21+L23'),
        or idealized case (e.g. '10', (pv.BicopFamily.gaussian, 0.5)).
    n_samples : int
        Number of samples to generate. Default is 1,000,000.
    plot : bool
        Plot the simulated data? Default is False.

    Returns
    -------
    u_n3 : np.array
        An array of the simulated data, with shape (n_samples, 3).
    """
    # Get vine copula
    tricop = quantify_trivariate_dependence(cop_workflow=cop_workflow)
    # Simulate data
    print(f'sample_trivariate_copula({cop_workflow}, {n_samples}, {plot}): drawing {n_samples} samples')
    u_n3 = tricop.simulate(n=n_samples, seeds=[1, 2, 3, 4, 5], num_threads=4)
    # Plot?
    if plot:
        sns.pairplot(pd.DataFrame(u_n3, columns=[f'u{n+1}' for n in range(3)]), kind='hist')
        plt.suptitle(cop_workflow, y=1.15)
        plt.show()
    return u_n3


@cache
def sample_trivariate_distribution(cop_workflow='wf_1e',
                                   marg_workflow='fusion_1e', marg_scenario='ssp585', marg_year=2100,
                                   sample_repeats=50, plot=False):
    """
    Sample EAIS-WAIS-GrIS joint distribution.

    Parameters
    ----------
    cop_workflow : str or tuple
        AR6 workflow (e.g. 'wf_1e', default), ISM ensemble (e.g. 'P21+L23'),
        or idealized case (e.g. '10', (pv.BicopFamily.gaussian, 0.5)), corresponding to the vine copula.
    marg_workflow : str
        AR6 workflow (e.g. 'wf_1e'), p-box bound (e.g. 'outer'), or fusion (e.g. 'fusion_1e', default),
        corresponding to the component marginals.
    marg_scenario : str
        Scenario to use for the component marginals. Options are 'ssp126' and 'ssp585' (default).
    marg_year : int
        Year to use for the component marginals. Default is 2100.
    sample_repeats : int or None
        Number of times to duplicate the 20,000 AR6 samples. Default is 50 (corresponding to 1 million samples total).
    plot : bool
        Plot the joint distribution? Default is False.

    Returns
    -------
    trivariate_df : pd.DataFrame
        DataFrame containing array of shape (n_samples, 3), containing the samples from the joint distribution.

    Notes
    -----
    The number of samples (n_samples) is determined by the length of the marginal quantile functions.
    """
    # Sample marginals of EAIS, WAIS, GrIS components
    components = COMPONENTS
    marginals = []  # empty list to hold samples for the marginals
    for component in components:
        qf_da = get_component_qf(workflow=marg_workflow, component=component, scenario=marg_scenario, year=marg_year)
        marginals.append(qf_da.data)
    marg_n3 = np.stack(marginals, axis=1)  # marginal array with shape (n_samples, 3)
    if sample_repeats:  # duplicate samples to increase n_samples?
        marg_n3 = np.tile(marg_n3, (sample_repeats, 1))
    n_samples = marg_n3.shape[0]
    # Sample copula
    u_n3 = sample_trivariate_copula(cop_workflow=cop_workflow, n_samples=n_samples)
    # Transform marginals of copula
    x_n3 = np.transpose(np.asarray([np.quantile(marg_n3[:, i], u_n3[:, i]) for i in range(3)]))
    # Convert to DataFrame
    trivariate_df = pd.DataFrame(x_n3, columns=components)
    trivariate_df = trivariate_df.astype(np.float32)  # use single precision
    # Plot?
    if plot:
        sns.pairplot(trivariate_df, kind='hist')
        plt.show()
    return trivariate_df


def fig_component_marginals(marg_workflow='fusion_1e', marg_scenario='ssp585', marg_year=2100):
    """
    Plot figure showing marginals for the ice sheet components.

    Parameters
    ----------
    marg_workflow : str
        AR6 workflow (e.g. 'wf_1e'), p-box bound (e.g. 'outer'), or fusion (e.g. 'fusion_1e', default).
    marg_scenario : str
        Scenario. Options are 'ssp126' and 'ssp585' (default).
    marg_year : int
        Year. Default is 2100.

    Returns
    -------
    fig : Figure
    axs : array of Axes
    """
    # Create Figure and Axes
    n_axs = len(COMPONENTS)  # number of subplots = number of components
    fig, axs = plt.subplots(n_axs, 1, figsize=(5, 2.1*n_axs), sharex=True, sharey=True, tight_layout=True)
    # Loop over components and Axes
    for i, (component, ax) in enumerate(zip(COMPONENTS, axs)):
        # Get marginal quantile function containing marginal samples
        qf_da = get_component_qf(workflow=marg_workflow, component=component, scenario=marg_scenario, year=marg_year)
        # Plot KDE
        sns.kdeplot(qf_da, bw_adjust=0.3, color='b', fill=True, cut=0, ax=ax)  # limit to data limits
        # Plot 5th, 50th, and 95th percentiles
        y_pos = 13  # position of percentile whiskers is tuned for the default parameters
        ax.plot([qf_da.quantile(p) for p in (0.05, 0.95)], [y_pos, y_pos], color='g', marker='|')
        ax.plot([qf_da.quantile(0.5), ], [y_pos, ], color='g', marker='x')
        if i == (n_axs-1):  # label percentiles in final subplot
            for p in [0.05, 0.5, 0.95]:
                ax.text(qf_da.quantile(p), y_pos-0.4, f'{int(p*100)}th', ha='center', va='top', color='g', rotation=90)
        # Skewness and kurtosis
        ax.text(0.75, 6.5,  # position tuned for the default parameters
                f"Skewness = {stats.skew(qf_da):.1f}\n"
                f"Fisher's kurtosis = {stats.kurtosis(qf_da, fisher=True):.1f}",
                ha='right', va='center', fontsize='medium', bbox=dict(boxstyle='square,pad=0.5', fc='1', ec='0.85'))
        # Title etc
        ax.set_title(f'({chr(97+i)}) {component}')
    # x-axis label and limits
    axs[-1].set_xlabel(f'Ice sheet mass loss, m')
    axs[-1].set_xlim([-0.2, 0.8])
    return fig, axs


def fig_ism_ensemble(ensemble='S20+P21+L23', ref_year=2015, target_year=2100, hue='Ensemble', style='Ensemble'):
    """
    Plot figure showing combined ISM ensemble WAIS vs EAIS on (a) GMSLR scale and (b) copula scale.

    Parameters
    ----------
    ensemble : str
        Ensemble to read. Options are 'S20' (Seroussi et al.), 'P21' (Payne et al.), 'L23' (Li et al.), and
        combinations such as 'S20+P21+L23' (default).
    ref_year : int
        Reference year. Default is 2015 (which is the start year for Payne et al. data).
    target_year : int
        Target year for difference. Default is 2100.
    hue : str
        Categorisation to use for hue of points. Default is 'Ensemble'.
    style : str
        Categorisation to use for style of points. Default is 'Ensemble'.

    Returns
    -------
    fig : Figure
    axs : array of Axes
    """
    # Read Antarctic ISM ensemble data
    ism_df = read_ism_ensemble_data(ensemble=ensemble, ref_year=ref_year, target_year=target_year).copy()
    # Include number of samples in label (for legend), and impose mimimum number of samples if hue/style is ESM/ISM
    if 'ESM' in (hue, style):
        for esm in ism_df['ESM'].unique():
            temp_df = ism_df.loc[ism_df['ESM'] == esm]
            n_samples = len(temp_df)
            if n_samples >= 10:
                ism_df = ism_df.replace(esm, f'{esm} (n = {n_samples})')
            else:  # drop if fewer than 10
                ism_df = ism_df.loc[ism_df['ESM'] != esm]
    if 'ISM' in (hue, style):
        for ism in ism_df['ISM'].unique():
            temp_df = ism_df.loc[ism_df['ISM'] == ism]
            n_samples = len(temp_df)
            if n_samples >= 10:
                ism_df = ism_df.replace(ism, f'{ism} (n = {n_samples})')
            else:  # drop if fewer than 10
                ism_df = ism_df.loc[ism_df['ISM'] != ism]
    if 'Ensemble' in (hue, style):
        for ens in ism_df['Ensemble'].unique():
            temp_df = ism_df.loc[ism_df['Ensemble'] == ens]
            n_samples = len(temp_df)
            if ens == 'S20':
                ism_df = ism_df.replace(ens, f'Seroussi et al. (n = {n_samples})')
            elif ens == 'P21':
                ism_df = ism_df.replace(ens, f'Payne et al. (n = {n_samples})')
            elif ens == 'L23':
                ism_df = ism_df.replace(ens, f'Li et al. (n = {n_samples})')
    # Create Figure and Axes
    fig, axs = plt.subplots(1, 2, figsize=(8, 4), tight_layout=True)
    # (a) WAIS vs EAIS on GMSLR scale (ie sea-level equivalent)
    ax = axs[0]
    sns.scatterplot(ism_df, x='EAIS', y='WAIS', hue=hue, style=style, ax=ax)
    ax.legend(loc='upper left')
    ax.set_title(f'(a) Sea-level equivalent scale')
    ax.set_xlabel('EAIS, m')
    ax.set_ylabel('WAIS, m')
    ax.set_xlim(-0.1, 0.7)
    ax.set_xticks(np.arange(-0.1, 0.71, 0.1))
    ax.set_ylim(-0.1, 0.5)
    ax.set_yticks(np.arange(-0.1, 0.51, 0.1))
    # (b) Pseudo-copula data on copula scale
    ax = axs[1]
    x_n2 = np.stack([ism_df['EAIS'], ism_df['WAIS']], axis=1)
    u_n2 = pv.to_pseudo_obs(x_n2)
    u_df = pd.DataFrame({'EAIS': u_n2[:, 0], 'WAIS': u_n2[:, 1],
                         'Ensemble': ism_df['Ensemble'], 'ESM': ism_df['ESM'], 'ISM': ism_df['ISM']})
    sns.scatterplot(u_df, x='EAIS', y='WAIS', hue=hue, style=style, legend=False, ax=ax)
    ax.set_title(f'(b) Copula scale')
    ax.set_xlabel('EAIS, unitless')
    ax.set_ylabel('\nWAIS, unitless')
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])
    return fig, axs


def fig_illustrate_copula():
    """
    Plot figure illustrating (a) bivariate distribution, (b) bivariate copula, and (c) vine copula.

    Returns
    -------
    fig : Figure
    """
    # Create folder to hold temporary component plots
    temp_dir = Path('temp')
    temp_dir.mkdir(exist_ok=True)
    # Workflow to use for copula
    cop_workflow = 'wf_4'
    # Sample corresponding vine copula and trivariate distribution
    u_nm = sample_trivariate_copula(cop_workflow=cop_workflow, n_samples=20000)
    u_df = pd.DataFrame(u_nm, columns=COMPONENTS)
    x_df = sample_trivariate_distribution(cop_workflow=cop_workflow,
                                          marg_workflow='fusion_1e', marg_scenario='ssp585', marg_year=2100,
                                          sample_repeats=None)
    # 1st component plot: bivariate joint distribution (with marginals)
    sns.set_style('ticks')
    g = sns.jointplot(x_df, x=COMPONENTS[0], y=COMPONENTS[1], kind='kde', cmap='Greens', fill=True,
                      levels=7, cut=0, marginal_ticks=False, marginal_kws={'bw_adjust': 0.3, 'color': 'b'},
                      xlim=[-0.2, 0.3], ylim=[-0.2, 0.5], height=3)
    g.set_axis_labels(xlabel=f'{COMPONENTS[0]}, m', ylabel=f'{COMPONENTS[1]}, m')
    g.savefig(temp_dir / 'temp_joint.png')
    plt.close(g.fig)
    # 2nd component: bivariate copula (with uniform marginals)
    g = sns.jointplot(u_df, x=COMPONENTS[0], y=COMPONENTS[1], kind='kde', cmap='Greens', fill=True, bw_adjust=1.8,
                      levels=7, cut=0, clip=[0, 1], marginal_ticks=False, marginal_kws={'bw_adjust': 0.3, 'color': 'b'},
                      xlim=[0, 1], ylim=[0, 1], height=3)
    g.set_axis_labels(xlabel=f'{COMPONENTS[0]}, unitless', ylabel=f'{COMPONENTS[1]}, unitless')
    g.savefig(temp_dir / 'temp_copula1.png')
    plt.close(g.fig)
    # 3rd & 4th components: copulas without marginals
    sns.set_style('white')
    for i, levels in enumerate([7, 5]):
        fig, ax = plt.subplots(1, 1, figsize=(3, 3), tight_layout=True)
        sns.kdeplot(u_df, x=COMPONENTS[i], y=COMPONENTS[i+1], cmap='Greens', fill=True, bw_adjust=1.8,
                    levels=levels, cut=0, clip=[0, 1], ax=ax)
        ax.set_xlabel(None)
        ax.set_ylabel(None)
        ax.set_xlim([0, 1])
        ax.set_ylim([0, 1])
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        fig.savefig(temp_dir / f'temp_copula{i+2}.png')
        plt.close(fig)
    # 5th-7th components: marginals
    for component in COMPONENTS:
        fig, ax = plt.subplots(1, 1, figsize=(2, 1), tight_layout=True)
        sns.kdeplot(x_df, x=component, bw_adjust=0.5, color='b', fill=True, cut=0, ax=ax)
        ax.set_xlabel(None)
        ax.set_ylabel(None)
        ax.set_xlim([-0.2, 0.6])
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        fig.savefig(temp_dir / f'temp_{component}.png')
        plt.close(fig)
    # Combine component images into composite figure
    fig = plt.figure(figsize=(10, 7), tight_layout=True)
    gs = gridspec.GridSpec(2, 3, width_ratios=[4, 2, 4], height_ratios=[4, 3])
    # (a) Bivariate joint distribution
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.set_axis_off()
    ax1.imshow(plt.imread(temp_dir / 'temp_joint.png'))
    ax1.annotate('Marginal\ndensity', xy=(0.55, 0.90), xytext=(1.0, 0.92),
                 va='center', ha='center', xycoords='axes fraction', fontsize='x-large', color='b',
                 arrowprops=dict(arrowstyle='->', ec='b'))
    ax1.annotate('Joint\ndensity', xy=(0.45, 0.55), xytext=(0.37, 0.75),
                 va='center', ha='center', xycoords='axes fraction', fontsize='x-large', color='g',
                 arrowprops=dict(arrowstyle='->', ec='g'))
    ax1.set_title('(a) Bivariate distribution', fontsize='xx-large')
    # Probability integral transform arrow
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.set_axis_off()
    ax2.text(0.5, 0.65, 'Transform to the\ncopula scale',
             ha='center', va='center', fontsize='large', fontweight='bold',
             bbox=dict(boxstyle='rarrow,pad=0.5', fc='lavender', ec='purple'))
    ax2.text(0.5, 0.35, 'Transform to the\nsea-level scale',
             ha='center', va='center', fontsize='large', fontweight='bold',
             bbox=dict(boxstyle='larrow,pad=0.5', fc='lavender', ec='purple'))
    # (b) Bivariate copula
    ax3 = fig.add_subplot(gs[0, 2])
    ax3.set_axis_off()
    ax3.imshow(plt.imread(temp_dir / 'temp_copula1.png'))
    ax3.annotate('Uniform\nmarginal\ndensity', xy=(0.23, 0.92), xytext=(-0.03, 0.92),
                 va='center', ha='center', xycoords='axes fraction', fontsize='x-large', color='b',
                 arrowprops=dict(arrowstyle='->', ec='b'))
    ax3.annotate('Copula\ndensity', xy=(0.45, 0.60), xytext=(0.35, 0.75),
                 va='center', ha='center', xycoords='axes fraction', fontsize='x-large', color='g',
                 arrowprops=dict(arrowstyle='->', ec='g'))
    ax3.set_title('(b) Bivariate copula', fontsize='xx-large')
    # (c) Vine copula
    ax4 = fig.add_subplot(gs[1, :])
    ax4.set_axis_off()
    ax4.set_xlim(-5, 5)  # specify coordinate system for arrangement of images and boxes
    ax4.set_ylim(-0.5, 2)
    ax4.plot([-4, 4], [0, 0], color='g')  # line connecting text boxes
    for i, x, s in zip(range(3), [-4.3, 0, 4.3], COMPONENTS):  # marginals
        ax4.text(x, 0, s, ha='center', va='center', fontsize=25,
                 bbox=dict(boxstyle='square,pad=0.5', fc='azure', ec='blue'))
        ax4.imshow(plt.imread(temp_dir / f'temp_{s}.png'), extent=[x-0.8, x+0.8, 0.4, 1.2])
    for i, x, s in zip(range(2), [-2.15, 2.15],
                       [f'{COMPONENTS[0]}–{COMPONENTS[1]}', f'{COMPONENTS[1]}–{COMPONENTS[2]}']):  # pair copulas
        ax4.text(x, -0.1, f'{s}\npair copula', ha='center', va='top', fontsize=18, color='g')
        ax4.imshow(plt.imread(temp_dir / f'temp_copula{i+2}.png'), extent=[x-0.9, x+0.9, 0, 1.8])
    ax4.set_title('(c) Vine copula', fontsize='xx-large', y=0.95)
    # Reset seaborn style
    sns.set_style(SNS_STYLE)
    return fig


def fig_dependence_table(cop_workflows=('S20+P21+L23', 'S20+P21', 'wf_2e', 'wf_3e', 'wf_4', 'wf_1e', '0', '1', '10'),
                         all_pairs=True, print_tricop=True):
    """
    Plot heatmap table of bivariate copulas for AR6 workflows and ISM ensemble.

    Parameters
    ----------
    cop_workflows : tuple of str
        AR6 workflows (e.g. 'wf_1e'), ice sheet model ensemble (e.g. 'P21+L23'), and/or idealized dependence (e.g. '1').
        Default is ('S20+P21+L23', 'S20+P21', 'wf_2e', 'wf_3e', 'wf_4', 'wf_1e', '0', '1', '10').
    all_pairs : bool
        If True (default), include all pairs of dependencies.
    print_tricop : bool
        If True (default), print the corresponding trivariate vine copula, to check for consistency with table.

    Returns
    -------
    fig : Figure
    axs : array of Axes
    """
    # Component combinations correspond to columns
    columns = [f'{COMPONENTS[i]}–{COMPONENTS[i+1]}\n(tree 1)' for i in range(2)]
    if all_pairs:
        columns.append(f'{COMPONENTS[0]}–{COMPONENTS[2]}\n(not used)')
    columns.append(f'{COMPONENTS[0]}–{COMPONENTS[2]}|{COMPONENTS[1]}\n(tree 2)')
    # DataFrames to hold bivariate copula annotation string and Kendall's tau
    annot_df = pd.DataFrame(dtype=object)
    tau_df = pd.DataFrame(dtype=float)
    # Add data to DataFrames
    for workflow in cop_workflows:  # loop over workflows
        for column in columns:
            if f'{COMPONENTS[0]}–{COMPONENTS[2]}|{COMPONENTS[1]}' in column:
                tricop = quantify_trivariate_dependence(cop_workflow=workflow)
                if print_tricop:
                    print(f'{WORKFLOW_LABELS[workflow]}:\n{tricop.format()}\n')
                try:
                    bicop = tricop.pair_copulas[1][0]  # pair copula in 2nd tree of fitted vine copula
                except IndexError:  # if truncated vine copula, there will be no pair copula in the 2nd tree
                    bicop = pv.Bicop(family=pv.BicopFamily.indep)
            else:
                components = tuple(column.split('\n')[0].split('–'))
                bicop = quantify_bivariate_dependence(cop_workflow=workflow, year=2100, components=components)[0]
            column_formatted = '$\\bf{'+column.split('\n')[0]+'}$\n'+column.split('\n')[1]  # make first part bold
            bicop_json = json.loads(bicop.to_json())  # get name etc for annotation string
            if bicop.rotation == 0:
                annot_str = f'{bicop_json["fam"]},\n{TAU_BOLD} = {bicop.tau:.2f}'
            else:
                annot_str = f'{bicop_json["fam"]} {bicop.rotation}°,\n{TAU_BOLD} = {bicop.tau:.2f}'
            annot_df.loc[workflow, column_formatted] = annot_str
            tau_df.loc[workflow, column_formatted] = bicop.tau
    # Create Figure and Axes
    if all_pairs:
        width = 11
    else:
        width = 9.5
    fig, ax = plt.subplots(1, 1, figsize=(width, 0.8*len(cop_workflows)), constrained_layout=True)
    # Plot heatmap
    sns.heatmap(tau_df, cmap='seismic', vmin=-1., vmax=1., annot=annot_df, fmt='',
                annot_kws={'weight': 'bold', 'size': 'large'}, linecolor='lightgrey', linewidths=1, ax=ax)
    # Customise plot
    ax.tick_params(top=False, bottom=False, left=False, right=False,
                   labeltop=True, labelbottom=False, labelleft=True, labelright=False, rotation=0)
    try:
        ax.set_yticklabels([WORKFLOW_NOTES[workflow] for workflow in cop_workflows], ha='center')
        ax.tick_params(axis='y', pad=95)
    except KeyError:
        pass
    cbar = ax.collections[0].colorbar
    cbar.set_ticks([-1., 0., 1.])
    cbar.set_label(f'Kendall\'s {TAU_BOLD}', size='large')
    return fig, ax


def fig_corr_vs_time():
    """
    Plot (a) Pearson's r and (b) Kendall's τ vs time for the ISM ensembles and IPCC AR6 workflows.

    Returns
    -------
    fig : Figure
    axs : array of Axes
    """
    # Create Figure and Axes
    fig, axs = plt.subplots(1, 2, figsize=(10, 4), tight_layout=True)
    # Loop over ensembles and AR6 workflows
    for cop_workflow in ['S20+P21+L23', 'S20+P21', 'L23', 'wf_1e', 'wf_2e', 'wf_3e', 'wf_4']:
        # Create Series to hold Pearson's r and Kendall's tau for different years
        r_ser = pd.Series()
        tau_ser = pd.Series()
        # Loop over years
        for year in np.arange(2050, 2101, 10):
            _, tau, r = quantify_bivariate_dependence(cop_workflow=cop_workflow, year=year)
            r_ser[year] = r
            tau_ser[year] = tau
        # Plot Pearson's r and Kendall's tau vs time
        if cop_workflow[0:3] == 'wf_':
            linestyle = '--'
            alpha = 0.3
        elif cop_workflow == 'S20+P21+L23':
            linestyle = '-'
            alpha = 1.
        else:
            linestyle = '-.'
            alpha = 1.
        axs[0].plot(r_ser, linestyle=linestyle, alpha=alpha, color=WORKFLOW_COLORS[cop_workflow],
                    label=WORKFLOW_LABELS[cop_workflow].rstrip(' corr.'))
        axs[1].plot(tau_ser, linestyle=linestyle, alpha=alpha, color=WORKFLOW_COLORS[cop_workflow],
                    label=WORKFLOW_LABELS[cop_workflow].rstrip(' corr.'))
    # Customise plot
    axs[0].legend()
    for ax in axs:
        ax.set_xlabel('Year')
        ax.set_xlim([2050, 2100])
        ax.set_ylim([-0.1, 1])
    axs[0].set_ylabel('Pearson\'s r')
    axs[0].set_title('(a) Pearson\'s r')
    axs[1].set_ylabel(f'Kendall\'s {TAU_BOLD}')
    axs[1].set_title(f'(b) Kendall\'s {TAU_BOLD}')
    return fig, axs


def ax_total_vs_tau(families=(pv.BicopFamily.joe, pv.BicopFamily.clayton), colors=('darkred', 'blue'),
                    marg_workflow='fusion_1e', marg_scenario='ssp585', marg_year=2100,
                    ax=None):
    """
    Plot median and 5th-95th percentile range of total ice sheet mass loss (y-axis) vs Kendall's tau (x-axis).

    Parameters
    ----------
    families : tuple
        Pair copula families. Default is (pv.BicopFamily.joe, pv.BicopFamily.clayton).
    colors : tuple
        Colors to use when plotting. Default is ('darkred', 'blue').
    marg_workflow : str
        AR6 workflow (e.g. 'wf_1e'), p-box bound (e.g. 'outer'), or fusion (e.g. 'fusion_1e', default),
        corresponding to the component marginals.
    marg_scenario : str
        Scenario to use for the component marginals. Options are 'ssp126' and 'ssp585' (default).
    marg_year : int
        Year to use for the component marginals. Default is 2100.
    ax : Axes.
        Axes on which to plot. If None, new Axes are created. Default is None.

    Returns
    -------
    ax : Axes
        Axes on which data have been plotted.
    """
    # Create axes?
    if ax is None:
        fig, ax = plt.subplots(1, 1)
    # For each copula, calculate total ice sheet mass loss for different tau values and plot median & 5th-95th
    tau_t = np.linspace(0, 1, 21)  # tau values to use (every 0.05)
    p95_t_list = []  # list to hold 95th percentile arrays
    for family, color, hatch, linestyle, linewidth in zip(families, colors, ('//', r'\\'), ('--', '-.'), (3, 2)):
        label = family.name.capitalize()
        p50_t = np.full(len(tau_t), np.nan)  # array to hold median at each tau
        p5_t = np.full(len(tau_t), np.nan)  # 5th percentile
        p95_t = np.full(len(tau_t), np.nan)  # 95th percentile
        for t, tau in enumerate(tau_t):  # for each tau, calculate total ice sheet mass loss
            trivariate_df = sample_trivariate_distribution(cop_workflow=(family, tau),
                                                           marg_workflow=marg_workflow, marg_scenario=marg_scenario,
                                                           marg_year=marg_year)
            sum_ser = trivariate_df.sum(axis=1)
            p50_t[t] = np.percentile(sum_ser, 50)  # median
            p5_t[t] = np.percentile(sum_ser, 5)  # 5th percentile
            p95_t[t] = np.percentile(sum_ser, 95)  # 95th percentile
        # Plot data for this family
        ax.fill_between(tau_t, p5_t, p95_t, color=color, alpha=0.2, label=f'{label} (5th–95th)', hatch=hatch)
        ax.plot(tau_t, p50_t, color=color, label=f'{label} (median)', linestyle=linestyle, linewidth=linewidth)
        # Save 95th percentile data to list (used below)
        p95_t_list.append(p95_t)
    # Annotate with diffs and percentage diffs at 95th percentile
    if len(families) == 1:  # if plotting a single family, use the diff between tau = 0 and tau = 1
        p95_min = p95_t[0]
        p95_max = p95_t[-1]
        for p95 in [p95_min, p95_max]:
            ax.axhline(p95, alpha=0.3, color='k', linestyle=':')
    else:  # if plotting two or more families, use the diff at tau = 0.5 (middle index of 10)
        p95_min = min([p95_t[10] for p95_t in p95_t_list])
        p95_max = max([p95_t[10] for p95_t in p95_t_list])
    p95_diff = p95_max - p95_min  # difference
    p95_perc = 100. * p95_diff / p95_min  # percentage difference
    ax.arrow(0.5, p95_min, 0., p95_diff,  # plot arrow
             color='k', head_width=0.02, head_length=0.06, length_includes_head=True)
    diff_str = f'{p95_diff:+.1f} m'  # annotate with absolute diff
    ax.text(0.48, np.mean([p95_min, p95_max]), diff_str, va='center', ha='right', fontsize='large')
    perc_str = f'{p95_perc:+.0f} %'  # annotate with percentage diff
    ax.text(0.52, np.mean([p95_min, p95_max]), perc_str, va='center', ha='left', fontsize='large')
    # Customize plot
    ax.legend(loc='upper left', fontsize='large')
    ax.set_xlim(tau_t[0], tau_t[-1])
    ax.set_xlabel(f"Kendall's {TAU_BOLD}")
    ax.set_ylabel(f'Total ice sheet mass loss, m')
    ax.yaxis.set_minor_locator(plt.MultipleLocator(0.1))
    ax.xaxis.set_minor_locator(plt.FixedLocator(tau_t))
    ax.tick_params(which='minor', direction='in', color='0.7', bottom=True, top=True, left=True, right=True)
    ax.set_title(f'{marg_workflow} {marg_scenario} {marg_year}')
    return ax


def fig_total_vs_tau(families_a=(pv.BicopFamily.gaussian, ), families_b=(pv.BicopFamily.joe, pv.BicopFamily.clayton),
                     colors_a=('green', ), colors_b=('darkred', 'blue'),
                     marg_workflow='fusion_1e', marg_scenario='ssp585', marg_year=2100, ylim=(-0.2, 2.7)):
    """
    Plot figure showing median and 5th-95th percentile range of total ice sheet mass loss (y-axis) vs tau (x-axis)
    for (a) Gaussian pair copulas and (b) Joe & Clayton pair copulas (default).

    Parameters
    ----------
    families_a and families_b : tuple
        Pair copula families to use for panels (a) and (b).
        Default is (pv.BicopFamily.gaussian, ) and (pv.BicopFamily.joe, pv.BicopFamily.clayton).
    colors_a and colors_b : tuple
        Colors to use when plotting. Default is ('green', ) and ('darkred', 'blue').
    marg_workflow : str
        AR6 workflow (e.g. 'wf_1e'), p-box bound (e.g. 'outer'), or fusion (e.g. 'fusion_1e', default),
        corresponding to the component marginals.
    marg_scenario : str
        Scenario to use for the component marginals. Options are 'ssp126' and 'ssp585' (default).
    marg_year : int
        Year to use for the component marginals. Default is 2100.
    ylim : tuple
        Limits for y-axis. Default is (-0.2, 2.7).

    Returns
    -------
    fig : Figure
    axs : array of Axes
    """
    # Create figure and axes
    fig, axs = plt.subplots(1, 2, figsize=(10, 4), sharey=True, constrained_layout=True)
    # (a)
    ax = axs[0]
    _ = ax_total_vs_tau(families=families_a, colors=colors_a,
                        marg_workflow=marg_workflow, marg_scenario=marg_scenario, marg_year=marg_year, ax=ax)
    ax.set_title(f'(a) {" & ".join(f.name.capitalize() for f in families_a)} pair copulas')
    # (b)
    ax = axs[1]
    _ = ax_total_vs_tau(families=families_b, colors=colors_b,
                        marg_workflow=marg_workflow, marg_scenario=marg_scenario, marg_year=marg_year, ax=ax)
    ax.set_title(f'(b) {" & ".join(f.name.capitalize() for f in families_b)} pair copulas')
    ax.set_ylabel(None)
    ax.set_ylim(ylim)
    return fig, axs


def ax_total_vs_time(cop_workflows=('wf_3e', '0'),
                     marg_workflow='fusion_1e', marg_scenario='ssp585', marg_years=np.arange(2020, 2101, 10),
                     show_percent_diff=True, thresh_for_timing_diff=(1.4, 0.2), ax=None):
    """
    Plot median and 5th-95th percentile range of total ice sheet mass loss (y-axis) vs time (x-axis).

    Parameters
    ----------
    cop_workflows : tuple of str
        AR6 workflows (e.g. 'wf_1e'), ice sheet model ensemble (e.g. 'P21+L23'), and/or idealized dependence (e.g. '1').
        Default is ('wf_3e', '0').
    marg_workflow : str
        AR6 workflow (e.g. 'wf_1e'), p-box bound ('lower', 'upper', 'outer'), or fusion (e.g. 'fusion_1e', default),
        corresponding to the component marginals.
    marg_scenario : str
        The scenario for the component marginals. Default is 'ssp585'.
    marg_years : np.array
        Target years for the component marginals. Default is np.arange(2020, 2101, 10).
    show_percent_diff : bool
        Show percentage difference in 95th percentile and median? Default is True.
    thresh_for_timing_diff : tuple, True, or None
        Thresholds to use if demonstrating the difference in timing at the 95th percentile and median.
        If True, select automatically. Default is (1.4, 0.2).
    ax : Axes.
        Axes on which to plot. If None, new Axes are created. Default is None.

    Returns
    -------
    ax : Axes
        Axes on which data have been plotted.
    """
    # Create axes?
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(5, 4))
    # List to hold DataFrames created below
    data_dfs = []
    # For each copula, calculate total ice sheet mass loss for different years and plot
    for cop_workflow, hatch, linestyle, linewidth in zip(cop_workflows, ('//', '..'), ('--', '-.'), (3, 2)):
        # Create DataFrame to hold percentile time series for this copula
        data_df = pd.DataFrame()
        # For each year, calculate percentiles of total ice sheet mass loss
        for year in marg_years:
            trivariate_df = sample_trivariate_distribution(cop_workflow=cop_workflow,
                                                           marg_workflow=marg_workflow, marg_scenario=marg_scenario,
                                                           marg_year=year)
            sum_ser = trivariate_df.sum(axis=1)
            for perc in (5, 50, 95):
                data_df.loc[year, perc] = np.percentile(sum_ser, perc)
        # Plot
        label = WORKFLOW_LABELS[cop_workflow]
        color = WORKFLOW_COLORS[cop_workflow]
        ax.fill_between(data_df.index, data_df[5], data_df[95], label=f'{label} (5th–95th)',  # plot
                        color=color, alpha=0.2, hatch=hatch)
        sns.lineplot(data_df[50], color=color, label=f'{label} (median)',
                     linestyle=linestyle, linewidth=linewidth, ax=ax)
        # Save percentile time series to list of DataFrames
        data_dfs.append(data_df)
    # Show percentage difference in 95th percentile and median?
    if show_percent_diff and len(data_dfs) > 1:
        year = marg_years[-1]  # final year
        for perc in (95, 50):  # loop over 95th percentile and median
            val0 = data_dfs[0].loc[year, perc]  # percentile value in final year
            val1 = data_dfs[1].loc[year, perc]
            diff = val0 - val1  # difference, using val1 as the reference
            percent_diff = 100. * diff / val1  # percentage difference
            ax.arrow(year+1, val1, 0., diff,  # plot arrow showing diff
                     color='k', head_width=1.5, head_length=0.02, length_includes_head=True, clip_on=False)
            if abs(percent_diff) > 0.95:  # format percentage diff as string, to nearest percent if >~1%
                percent_str = f'{percent_diff:+.0f} %'
            else:
                percent_str = f'{percent_diff:+.1f} %'
            ax.text(year+2.5, np.mean([val1, val0]), percent_str,  # annotate with percentage diff
                    color='k', va='center', ha='left', fontsize='large')
            print(f'{cop_workflows}, {perc}th: {val0:.2f} - {val1:.2f} = {diff:.2f} m ({percent_str})')  # print values
    # Plot lines showing timing differences?
    if thresh_for_timing_diff:
        # Select thresholds automatically?
        if thresh_for_timing_diff is True:
            thresh_for_timing_diff = [min(max(data_dfs[0][perc]), max(data_dfs[1][perc])) for perc in (95, 50)]
        # Loop over thresholds
        for thresh, perc in zip(thresh_for_timing_diff, (95, 50)):
            # Find year at which percentile is closest to threshold for first two copulas
            year_eq_threshs = []  # list to hold years closest to threshold
            for data_df in data_dfs[0:2]:
                interp_years = np.arange(marg_years[0], marg_years[-1]+1, 1)  # interpolate years
                interp_perc = np.interp(interp_years, data_df.index, data_df[perc])  # interpolate data at percentile
                idx = np.abs(interp_perc - thresh).argmin()  # index closest to threshold
                year_eq_threshs.append(interp_years[idx])  # year closest to threshold
            # Calculate timing difference
            timing_diff = year_eq_threshs[0] - year_eq_threshs[1]
            if timing_diff != 0:
                # Plot arrow and text showing timing difference
                ax.arrow(year_eq_threshs[1], thresh, timing_diff, 0,
                         color='k', head_width=0.05, head_length=0.7, length_includes_head=True, zorder=3)
                text_str = f'{timing_diff:+.0f} yr'
                ax.text(np.mean(year_eq_threshs), thresh-0.05, text_str,
                        ha='center', va='top', fontsize='large')
                # Plot line showing threshold
                ax.axhline(thresh, alpha=0.3, color='k', linestyle=':')
    # Customize plot
    ax.set_xlim(marg_years[0], marg_years[-1])
    ax.set_xlabel('Year')
    ax.set_ylabel(f'Total ice sheet mass loss, m')
    ax.yaxis.set_minor_locator(plt.MultipleLocator(0.1))
    ax.xaxis.set_minor_locator(plt.MultipleLocator(1))
    ax.tick_params(which='minor', direction='in', color='0.7', bottom=True, top=True, left=True, right=True)
    ax.legend(loc='upper left', fontsize='large')
    return ax


def fig_total_vs_time(cop_workflows=('1', '10', 'S20+P21+L23', 'S20+P21'),
                      ref_workflows=('0', '0', '0', '0'),
                      marg_workflow='fusion_1e', marg_scenario='ssp585', marg_years=np.arange(2020, 2101, 10),
                      thresh_for_timing_diff=None, ylim=(-0.2, 2.0)):
    """
    Plot figure showing median and 5th-95th percentile range of total ice sheet mass loss (y-axis) vs time (x-axis)
    for different copulas (in multiple panels).

    Parameters
    ----------
    cop_workflows : tuple of str
        AR6 workflows (e.g. 'wf_1e'), ice sheet model ensemble (e.g. 'P21+L23'), and/or idealized dependence (e.g. '1').
        Note, these will be plotted in separate panels.
        Default is ('1', '10', 'S20+P21+L23', 'S20+P21').
    ref_workflows : tuple of str
        Workflows corresponding to the vine copulas to be used as the reference in each panel.
        Default is ('0', '0', '0', '0').
    marg_workflow : str
        AR6 workflow (e.g. 'wf_1e'), p-box bound ('lower', 'upper', 'outer'), or fusion (e.g. 'fusion_1e', default),
        corresponding to the component marginals.
    marg_scenario : str
        The scenario for the component marginals. Default is 'ssp585'.
    marg_years : np.array
        Target years for the component marginals. Default is np.arange(2020, 2101, 10).
    thresh_for_timing_diff : tuple, True, or None
        Thresholds to use if demonstrating the difference in timing at the 95th percentile and median.
        If True, select automatically. Default is None.
    ylim : tuple
        Limits for y-axis. Default is (-0.2, 2.0).

    Returns
    -------
    fig : Figure
    axs : array of Axes
    """
    # Create figure and axes
    if len(cop_workflows) == 1:
        ncols = 1
    else:
        ncols = 2
    nrows = math.ceil(len(cop_workflows) / 2)
    fig, axs = plt.subplots(nrows, ncols, figsize=(5*ncols, (3*nrows + 1)), sharey=True, sharex=True, tight_layout=True)
    # Flatten axs
    try:
        axs_flat = axs.flatten()
    except AttributeError:  # if only one panel
        axs_flat = [axs, ]
    # Plot panels
    for i, (cop_workflow, ref_workflow, ax) in enumerate(zip(cop_workflows, ref_workflows, axs_flat)):
        _ = ax_total_vs_time(cop_workflows=(cop_workflow, ref_workflow),
                             marg_workflow=marg_workflow, marg_scenario=marg_scenario, marg_years=marg_years,
                             show_percent_diff=True, thresh_for_timing_diff=thresh_for_timing_diff, ax=ax)
        if len(cop_workflows) == 1:
            ax.set_title(f'{WORKFLOW_LABELS[cop_workflow]} & {WORKFLOW_LABELS[ref_workflow]}')
        else:
            ax.set_title(f'({chr(97+i)}) {WORKFLOW_LABELS[cop_workflow].replace("corr.", "correlation")}')
        ax.set_ylim(ylim)
    return fig, axs


def ax_sum_vs_gris_fingerprint(cop_workflows=('1', '0'),
                               marg_workflow='fusion_1e', marg_scenario='ssp585', marg_year=2100,
                               ax=None):
    """
    Plot median and 5th-95th percentile range of total ice sheet mass loss (y-axis) vs GrIS GRD fingerprint (x-axis).

    Parameters
    ----------
    cop_workflows : tuple of str
        AR6 workflows (e.g. 'wf_1e'), ice sheet model ensemble (e.g. 'P21+L23'), and/or idealized dependence (e.g. '1').
        Default is ('1', '0').
    marg_workflow : str
        AR6 workflow (e.g. 'wf_1e'), p-box bound (e.g. 'outer'), or fusion (e.g. 'fusion_1e', default),
        corresponding to the component marginals.
    marg_scenario : str
        Scenario to use for the component marginals. Options are 'ssp126' and 'ssp585' (default).
    marg_year : int
        Year to use for the component marginals. Default is 2100.
    ax : Axes.
        Axes on which to plot. If None, new Axes are created. Default is None.

    Returns
    -------
    ax : Axes
        Axes on which data have been plotted.
    """
    # Create axes?
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(6, 4.7), constrained_layout=True)
    # GRD fingerprints to use
    eais_fp = 1.10
    wais_fp = 1.15
    gris_fp_g = np.arange(-1.8, 1.21, 0.05)  # _g indicates GrIS fingerprint dimension
    # For each copula, calculate total ice sheet mass loss for different GrIS fingerprints and plot median & 5th-95th
    for cop_workflow, hatch, linestyle, linewidth in zip(cop_workflows, ('//', '..'), ('--', '-.'), (3, 2)):
        # Get trivariate distribution data for global mean (ie fingerprints all 1.0)
        x_df = sample_trivariate_distribution(cop_workflow=cop_workflow,
                                              marg_workflow=marg_workflow, marg_scenario=marg_scenario,
                                              marg_year=marg_year)
        # Create DataFrame to hold percentile data across GrIS fingerprints
        data_df = pd.DataFrame()
        # Loop over GrIS fingerprints and calculate 5th, 50th, and 95th percentiles of total ice sheet mass loss
        for gris_fp in gris_fp_g:
            sum_ser = eais_fp * x_df['EAIS'] + wais_fp * x_df['WAIS'] + gris_fp * x_df['GrIS']
            for perc in (5, 50, 95):
                data_df.loc[gris_fp, perc] = np.percentile(sum_ser, perc)
        # Plot data for this copula
        label = WORKFLOW_LABELS[cop_workflow]
        color = WORKFLOW_COLORS[cop_workflow]
        ax.fill_between(data_df.index, data_df[5], data_df[95], color=color, alpha=0.2, hatch=hatch,
                        label=f'{label} (5th–95th)')
        sns.lineplot(data_df[50], color=color, label=f'{label} (median)', linestyle=linestyle, linewidth=linewidth,
                     ax=ax)
    # Customize plot
    ax.legend(loc='upper left', framealpha=1, fontsize='large')
    ax.set_xlim(gris_fp_g[0], gris_fp_g[-1])
    ax.set_xlabel('Fingerprint of GrIS')
    ax.set_ylabel(f'Total ice sheet mass loss, m')
    ax.yaxis.set_minor_locator(plt.MultipleLocator(0.1))
    ax.xaxis.set_minor_locator(plt.FixedLocator(gris_fp_g))
    ax.tick_params(which='minor', direction='in', color='0.7', bottom=True, top=True, left=True, right=True)
    # Annotations
    ax.text(1, -0.15, f'Fingerprint of EAIS = {eais_fp:.2f}\nFingerprint of WAIS = {wais_fp:.2f}',
            transform=ax.transAxes, ha='right', va='bottom')
    ax.set_ylim(ax.get_ylim())  # fix y-axis limits before plotting points near limit
    for gauge, city in [('REYKJAVIK', 'Reykjavik'), ('DUBLIN', 'Dublin'), ('TANJONG_PAGAR', 'Singapore')]:
        gris_fp = read_gauge_grd(gauge=gauge)['GrIS']
        plt.axvline(gris_fp, color='darkgreen', linestyle='--', alpha=0.5)
        ax.text(gris_fp, ax.get_ylim()[0]+0.05, city, color='darkgreen', fontsize='large',
                ha='right', va='bottom', rotation=90)
    return ax


def name_save_fig(fig, fso='o', exts=('pdf', 'png'), close=False):
    """
    Name & save a figure, then increase counter.

    Parameters
    ----------
    fig : Figure
        Figure to save.
    fso : str
        Figure type. Either 'f' (main), 's' (supplement), or 'o' (other; default).
    exts : tuple
        Extensions to use. Default is ('pdf', 'png').
    close : bool
        Suppress output in notebook? Default is False.

    Returns
    -------
    fig_name : str
        Name of figure.

    Notes
    -----
    This function follows https://github.com/grandey/d22a-mcdc & https://github.com/grandey/d23a-fusion.
    """
    # Name based on counter, then update counter (in preparation for next figure)
    if fso == 'f':
        fig_name = f'fig{next(F_NUM):02}'
    elif fso == 's':
        fig_name = f's{next(S_NUM):02}'
    else:
        fig_name = f'o{next(O_NUM):02}'
    # File location based on extension(s)
    for ext in exts:
        # Sub-directory
        sub_dir = FIG_DIR.joinpath(f'{fso}_{ext}')
        sub_dir.mkdir(exist_ok=True)
        # Save
        fig_path = sub_dir.joinpath(f'{fig_name}.{ext}')
        fig.savefig(fig_path)
        # Print file name and size
        fig_size = fig_path.stat().st_size / 1024 / 1024  # bytes -> MB
        print(f'Written {fig_name}.{ext} ({fig_size:.2f} MB)')
    # Suppress output in notebook?
    if close:
        plt.close()
    return fig_name
