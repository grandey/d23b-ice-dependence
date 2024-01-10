"""
d23b:
    Functions that support the analysis contained in the d23b-ice-dependence repository.

Author:
    Benjamin S. Grandey, 2022-2023.
"""


from functools import cache
import itertools
import math
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path
import pyvinecopulib as pv
from scipy import stats
import seaborn as sns
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
COMPONENTS = ['EAIS', 'WAIS', 'GrIS']  # ice-sheet components of sea level, ordered according to vine copula
WORKFLOW_LABELS = {'wf_1e': 'Workflow 1e',  # names of "workflows", inc. ISM ensemble, fusion, idealized dependence
                   'wf_3e': 'Workflow 3e',
                   'wf_4': 'Workflow 4',
                   'P21+L23': 'P21+L23 ensemble',
                   'fusion_1e': 'Fusion',  # fusion used only for component marginals
                   '0': 'Independence',  # idealized indepedence used only when coupling with copulas
                   '1': 'Perfect correlation',  # idealized perfect dependence used only when coupling with copulas
                   '10': f'{COMPONENTS[0]}—{COMPONENTS[1]} perfect corr.',  # perfect dependence & independence
                   '01': f'{COMPONENTS[1]}—{COMPONENTS[2]} perfect corr.',  # independence & perfect dependence
}
WORKFLOW_NOTES = {'wf_1e': 'Shared dependence\non GMST\n(Edwards et al., 2021)',  # notes used by fig_dependence_table()
                  'wf_3e': 'Antarctic ISM\nensemble\n(DeConto et al., 2021)',
                  'P21+L23': 'Antarctic ISM\nensemble\n(Payne et al., 2021;\nLi et al., 2023)',
                  'wf_4': 'Structured\nexpert judgment\n(Bamber et al., 2019)',
                  }
WORKFLOW_COLORS = {'wf_1e': 'darkblue',  # colors used by ax_total_vs_time(), ax_sum_vs_gris_fingerprint()
                   'wf_3e': 'darkred',
                   'wf_4': 'darkgreen',
                   'P21+L23': 'purple',
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
        AR6 workflow.  Options are 'wf_1e' (default), 'wf_3e', or 'wf_4'.
    component : str
        Component of GMSLR. Options are 'EAIS' (East Antarctic Ice Sheet, default),
        'WAIS' (West Antarctic Ice Sheet), 'GrIS' (Greenland Ice Sheet), and 'GMSLR' (total GMSLR).
        Note: for ISMIP6, 'PEN' (Antarctic peninsula) is also included in 'WAIS'.
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
        if workflow in ['wf_1e', 'wf_3e']:
            gris_source = 'ipccar6-ismipemu'
        elif workflow == 'wf_4':
            gris_source = 'ipccar6-bamber'
        in_fn = in_dir / f'icesheets-{gris_source}icesheet-{scenario}_GIS_globalsl.nc'
    elif component in ['EAIS', 'WAIS', 'PEN']:  # Antarctica
        in_dir = IN_BASE / 'ar6' / 'global' / 'full_sample_components'
        if workflow == 'wf_1e':
            ais_source = 'ipccar6-ismipemu'
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
    if workflow == 'wf_1e' and component == 'WAIS':
        samples_da += read_ar6_samples(workflow=workflow, component='PEN', scenario=scenario, year=year)
        print(f'read_ar6_samples({workflow}, {component}, {scenario}, {year}): including PEN in WAIS')
    # Return result (without sorting/ordering)
    return samples_da


@cache
def read_ism_ensemble_data(ensemble='P21+L23', ref_year=2015, target_year=2100):
    """
    Read Antarctic ISM ensemble data from Payne et al. (2021) and Li et al. (2023).

    This function uses data from https://doi.org/10.5281/zenodo.4498331 and https://doi.org/10.5281/zenodo.7380180.

    Parameters
    ----------
    ensemble : str
        Ensemble to read. Options are 'P21' (Payne et al.) and 'P21+L23' (P21 and a subset of Li et al., default).
    ref_year : int
        Reference year. Default is 2015 (which is the start year for the P21 data).
    target_year : int
        Target year for difference. Default is 2100.

    Returns
    -------
    ism_df : pandas DataFrame
        A DataFrame containing WAIS and EAIS sea-level equivalents (in m), Group (P21 or L23), and Notes.
    """
    # DataFrame to hold data
    ism_df = pd.DataFrame(columns=['WAIS', 'EAIS', 'Group', 'Notes'])
    # Read Payne et al. data
    if 'P21' in ensemble:
        # Location of data
        in_dir = IN_BASE / 'CMIP5_CMIP6_Scalars_Paper' / 'AIS' / 'Ice'
        # Conversion factor for ice sheet mass above floatation (Gt) to sea-level equivalent (m)
        convert_Gt_m = 1. / 362.5 / 1e3  # Goelzer et al (2020): 362.5 Gt ~ 1 mm SLE
        # Experiments of interest (all SSP5-8.5; see https://doi.org/10.5281/zenodo.4498331 README.txt)
        exp_list = ['B1',  # CNRM-CM6-1 SSP5-8.5, open protocol
                    'B3',  # UKESM1-0-LL SSP5-8.5, open protocol
                    'B4',  # CESM2 SSP5-8.5, open protocol
                    'B5',  # CNRM-ESM2-1 SSP5-8.5, open protocol
                    'B6',  # CNRM-CM6-1 SSP5-8.5, standard protocol
                    'B8',  # UKESM1-0-LL SSP5-8.5, standard protocol
                    'B9',  # CESM2 SSP5-8.5, standard protocol
                    'B10']  # CNRM-ESM2-1 SSP5-8.5, standard protocol
        # Loop over experiments
        for exp in exp_list:
            # Loop over available input files
            in_fns = sorted(in_dir.glob(f'computed_limnsw_minus_ctrl_proj_AIS_*_exp{exp}.nc'))
            for in_fn in in_fns:
                # Create dictionary to hold data for this input file
                ais_dict = {'Group': f'P21'}
                # Get ice-sheet model institute and name
                ais_dict['Notes'] = f'{exp}_' + '_'.join(in_fn.name.split('_')[-3:-1])
                # Read DataSet
                in_ds = xr.load_dataset(in_fn)
                # Calculate SLE for target year relative to reference year for WAIS and EAIS; remember sign
                wais_da = in_ds[f'limnsw_region_{1}'] + in_ds[f'limnsw_region_{3}']  # include peninsula in WAIS
                eais_da = in_ds[f'limnsw_region_{2}']
                for region_name, in_da in [('WAIS', wais_da), ('EAIS', eais_da)]:
                    if ref_year == 2015:
                        ais_dict[region_name] = -1. * float(in_da.sel(time=target_year)) * convert_Gt_m
                    else:
                        ais_dict[region_name] = float(in_da.sel(time=ref_year) -
                                                      in_da.sel(time=target_year)) * convert_Gt_m
                # Append to DataFrame
                ism_df.loc[len(ism_df)] = ais_dict
    # Read Li et al. data
    if 'L23' in ensemble:
        # Lists containing experiments of interest and CMIP6 ESMs
        exp_dict = {'CMIP6_BC_1850-2100': 'L23'}
        esm_list = ['CNRM-CM6-1', 'UKESM1-0-LL', 'CESM2', 'CNRM-ESM2-1']  # ESMs also used by P21
        # Loop over experiments
        for exp, group in exp_dict.items():
            # Loop over ESMs
            for esm in esm_list:
                # Create dictionary to hold data for this input file
                ais_dict = {'Group': group}
                # Get ice-sheet model institute and name
                ais_dict['Notes'] = f'{exp} {esm}'
                # Read data
                in_fn = IN_BASE / exp / esm / 'fort.22'
                try:
                    in_df = pd.read_fwf(in_fn, skiprows=1, index_col='time')
                except ValueError:
                    in_df = pd.read_fwf(in_fn, skiprows=2, index_col='time')
                # Get SLE for target year relative to reference year for WAIS and EAIS; remember sign
                for region_name, in_varname in [('WAIS', 'eofw(m)'), ('EAIS', 'eofe(m)')]:
                    ais_dict[region_name] = in_df.loc[ref_year][in_varname] - in_df.loc[target_year][in_varname]
                # Append to DataFrame
                ism_df.loc[len(ism_df)] = ais_dict
    # Return result
    return ism_df


@cache
def read_gauge_info(gauge='TANJONG_PAGAR'):
    """
    Read name, ID, latitude, and longitude of tide gauge, using location.lst in FACTS
    (https://doi.org/10.5281/zenodo.7573653).

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
    in_dir = Path('data/radical-collaboration-facts-5086a75/input_files')
    in_fn = in_dir / 'location.lst'
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
        ID or name of gauge. Default is 'TANJONG_PAGAR' (equivalent to 1746)

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
    in_dir = Path('data/grd_fingerprints_data/FPRINT')
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
    component : str
        Component of GMSLR. Options are 'EAIS' (East Antarctic Ice Sheet, default),
        'WAIS' (West Antarctic Ice Sheet), 'GrIS' (Greenland Ice Sheet), and 'GMSLR' (total GMSLR).
        Note: for ISMIP6, 'PEN' (Antarctic peninsula) is also included in 'WAIS'.
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
        # Derive fusion distribution; rely on automatic broadcasting/alignment
        qf_da = w_da * pref_da + (1 - w_da) * outer_da
        # Correct median (which is currently nan due to nan in outer_da)
        med_idx = len(qf_da) // 2  # index corresponding to median
        qf_da[med_idx] = pref_da[med_idx]  # median follows preferred workflow
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
    Return triangular weighting function for fusion.

    Returns
    -------
    w_da : xarray DataArray
        DataArray of weights for preferred workflow, with weights depending on probability

    Notes
    -----
    This function follows https://github.com/grandey/d23a-fusion.
    """
    # Get a quantile function corresponding to a projection of total RSLC, using default parameters
    w_da = get_component_qf(workflow='wf_1e', component='EAIS', scenario='ssp585', year=2100
                            ).copy()  # use as template for w_da, with data to be updated
    # Update data to triangular weighting function, with weights depending on probability
    w_da.data = 1 - np.abs(w_da.p - 0.5) * 2
    # Rename
    w_da = w_da.rename('weights')
    return w_da


@cache
def quantify_bivariate_dependence(cop_workflow='wf_1e', components=('EAIS', 'WAIS')):
    """
    Quantify dependence between two ice-sheet components by fitting a bivariate copula to the year-2100 SSP5-8.5 data.

    Parameters
    ----------
    cop_workflow : str
        AR6 workflow (e.g. 'wf_1e', default), ice-sheet model ensemble (e.g. 'P21+L23'), or idealized dependence
        (e.g. '1'), for which to fit/specify the bivariate copula.
    components : tuple of str
        Two ice-sheet components. Default is ('EAIS', 'WAIS').

    Returns
    -------
    bicop : pv.Bicop
        Fitted bivariate copula (limited to single-parameter families).
    """
    # Check that two and only two components have been specified
    if len(components) != 2:
        raise ValueError(f'Unrecognized argument value: components={components}. Length should be 2.')
    # Case 1: idealised dependence by specifying the copula
    if cop_workflow in ('0', '1', '10', '01'):
        if cop_workflow == '1':  # perfect dependence
            bicop = pv.Bicop(family=pv.BicopFamily.gaussian, parameters=[1,])
        elif cop_workflow == '10' and components == tuple(COMPONENTS[:2]):  # perfect dep. between 1st & 2nd components
            bicop = pv.Bicop(family=pv.BicopFamily.gaussian, parameters=[1,])
        elif cop_workflow == '01' and components == tuple(COMPONENTS[1:]):  # perfect dep. between 2nd & 3rd components
            bicop = pv.Bicop(family=pv.BicopFamily.gaussian, parameters=[1,])
        else:  # independence
            bicop = pv.Bicop(family=pv.BicopFamily.indep)
    # Case 2: quantify dependence by fitting copula to samples
    else:
        # Read samples
        samples_list = []
        for component in components:
            if 'wf' in cop_workflow:  # if workflow, read samples DataArray and extract data
                samples = read_ar6_samples(workflow=cop_workflow, component=component, scenario='ssp585',
                                           year=2100).data
            else:  # if ISM ensemble, read samples DataFrame and extract data
                samples = read_ism_ensemble_data(ensemble=cop_workflow, ref_year=2015,
                                                 target_year=2100)[component].values
            samples_list.append(samples)
        # Fit copula (limited to single-parameter families)
        x_n2 = np.stack(samples_list, axis=1)
        u_n2 = pv.to_pseudo_obs(x_n2)
        controls = pv.FitControlsBicop(family_set=[pv.BicopFamily.indep, pv.BicopFamily.joe, pv.BicopFamily.gumbel,
                                                   pv.BicopFamily.gaussian, pv.BicopFamily.frank,
                                                   pv.BicopFamily.clayton])
        bicop = pv.Bicop(data=u_n2, controls=controls)  # fit
    # Return result
    return bicop


@cache
def sample_dvine_copula(families=(pv.BicopFamily.joe, pv.BicopFamily.clayton), rotations=(0, 0), taus=(0.8, 0.5),
                        n_samples=20000, plot=False):
    """
    Sample truncated D-vine copula with given families, rotations, Kendall's tau values, and number of samples.

    Parameters
    ----------
    families : tuple of pv.BicopFamily
        Pair copula families. Default is (pv.BicopFamily.joe, pv.BicopFamily.clayton).
    rotations : tuple of int
        Pair copula rotations. Ignored for Independence, Gaussian, and Frank copulas. Default is (0, 0).
    taus : tuple of float
        Pair copula Kendall's tau values. Default is (0.8, 0.5).
    n_samples : int
        Number of samples to generate. Default is 20000.
    plot : bool
        Plot the simulated data? Default is False.

    Returns
    -------
    u_nm : np.array
        An array of the simulated data, with shape (n_samples, len(families)+1).
    """
    # Check that tau values are all floats
    for tau in taus:
        if type(tau) not in [float, np.float64, int]:
            raise ValueError(f'tau={tau} is not a float.')
    # Derive parameters and create bivariate pair copulas
    bicops = []  # list to hold pair copulas
    for family, rotation, tau in zip(families, rotations, taus):
        parameters = pv.Bicop(family=family).tau_to_parameters(tau)
        if family in (pv.BicopFamily.indep, pv.BicopFamily.gaussian, pv.BicopFamily.frank):  # ignore rotation
            bicop = pv.Bicop(family=family, parameters=parameters)
        else:
            bicop = pv.Bicop(family=family, rotation=rotation, parameters=parameters)
        bicops.append(bicop)
    # Specify truncated D-vine structure
    struc = pv.DVineStructure(np.arange(len(families)+1)+1, trunc_lvl=1)
    # Create vine copula
    cop = pv.Vinecop(struc, [bicops, ])
    # Simulate data
    u_nm = cop.simulate(n=n_samples, seeds=[1, 2, 3, 4, 5])
    # Plot?
    if plot:
        sns.pairplot(pd.DataFrame(u_nm, columns=[f'u{n+1}' for n in range(len(families)+1)]), kind='hist')
        plt.suptitle(f'{cop.str()}', y=1.05)
        plt.show()
    return u_nm


@cache
def sample_trivariate_distribution(families=(pv.BicopFamily.joe, pv.BicopFamily.clayton),
                                   rotations=(0, 0), taus=(0.8, 0.5),
                                   marg_workflow='fusion_1e', marg_scenario='ssp585', marg_year=2100,
                                   plot=False):
    """
    Sample EAIS-WAIS-GrIS joint distribution.

    Parameters
    ----------
    families : tuple of pv.BicopFamily
        Pair copula families. Default is (pv.BicopFamily.joe, pv.BicopFamily.clayton).
    rotations : tuple of int
        Pair copula rotations. Ignored for Independence, Gaussian, and Frank copulas. Default is (0, 0).
    taus : tuple of float
        Pair copula Kendall's tau values. Default is (0.8, 0.5).
    marg_workflow : str
        AR6 workflow (e.g. 'wf_1e'), p-box bound (e.g. 'outer'), or fusion (e.g. 'fusion_1e', default),
        corresponding to the component marginals.
    marg_scenario : str
        Scenario to use for the component marginals. Options are 'ssp126' and 'ssp585' (default).
    marg_year : int
        Year to use for the component marginals. Default is 2100.
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
    n_samples = marg_n3.shape[0]
    # Sample copula
    u_n3 = sample_dvine_copula(families=families, rotations=rotations, taus=taus, n_samples=n_samples)
    # Transform marginals of copula
    x_n3 = np.transpose(np.asarray([np.quantile(marg_n3[:, i], u_n3[:, i]) for i in range(3)]))
    # Convert to DataFrame
    trivariate_df = pd.DataFrame(x_n3, columns=components)
    # Plot?
    if plot:
        sns.pairplot(trivariate_df, kind='hist')
        plt.show()
    return trivariate_df


def fig_component_marginals(marg_workflow='fusion_1e', marg_scenario='ssp585', marg_year=2100):
    """
    Plot figure showing marginals for the ice-sheet components.

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
        sns.kdeplot(qf_da, bw_adjust=0.3, color='b', fill=True, cut=0,  # limit to data limits
                    ax=ax)
        # Plot 5th, 50th, and 95th percentiles
        y_pos = 12.3  # position of percentile whiskers is tuned for the default parameters
        ax.plot([qf_da.quantile(p) for p in (0.05, 0.95)], [y_pos, y_pos], color='g', marker='|')
        ax.plot([qf_da.quantile(0.5), ], [y_pos, ], color='g', marker='x')
        if i == (n_axs-1):  # label percentiles in final subplot
            for p in [0.05, 0.5, 0.95]:
                ax.text(qf_da.quantile(p), y_pos-0.4, f'{int(p*100)}th', ha='center', va='top', color='g', rotation=90)
        # Skewness and kurtosis
        ax.text(1.15, 6.5,  # position tuned for the default parameters
                f"Skewness = {stats.skew(qf_da):.1f}\n"
                f"Fisher's kurtosis = {stats.kurtosis(qf_da, fisher=True):0.2f}",
                ha='right', va='center', fontsize='medium', bbox=dict(boxstyle='square,pad=0.5', fc='1', ec='0.85'))
        # Title etc
        ax.set_title(f'({chr(97+i)}) {component}')
    # x-axis label and limits
    axs[-1].set_xlabel(f'Contribution to GMSLR, m')
    axs[-1].set_xlim([-0.2, 1.2])
    return fig, axs


def fig_ism_ensemble(ref_year=2015, target_year=2100):
    """
    Plot figure showing combined ISM ensemble WAIS vs EAIS on (a) GMSLR scale and (b) copula scale.

    Parameters
    ----------
    ref_year : int
        Reference year. Default is 2015 (which is the start year for Payne et al. data).
    target_year : int
        Target year for difference. Default is 2100.

    Returns
    -------
    fig : Figure
    axs : array of Axes
    """
    # Read combined Antarctic ISM ensemble data from Payne et al. (2021) and Li et al. (2023)
    ism_df = read_ism_ensemble_data(ensemble='P21+L23', ref_year=ref_year, target_year=target_year).copy()
    # Include number of samples in label (for legend)
    for group in ism_df['Group'].unique():
        group_df = ism_df.loc[ism_df['Group'] == group]
        n_samples = len(group_df)
        ism_df = ism_df.replace(group, f'{group} (n = {n_samples})')
    # Create Figure and Axes
    fig, axs = plt.subplots(1, 2, figsize=(8, 4), tight_layout=True)
    # (a) WAIS vs EAIS on GMSLR scale (ie sea-level equivalent)
    ax = axs[0]
    sns.scatterplot(ism_df, x='EAIS', y='WAIS', hue='Group', style='Group', ax=ax)
    ax.legend(loc='lower right', fontsize='large', framealpha=1,
              edgecolor='0.85')  # specify edgecolor consistent with box in (b)
    ax.set_title(f'(a) Sea-level equivalent data')
    ax.set_xlabel('EAIS, m')
    ax.set_ylabel('WAIS, m')
    ax.set_xlim(-0.15, 0.65)
    ax.set_xticks(np.arange(-0.1, 0.61, 0.1))
    # (b) Pseudo-copula data on copula scale
    ax = axs[1]
    x_n2 = np.stack([ism_df['EAIS'], ism_df['WAIS']], axis=1)
    u_n2 = pv.to_pseudo_obs(x_n2)
    u_df = pd.DataFrame({'EAIS': u_n2[:, 0], 'WAIS': u_n2[:, 1], 'Group': ism_df['Group']})
    sns.scatterplot(u_df, x='EAIS', y='WAIS', hue='Group', style='Group', legend=False, ax=ax)
    ax.set_title(f'(b) Pseudo-copula data')
    ax.set_xlabel('EAIS, unitless')
    ax.set_ylabel('WAIS, unitless')
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])
    # Annotate with best-fit copula (limited to single-parameter families)
    bicop = quantify_bivariate_dependence(cop_workflow='P21+L23', components=('EAIS', 'WAIS'))
    ax.text(0.945, 0.06, f'Best fit: {bicop.str().split(",")[0]},\nwith {TAU_REG} = {bicop.tau:.2f}',
            fontsize='large', ha='right', va='bottom', bbox=dict(boxstyle='square,pad=0.5', fc='1', ec='0.85'))
    # Main title
    fig.suptitle('Antarctic ISM ensemble')
    return fig, axs


def fig_illustrate_copula():
    """
    Plot figure illustrating (a) bivariate distribution, (b) bivariate copula, and (c) truncated D-vine copula.

    Returns
    -------
    fig : Figure
    """
    # Create folder to hold temporary component plots
    temp_dir = Path('temp')
    temp_dir.mkdir(exist_ok=True)
    # Bivariate copulas to use
    bicop1 = quantify_bivariate_dependence(cop_workflow='wf_4', components=tuple(COMPONENTS[:2]))
    bicop2 = quantify_bivariate_dependence(cop_workflow='wf_4', components=tuple(COMPONENTS[1:]))
    # Sample corresponding vine copula and trivariate distribution
    families = (bicop1.family, bicop2.family)
    rotations = (bicop1.rotation, bicop2.rotation)
    taus = (bicop1.tau, bicop2.tau)
    u_nm = sample_dvine_copula(families=families, rotations=rotations, taus=taus, n_samples=20000)
    u_df = pd.DataFrame(u_nm, columns=COMPONENTS)
    x_df = sample_trivariate_distribution(families=families, rotations=rotations, taus=taus,
                                          marg_workflow='fusion_1e', marg_scenario='ssp585', marg_year=2100)
    # 1st component plot: bivariate joint distribution (with marginals)
    sns
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
    ax1.annotate('Marginal\ndensity', xy=(0.6, 0.92), xytext=(0.95, 0.92),
                 va='center', ha='center', xycoords='axes fraction', fontsize='large', color='b',
                 arrowprops=dict(arrowstyle='->', ec='b'))
    ax1.annotate('Joint\ndensity', xy=(0.45, 0.55), xytext=(0.37, 0.75),
                 va='center', ha='center', xycoords='axes fraction', fontsize='large', color='g',
                 arrowprops=dict(arrowstyle='->', ec='g'))
    ax1.set_title('(a) Bivariate distribution', fontsize='x-large')
    # Probability integral transform arrow
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.set_axis_off()
    ax2.text(0.5, 0.65, 'Transform to the copula scale\n(via probability integral transform)',
             ha='center', va='center',
             bbox=dict(boxstyle='rarrow,pad=0.5', fc='lavender', ec='purple'))
    ax2.text(0.5, 0.35, 'Transform to the sea-level scale\n(with desired marginal distributions)',
             ha='center', va='center',
             bbox=dict(boxstyle='larrow,pad=0.5', fc='lavender', ec='purple'))
    # (b) Bivariate copula
    ax3 = fig.add_subplot(gs[0, 2])
    ax3.set_axis_off()
    ax3.imshow(plt.imread(temp_dir / 'temp_copula1.png'))
    ax3.annotate('Uniform\nmarginal\ndensity', xy=(0.2, 0.92), xytext=(0, 0.92),
                 va='center', ha='center', xycoords='axes fraction', fontsize='large', color='b',
                 arrowprops=dict(arrowstyle='->', ec='b'))
    ax3.annotate('Copula\ndensity', xy=(0.45, 0.65), xytext=(0.33, 0.75),
                 va='center', ha='center', xycoords='axes fraction', fontsize='large', color='g',
                 arrowprops=dict(arrowstyle='->', ec='g'))
    ax3.set_title('(b) Bivariate copula', fontsize='x-large')
    # (c) Truncated D-vine copula
    ax4 = fig.add_subplot(gs[1, :])
    ax4.set_axis_off()
    ax4.set_xlim(-5, 5)  # specify coordinate system for arrangement of images and boxes
    ax4.set_ylim(-1, 2)
    ax4.plot([-4, 4], [0, 0], color='g')  # line connecting text boxes
    for i, x, s in zip(range(3), [-4.3, 0, 4.3], COMPONENTS):  # marginals
        ax4.text(x, 0, s, ha='center', va='center', fontsize=25,
                 bbox=dict(boxstyle='square,pad=0.5', fc='azure', ec='blue'))
        ax4.imshow(plt.imread(temp_dir / f'temp_{s}.png'), extent=[x-0.8, x+0.8, 0.4, 1.2])
    for i, x, s in zip(range(2), [-2, 2],
                       [f'{COMPONENTS[0]}–{COMPONENTS[1]}', f'{COMPONENTS[1]}–{COMPONENTS[2]}']):  # pair copulas
        ax4.text(x, -0.1, f'{s}\npair copula', ha='center', va='top', fontsize=18, color='g')
        ax4.imshow(plt.imread(temp_dir / f'temp_copula{i+2}.png'), extent=[x-0.9, x+0.9, 0, 1.8])
    ax4.set_title('(c) Truncated vine copula', fontsize='x-large')
    # Reset seaborn style
    sns.set_style(SNS_STYLE)
    return fig


def fig_dependence_table(cop_workflows=('wf_1e', 'wf_3e', 'P21+L23', 'wf_4')):
    """
    Plot heatmap table of bivariate copulas for AR6 workflows and ISM ensemble.

    Parameters
    ----------
    cop_workflows : tuple of str
        AR6 workflows (e.g. 'wf_1e'), ice-sheet model ensemble (e.g. 'P21+L23'), and/or idealized dependence (e.g. '1').
        Default is ('wf_1e', 'wf_3e', 'P21+L23', 'wf_4').

    Returns
    -------
    fig : Figure
    axs : array of Axes
    """
    # Component combinations correspond to columns
    columns = ['Notes',]
    columns += [f'{COMPONENTS[i]}—{COMPONENTS[i+1]}' for i in range(2)]
    columns.append(f'{COMPONENTS[0]}—{COMPONENTS[2]}')
    # DataFrames to hold bivariate copula annotation string and Kendall's tau
    annot_df = pd.DataFrame(columns=columns, dtype=object)
    tau_df = pd.DataFrame(columns=columns, dtype=float)
    # Add data to DataFrames
    for workflow in cop_workflows:  # loop over workflows
        for column in columns[1:]:  # loop over component combinations (ignoring Notes column)
            try:  # quantify dependence
                bicop = quantify_bivariate_dependence(cop_workflow=workflow, components=tuple(column.split('—')))
                annot_df.loc[workflow, column] = f'{bicop.str().split(",")[0]},\n{TAU_BOLD} = {bicop.tau:.2f}'
                tau_df.loc[workflow, column] = bicop.tau
            except KeyError:
                annot_df.loc[workflow, column] = 'N/A'
                tau_df.loc[workflow, column] = 0.  # set as zero not missing, so that annotations are not masked
        try:
            annot_df.loc[workflow, 'Notes'] = WORKFLOW_NOTES[workflow]  # include workflow notes in annotation DataFrame
        except KeyError:
            annot_df.loc[workflow, 'Notes'] = ''
        tau_df.loc[workflow, 'Notes'] = 0.
    # Create Figure and Axes
    fig, ax = plt.subplots(1, 1, figsize=(10, 1*len(cop_workflows)), tight_layout=True)
    # Plot heatmap
    sns.heatmap(tau_df, cmap='seismic', vmin=-1., vmax=1., annot=annot_df, fmt='', annot_kws={'weight': 'bold'},
                linecolor='lightgrey', linewidths=1, ax=ax)
    # Customise plot
    ax.tick_params(top=False, bottom=False, left=False, right=False, labeltop=True, labelbottom=False, rotation=0)
    ax.set_yticklabels([WORKFLOW_LABELS[workflow] for workflow in cop_workflows])
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontweight('bold')
    cbar = ax.collections[0].colorbar
    cbar.set_ticks([-1., 0., 1.])
    cbar.set_label(f'Kendall\'s {TAU_BOLD}')
    if cop_workflows == ('wf_1e', 'wf_3e', 'P21+L23', 'wf_4'):
        fig.suptitle('Bivariate copulas fitted to the workflow samples and ISM ensemble')
    return fig, ax


def ax_total_vs_tau(families=(pv.BicopFamily.joe, pv.BicopFamily.clayton), rotations=(0, 0), colors=('darkred', 'blue'),
                    marg_workflow='fusion_1e', marg_scenario='ssp585', marg_year=2100,
                    ax=None):
    """
    Plot median and 5th-95th percentile range of total ice-sheet contribution (y-axis) vs Kendall's tau (x-axis).

    Parameters
    ----------
    families : tuple
        Pair copula families. Default is (pv.BicopFamily.joe, pv.BicopFamily.clayton).
    rotations : tuple
        Pair copula rotations. Default is (0, 0).
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
    # For each copula, calculate total ice-sheet contribution for different tau values and plot median & 5th-95th
    tau_t = np.linspace(0, 1, 51)  # tau values to use
    p95_t_list = []  # list to hold 95th percentile arrays
    for family, rotation, color, hatch, linestyle, linewidth in zip(families, rotations, colors,
                                                                    ('//', r'\\'), ('--', '-.'), (3, 2)):
        families2 = (family, )*2
        label = family.name.capitalize()
        p50_t = np.full(len(tau_t), np.nan)  # array to hold median at each tau
        p5_t = np.full(len(tau_t), np.nan)  # 5th percentile
        p95_t = np.full(len(tau_t), np.nan)  # 95th percentile
        for t, tau in enumerate(tau_t):  # for each tau, calculate total ice-sheet contribution
            trivariate_df = sample_trivariate_distribution(families=families2, rotations=(rotation, )*2, taus=(tau, )*2,
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
    else:  # if plotting two or more families, use the diff at tau = 0.5 (middle index of 25)
        p95_min = min([p95_t[25] for p95_t in p95_t_list])
        p95_max = max([p95_t[25] for p95_t in p95_t_list])
    p95_diff = p95_max - p95_min  # difference
    p95_perc = 100. * p95_diff / p95_min  # percentage difference
    ax.arrow(0.5, p95_min, 0., p95_diff, color='k', head_width=0.02, length_includes_head=True)  # plot arrow
    diff_str = f'+ {p95_diff:.1f} m'  # annotate with absolute diff
    ax.text(0.47, np.mean([p95_min, p95_max]), diff_str, va='center', ha='right', fontsize='large')
    perc_str = f'+ {p95_perc:.0f} %'  # annotate with percentage diff
    ax.text(0.53, np.mean([p95_min, p95_max]), perc_str, va='center', ha='left', fontsize='large')
    # Customize plot
    ax.legend(loc='upper left', fontsize='large')
    ax.set_xlim(tau_t[0], tau_t[-1])
    ax.set_xlabel(f"Kendall's {TAU_BOLD}")
    ax.set_ylabel(f'Total ice-sheet contribution, m')
    ax.yaxis.set_minor_locator(plt.MultipleLocator(0.1))
    ax.xaxis.set_minor_locator(plt.FixedLocator(tau_t))
    ax.tick_params(which='minor', direction='in', color='0.7', bottom=True, top=True, left=True, right=True)
    ax.set_title(f'{marg_workflow} {marg_scenario} {marg_year}')
    return ax


def fig_total_vs_tau(families_a=(pv.BicopFamily.gaussian, ), families_b=(pv.BicopFamily.joe, pv.BicopFamily.clayton),
                     colors_a=('green', ), colors_b=('darkred', 'blue'),
                     marg_workflow='fusion_1e', marg_scenario='ssp585', marg_year=2100, ylim=(-0.2, 3.25)):
    """
    Plot figure showing median and 5th-95th percentile range of total ice-sheet contribution (y-axis) vs tau (x-axis)
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
        Limits for y-axis. Default is (-0.2, 3.25).

    Returns
    -------
    fig : Figure
    axs : array of Axes
    """
    # Create figure and axes
    fig, axs = plt.subplots(1, 2, figsize=(10, 4), sharey=True, constrained_layout=True)
    # (a)
    ax = axs[0]
    _ = ax_total_vs_tau(families=families_a, rotations=(0, 0), colors=colors_a,
                        marg_workflow=marg_workflow, marg_scenario=marg_scenario, marg_year=marg_year, ax=ax)
    ax.set_title(f'(a) {" & ".join(f.name.capitalize() for f in families_a)} pair copulas')
    # (b)
    ax = axs[1]
    _ = ax_total_vs_tau(families=families_b, rotations=(0, 0), colors=colors_b,
                        marg_workflow=marg_workflow, marg_scenario=marg_scenario, marg_year=marg_year, ax=ax)
    ax.set_title(f'(b) {" & ".join(f.name.capitalize() for f in families_b)} pair copulas')
    ax.set_ylabel(None)
    ax.set_ylim(ylim)
    return fig, axs


def ax_total_vs_time(cop_workflows=('wf_3e', '0'),
                     marg_workflow='fusion_1e', marg_scenario='ssp585', marg_years=np.arange(2020, 2101, 10),
                     show_percent_diff=True, thresh_for_timing_diff=(1.4, 0.2), ax=None):
    """
    Plot median and 5th-95th percentile range of total ice-sheet contribution (y-axis) vs time (x-axis).

    Parameters
    ----------
    cop_workflows : tuple of str
        AR6 workflow (e.g. 'wf_1e'), ISM ensemble (e.g. 'P21+L23), perfect dependence ('1'), independence ('0'),
        or perfect dependence between two components (e.g. '10') corresponding to the vine copula to be used.
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
    # For each copula, calculate total ice-sheet contribution for different years and plot
    for cop_workflow, hatch, linestyle, linewidth in zip(cop_workflows, ('//', '..'), ('--', '-.'), (3, 2)):
        # Specify pair copula families, rotations, and tau
        bicop1 = quantify_bivariate_dependence(cop_workflow=cop_workflow, components=tuple(COMPONENTS[:2]))
        try:
            bicop2 = quantify_bivariate_dependence(cop_workflow=cop_workflow, components=tuple(COMPONENTS[1:]))
        except KeyError:
            print(f'No {COMPONENTS[1]}-{COMPONENTS[1]} dependence found for {cop_workflow}; using independence')
            bicop2 = pv.Bicop(family=pv.BicopFamily.indep)
        families = (bicop1.family, bicop2.family)
        rotations = (bicop1.rotation, bicop2.rotation)
        taus = (bicop1.tau, bicop2.tau)
        # Create DataFrame to hold percentile time series for this copula
        data_df = pd.DataFrame()
        # For each year, calculate percentiles of total ice-sheet contribution
        for year in marg_years:
            trivariate_df = sample_trivariate_distribution(families=families, rotations=rotations, taus=taus,
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
            if abs(percent_diff) > 0.5:  # format percentage diff as string
                percent_str = f'{percent_diff:+.0f} %'
            else:
                percent_str = f'{percent_diff:+.1f} %'
            ax.text(year+2.5, np.mean([val1, val0]), percent_str,  # annotate with percentage diff
                    color='k', va='center', ha='left', fontsize='large')
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
    ax.set_ylabel(f'Total ice-sheet contribution, m')
    ax.yaxis.set_minor_locator(plt.MultipleLocator(0.1))
    ax.xaxis.set_minor_locator(plt.MultipleLocator(1))
    ax.tick_params(which='minor', direction='in', color='0.7', bottom=True, top=True, left=True, right=True)
    ax.legend(loc='upper left', fontsize='large')
    return ax


def fig_total_vs_time(cop_workflows=('wf_1e', 'wf_3e', 'P21+L23', 'wf_4'), ref_workflow='0',
                      marg_workflow='fusion_1e', marg_scenario='ssp585', marg_years=np.arange(2020, 2101, 10),
                      thresh_for_timing_diff=(1.4, 0.2), ylim=(-0.2, 2.3)):
    """
    Plot figure showing median and 5th-95th percentile range of total ice-sheet contribution (y-axis) vs time (x-axis)
    for different copulas (in multiple panels).

    Parameters
    ----------
    cop_workflows : tuple of str
        AR6 workflows (e.g. 'wf_1e'), ISM ensemble (e.g. 'P21+L23), perfect dependence ('1'), independence ('0'),
        or perfect dependence between two components (e.g. '10') corresponding to the vine copula to be used.
        Note, these will be plotted in separate panels. Default is ('wf_1e', 'wf_3e', 'P21+L23', 'wf_4').
    ref_workflow : str
        Workflow corresponding to the vine copula to be used as the reference in all panels. Default is '0'.
    marg_workflow : str
        AR6 workflow (e.g. 'wf_1e'), p-box bound ('lower', 'upper', 'outer'), or fusion (e.g. 'fusion_1e', default),
        corresponding to the component marginals.
    marg_scenario : str
        The scenario for the component marginals. Default is 'ssp585'.
    marg_years : np.array
        Target years for the component marginals. Default is np.arange(2020, 2101, 10).
    thresh_for_timing_diff : tuple, True, or None
        Thresholds to use if demonstrating the difference in timing at the 95th percentile and median.
        If True, select automatically. Default is (1.4, 0.2).
    ylim : tuple
        Limits for y-axis. Default is (-0.2, 2.3).

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
    fig, axs = plt.subplots(nrows, ncols, figsize=(5*ncols, 4*nrows), sharey=True, sharex=True, tight_layout=True)
    # Flatten axs
    try:
        axs_flat = axs.flatten()
    except AttributeError:  # if only one panel
        axs_flat = [axs, ]
    # Plot panels
    for i, (cop_workflow, ax) in enumerate(zip(cop_workflows, axs_flat)):
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
    Plot median and 5th-95th percentile range of total ice-sheet contribution (y-axis) vs GrIS GRD fingerprint (x-axis).

    Parameters
    ----------
    cop_workflows : tuple of str
        AR6 workflow (e.g. 'wf_1e'), ISM ensemble (e.g. 'P21+L23), perfect dependence ('1'), independence ('0'),
        or perfect dependence between two components (e.g. '10') corresponding to the vine copula to be used.
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
    # For each copula, calculate total ice-sheet contribution for diff. GrIS fingerprints and plot median & 5th-95th
    for cop_workflow, hatch, linestyle, linewidth in zip(cop_workflows, ('//', '..'), ('--', '-.'), (3, 2)):
        # Specify pair copula families, rotations, and tau
        bicop1 = quantify_bivariate_dependence(cop_workflow=cop_workflow, components=tuple(COMPONENTS[:2]))
        try:
            bicop2 = quantify_bivariate_dependence(cop_workflow=cop_workflow, components=tuple(COMPONENTS[1:]))
        except KeyError:
            print(f'No {COMPONENTS[1]}-{COMPONENTS[1]} dependence found for {cop_workflow}; using independence')
            bicop2 = pv.Bicop(family=pv.BicopFamily.indep)
        families = (bicop1.family, bicop2.family)
        rotations = (bicop1.rotation, bicop2.rotation)
        taus = (bicop1.tau, bicop2.tau)
        # Get trivariate distribution data for global mean (ie fingerprints all 1.0)
        x_df = sample_trivariate_distribution(families=families, rotations=rotations, taus=taus,
                                              marg_workflow=marg_workflow, marg_scenario=marg_scenario,
                                              marg_year=marg_year)
        # Create DataFrame to hold percentile data across GrIS fingerprints
        data_df = pd.DataFrame()
        # Loop over GrIS fingerprints and calculate 5th, 50th, and 95th percentiles of total ice-sheet contribution
        for gris_fp in gris_fp_g:
            for perc in (5, 50, 95):
                sum_ser = eais_fp * x_df['EAIS'] + wais_fp * x_df['WAIS'] + gris_fp * x_df['GrIS']
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
    ax.set_ylabel(f'Total ice-sheet contribution, m')
    ax.yaxis.set_minor_locator(plt.MultipleLocator(0.1))
    ax.xaxis.set_minor_locator(plt.FixedLocator(gris_fp_g))
    ax.tick_params(which='minor', direction='in', color='0.7', bottom=True, top=True, left=True, right=True)
    ax.set_title('Ice-sheet contribution to RSLC vs fingerprint of GrIS')
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


def name_save_fig(fig, fso='f', exts=('pdf', 'png'), close=False):
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
