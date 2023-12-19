"""
d23b:
    Functions that support the analysis contained in the d23b-ice-dependence repository.

Author:
    Benjamin S. Grandey, 2022-2023.
"""


from functools import cache
import itertools
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path
import pyvinecopulib as pv
from scipy import stats
import seaborn as sns
import warnings
from watermark import watermark
import xarray as xr


# Matplotlib settings
plt.rcParams['axes.titlesize'] = 'x-large'
plt.rcParams['axes.titleweight'] = 'bold'
plt.rcParams['axes.labelsize'] = 'large'
plt.rcParams['axes.labelweight'] = 'bold'
plt.rcParams['xtick.labelsize'] = 'large'
plt.rcParams['ytick.labelsize'] = 'large'
plt.rcParams['savefig.dpi'] = 300

# Seaborn style
sns.set_style('whitegrid')


# Constants
IN_BASE = Path.cwd() / 'data'  # base directory of input data
COMPONENTS = ['EAIS', 'WAIS', 'GrIS']  # ice-sheet components of sea level, ordered according to vine copula


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
def quantify_bivariate_dependence(workflow='wf_1e', components=('EAIS', 'WAIS'), year=2100):
    """
    Quantify dependence between two ice-sheet components by fitting a bivariate copula to the SSP5-8.5 data.

    Parameters
    ----------
    workflow : str
        AR6 workflow (e.g. 'wf_1e', default) or ice-sheet model ensemble (e.g. 'P21+L23').
    components : tuple of str
        Two ice-sheet components. Default is ('EAIS', 'WAIS').
    year : int
        Year. Default is 2100.

    Returns
    -------
    bicop : pv.Bicop
        Fitted bivariate copula (limited to single-parameter families).
    """
    # Check that two and only two components have been specified
    if len(components) != 2:
        raise ValueError(f'Unrecognized argument value: components={components}. Length should be 2.')
    # Read samples
    samples_list = []
    for component in components:
        if 'wf' in workflow:  # if workflow, read samples DataArray and extract data
            samples = read_ar6_samples(workflow=workflow, component=component, scenario='ssp585', year=year).data
        else:  # if ISM ensemble, read samples DataFrame and extract data
            samples = read_ism_ensemble_data(ensemble=workflow, ref_year=2015, target_year=year)[component].values
        samples_list.append(samples)
    # Fit copula (limited to single-parameter families)
    x_n2 = np.stack(samples_list, axis=1)
    u_n2 = pv.to_pseudo_obs(x_n2)
    controls = pv.FitControlsBicop(family_set=[pv.BicopFamily.indep, pv.BicopFamily.joe, pv.BicopFamily.gumbel,
                                               pv.BicopFamily.gaussian, pv.BicopFamily.frank, pv.BicopFamily.clayton])
    bicop = pv.Bicop(data=u_n2, controls=controls)  # fit
    # Return result
    return bicop


@cache
def sample_dvine_copula(families=(pv.BicopFamily.gaussian, pv.BicopFamily.gaussian), rotations=(0, 0), taus=(0.5, 0.5),
                        n_samples=20000, plot=False):
    """
    Sample truncated D-vine copula with given families, rotations, Kendall's tau values, and number of samples.

    Parameters
    ----------
    families : tuple of pv.BicopFamily
        Pair copula families. Default is (pv.BicopFamily.gaussian, pv.BicopFamily.gaussian).
    rotations : tuple of int
        Pair copula rotations. Ignored for Independence, Gaussian, and Frank copulas. Default is (0, 0).
    taus : tuple of float
        Pair copula Kendall's tau values. Default is (0.5, 0.5).
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
def sample_trivariate_distribution(workflow='fusion_1e', scenario='ssp585', year=2100,
                                   families=(pv.BicopFamily.gaussian, pv.BicopFamily.gaussian),
                                   rotations=(0, 0), taus=(0.5, 0.5), plot=False):
    """
    Sample EAIS-WAIS-GrIS joint distribution.

    Parameters
    ----------
    workflow : str
        AR6 workflow (e.g. 'wf_1e'), p-box bound (e.g. 'outer'), or fusion (e.g. 'fusion_1e', default).
    scenario : str
        Scenario. Options are 'ssp126' and 'ssp585' (default).
    year : int
        Year. Default is 2100.
    families : tuple of pv.BicopFamily
        Pair copula families. Default is (pv.BicopFamily.gaussian, pv.BicopFamily.gaussian).
    rotations : tuple of int
        Pair copula rotations. Ignored for Independence, Gaussian, and Frank copulas. Default is (0, 0).
    taus : tuple of float
        Pair copula Kendall's tau values. Default is (0.5, 0.5).
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
        qf_da = get_component_qf(workflow=workflow, component=component, scenario=scenario, year=year)
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


def fig_component_marginals(workflow='fusion_1e', scenario='ssp585', year=2100):
    """
    Plot figure showing marginals for the ice-sheet components.

    Parameters
    ----------
    workflow : str
        AR6 workflow (e.g. 'wf_1e'), p-box bound (e.g. 'outer'), or fusion (e.g. 'fusion_1e', default).
    scenario : str
        Scenario. Options are 'ssp126' and 'ssp585' (default).
    year : int
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
        qf_da = get_component_qf(workflow=workflow, component=component, scenario=scenario, year=year)
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
    axs[-1].set_xlabel(f'Contribution to GMSLR (2005–{year}), m')
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
    ax.set_title(f'(a) ISM ensemble WAIS vs EAIS')
    ax.set_xlabel('EAIS, m (sea-level equivalent)')
    ax.set_ylabel('WAIS, m (sea-level equivalent)')
    ax.set_xlim(-0.15, 0.65)
    ax.set_xticks(np.arange(-0.1, 0.6, 0.1))
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
    # Fit copula (limited to single-parameter families)
    bicop = quantify_bivariate_dependence(workflow='P21+L23', components=('EAIS', 'WAIS'), year=2100)
    ax.text(0.75, 0.06, f'Best fit: {bicop.family.name.capitalize()}\nwith $\\tau$ = {bicop.tau:.2f}',
            fontsize='large', ha='center', va='bottom', bbox=dict(boxstyle='square,pad=0.5', fc='1', ec='0.85'))
    return fig, axs


# OLDER CODE BELOW - TO REVISE

@cache
def sample_bivariate_copula(family=pv.BicopFamily.gaussian, rotation=0, tau=0.5, n_samples=int(1e5), plot=False):
    """
    Sample bivariate copula with a given family, rotation, Kendall's tau, and number of samples.

    Parameters
    ----------
    family : pv.BicopFamily
        Bivariate copula family. Default is pv.BicopFamily.gaussian.
    rotation : int
        Bivariate copula rotation. Ignored for Independence, Gaussian, and Frank copulas. Default is 0.
    tau : float
        Kendall's tau for the copula. Default is 0.5.
    n_samples : int
        Number of samples to generate. Default is int(1e5).
    plot : bool
        Plot the simulated data? Default is False.

    Returns
    -------
    u_n2 : np.array
        An array of the simulated data, with shape (n_samples, 2).
    """
    # Check that tau is a float
    if type(tau) not in [float, np.float64, int]:
        raise ValueError(f'tau={tau} is not a float.')
    # Derive parameters and create bivariate copula
    parameters = pv.Bicop(family=family).tau_to_parameters(tau)
    if family in (pv.BicopFamily.indep, pv.BicopFamily.gaussian, pv.BicopFamily.frank):  # ignore rotation
        cop = pv.Bicop(family=family, parameters=parameters)
    else:
        cop = pv.Bicop(family=family, rotation=rotation, parameters=parameters)
    # Simulate data
    u_n2 = cop.simulate(n=n_samples, seeds=[1, 2, 3, 4, 5])
    # Plot?
    if plot:
        sns.jointplot(pd.DataFrame(u_n2, columns=['u1', 'u2']), x='u1', y='u2', kind='hex')
        plt.suptitle(f'{cop.str()}', y=1.01)
        plt.show()
    return u_n2


# Figure illustrating bivariate distribution, bivariate copula, and truncated vine copula

def fig_illustrate_bivariate_copula_vine(projection_source='fusion', scenario='SSP5-8.5', year=2100,
                                         family=pv.BicopFamily.gaussian, rotation=0, tau=0.5, n_samples=int(1e5)):
    """
    Plot figure illustrating (a) bivariate distribution, (b) bivariate copula, and (c) truncated D-vine copula.

    Parameters
    ----------
    projection_source : str
        Projection source. Options are 'ISMIP6'/'model-based',  'SEJ'/'expert-based',
        'p-box'/'bounding quantile function', and 'fusion' (default).
    scenario : str
        Scenario. Options are 'SSP1-2.6' and 'SSP5-8.5' (default).
    year : int
        Year. Default is 2100.
    family : pv.BicopFamily
        Pair copula family. (We assume that the two pair copulas are identical.) Default is pv.BicopFamily.gaussian.
    rotation : int
        Pair copula rotation. Default is 0.
    tau : float
        Pair copula Kendall's tau. Default is 0.5.
    n_samples : int
        Number of samples. Default is int(1e5).

    Returns
    -------
    fig : Figure
    """
    # Create folder to hold temporary component plots
    temp_dir = Path('temp')
    temp_dir.mkdir(exist_ok=True)
    # 1st component plot: bivariate joint distribution (with marginals)
    x_n3 = sample_trivariate_distribution(projection_source=projection_source, scenario=scenario, year=year,
                                          families=(family, )*2, rotations=(rotation, )*2, taus=(tau, )*2,
                                          n_samples=n_samples, plot=False)
    x_df = pd.DataFrame(x_n3, columns=['EAIS, m', 'WAIS, m', 'GrIS, m'])
    g = sns.jointplot(x_df, x='EAIS, m', y='WAIS, m', kind='kde', cmap='Greens', fill=True, levels=7, cut=0,
                      marginal_ticks=False, marginal_kws={'bw_adjust': 0.2, 'color': 'b'},
                      xlim=[-0.2, 0.2], ylim=[-0.2, 0.5], height=3)
    g.savefig(temp_dir / 'temp_joint.png')
    plt.close(g.fig)
    # 2nd component: bivariate copula (with uniform marginals)
    u_n2 = sample_bivariate_copula(family=family, rotation=rotation, tau=tau, n_samples=n_samples, plot=False)
    u_df = pd.DataFrame(u_n2, columns=['EAIS', 'WAIS'])
    g = sns.jointplot(u_df, x='EAIS', y='WAIS', kind='kde', cmap='Greens', fill=True, levels=7, cut=0, clip=[0, 1],
                      marginal_ticks=False, marginal_kws={'bw_adjust': 0.2, 'color': 'b'},
                      xlim=[0, 1], ylim=[0, 1], height=3)
    g.savefig(temp_dir / 'temp_copula1.png')
    plt.close(g.fig)
    # 3rd component: copula without marginals
    fig, ax = plt.subplots(1, 1, figsize=(3, 3), tight_layout=True)
    sns.kdeplot(u_df, x='EAIS', y='WAIS', cmap='Greens', fill=True, levels=7, cut=0, clip=[0, 1], ax=ax)
    ax.grid(False)
    ax.set_xlabel(None)
    ax.set_ylabel(None)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    fig.savefig(temp_dir / 'temp_copula2.png')
    plt.close(fig)
    # 4th-6th components: marginals
    for s in ['EAIS', 'WAIS', 'GrIS']:
        fig, ax = plt.subplots(1, 1, figsize=(2, 1), tight_layout=True)
        sns.kdeplot(x_df, x=f'{s}, m', bw_adjust=0.2, color='b', fill=True, cut=0, ax=ax)
        ax.grid(False)
        ax.set_xlabel(None)
        ax.set_ylabel(None)
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        fig.savefig(temp_dir / f'temp_{s}.png')
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
    ax1.annotate('Joint\ndensity', xy=(0.5, 0.6), xytext=(0.33, 0.79),
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
    ax3.annotate('Copula\ndensity', xy=(0.45, 0.65), xytext=(0.3, 0.77),
                 va='center', ha='center', xycoords='axes fraction', fontsize='large', color='g',
                 arrowprops=dict(arrowstyle='->', ec='g'))
    ax3.set_title('(b) Bivariate copula', fontsize='x-large')
    # (c) Truncated D-vine copula
    ax4 = fig.add_subplot(gs[1, :])
    ax4.set_axis_off()
    ax4.set_xlim(-5, 5)  # specify coordinate system for arrangement of images and boxes
    ax4.set_ylim(-1, 2)
    ax4.plot([-4, 4], [0, 0], color='g')  # line connecting text boxes
    for x, s in zip([-4.3, 0, 4.3], ['EAIS', 'WAIS', 'GrIS']):  # marginals
        ax4.text(x, 0, s, ha='center', va='center', fontsize=25,
                 bbox=dict(boxstyle='square,pad=0.5', fc='azure', ec='blue'))
        ax4.imshow(plt.imread(temp_dir / f'temp_{s}.png'), extent=[x-0.8, x+0.8, 0.4, 1.2])
    for x, s in zip([-2, 2], ['EAIS–WAIS', 'WAIS–GrIS']):  # pair copulas
        ax4.text(x, -0.1, f'{s}\npair copula', ha='center', va='top', fontsize=18, color='g')
        ax4.imshow(plt.imread(temp_dir / 'temp_copula2.png'), extent=[x-0.9, x+0.9, 0, 1.8])
    ax4.set_title('(c) Truncated vine copula', fontsize='x-large')
    return fig


# Figures showing total ice-sheet contribution

def ax_total_vs_tau(projection_source='fusion', scenario='SSP5-8.5', year=2100,
                    families=(pv.BicopFamily.joe, pv.BicopFamily.clayton),
                    rotations=(0, 0),
                    colors=('darkred', 'blue'),
                    n_samples=int(1e5), ax=None):
    """
    Plot median and 5th-95th percentile range of total ice-sheet contribution (y-axis) vs Kendall's tau (x-axis).

    Parameters
    ----------
    projection_source : str
        The projection source for the marginal distributions. Default is 'fusion'.
    scenario : str
        The scenario for the marginal distributions. Default is 'SSP5-8.5'
    year : int
        Year for which to plot data. Default is 2100.
    families : tuple
        Pair copula families. Default is (pv.BicopFamily.joe, pv.BicopFamily.clayton).
    rotations : tuple
        Pair copula rotations. Default is (0, 0).
    colors : tuple
        Colors to use when plotting. Default is ('darkred', 'blue').
    n_samples : int
        Number of samples to generate for each family and tau. Default is int(1e5).
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
    # For each copula, calculate EAIS+WAIS+GIS for different tau values and plot median & 5th-95th
    for family, rotation, color, hatch, linestyle, linewidth in zip(families, rotations, colors,
                                                                    ('//', r'\\'), ('--', '-.'), (3, 2)):
        if family == 'Mixture':
            families2 = 'Mixture'  # used when calling sample_trivariate_distribution() below
            label = 'Mixture'  # label to use in legend
        else:
            families2 = (family, )*2
            label = family.name.capitalize()
        tau_t = np.linspace(0, 1, 51)  # tau values to use
        p50_t = np.full(len(tau_t), np.nan)  # array to hold median at each tau
        p5_t = np.full(len(tau_t), np.nan)  # 5th percentile
        p95_t = np.full(len(tau_t), np.nan)  # 95th percentile
        for t, tau in enumerate(tau_t):  # for each tau, calculate total ice-sheet contribution
            x_n3 = sample_trivariate_distribution(projection_source=projection_source, scenario=scenario, year=year,
                                                  families=families2, rotations=(rotation, )*2, taus=(tau, )*2,
                                                  n_samples=n_samples, plot=False)
            p50_t[t] = np.percentile(x_n3.sum(axis=1), 50)  # median
            p5_t[t] = np.percentile(x_n3.sum(axis=1), 5)  # 5th percentile
            p95_t[t] = np.percentile(x_n3.sum(axis=1), 95)  # 95th percentile
        # Plot data for this family
        ax.fill_between(tau_t, p5_t, p95_t, color=color, alpha=0.2, label=f'{label} (5th–95th)', hatch=hatch)
        ax.plot(tau_t, p50_t, color=color, label=f'{label} (median)', linestyle=linestyle, linewidth=linewidth)
    # Customize plot
    ax.legend(loc='upper left', fontsize='large')
    ax.set_xlim(tau_t[0], tau_t[-1])
    ax.set_xlabel(r"Kendall's $\bf{\tau}$")
    ax.set_ylabel(f'Total ice-sheet contribution (2005–{year}), m')
    ax.yaxis.set_minor_locator(plt.MultipleLocator(0.1))
    ax.xaxis.set_minor_locator(plt.FixedLocator(tau_t))
    ax.tick_params(which='minor', direction='in', color='0.7', bottom=True, top=True, left=True, right=True)
    ax.set_title(f'{projection_source} {scenario} {year}')
    return ax


def fig_total_vs_tau(projection_source='fusion', scenario='SSP5-8.5', year=2100,
                     families_a=(pv.BicopFamily.gaussian, ), families_b=(pv.BicopFamily.joe, pv.BicopFamily.clayton),
                     colors_a=('green', ), colors_b=('darkred', 'blue'), ylim=(-0.2, 3.25),
                     n_samples=int(1e5)):
    """
    Plot figure showing median and 5th-95th percentile range of total ice-sheet contribution (y-axis) vs Kendall's
    tau (x-axis) for (a) Gaussian pair copulas and (b) Joe/Clayton pair copulas (default).

    Parameters
    ----------
    projection_source : str
        The projection source for the marginal distributions. Default is 'fusion'.
    scenario : str
        The scenario for the marginal distributions. Default is 'SSP5-8.5'
    year : int
        Year for which to plot data. Default is 2100.
    families_a and families_b : tuple
        Pair copula families to use for panels (a) and (b).
        Default is (pv.BicopFamily.gaussian, ) and (pv.BicopFamily.joe, pv.BicopFamily.clayton).
    colors_a and colors_b : tuple
        Colors to use when plotting. Default is ('green', ) and ('darkred', 'blue').
    ylim : tuple
        Limits for y-axis. Default is (-0.2, 3.25).
    n_samples : int
        Number of samples to generate for each family and tau. Default is int(1e5).

    Returns
    -------
    fig : Figure
    axs : array of Axes
    """
    # Create figure and axes
    fig, axs = plt.subplots(1, 2, figsize=(10, 4), sharey=True, constrained_layout=True)
    # (a)
    ax = axs[0]
    _ = ax_total_vs_tau(projection_source=projection_source, scenario=scenario, year=year,
                        families=families_a, colors=colors_a,
                        n_samples=n_samples, ax=ax)
    ax.set_title(f'(a) {" and ".join(f.name.capitalize() for f in families_a)} pair copulas')
    # (b)
    ax = axs[1]
    _ = ax_total_vs_tau(projection_source=projection_source, scenario=scenario, year=year,
                        families=families_b, colors=colors_b,
                        n_samples=n_samples, ax=ax)
    ax.set_title(f'(b) {" and ".join(f.name.capitalize() for f in families_b)} pair copulas')
    ax.set_ylabel(None)
    ax.set_ylim(ylim)
    return fig, axs


def ax_total_vs_time(projection_source='fusion', scenario='SSP5-8.5', years=np.arange(2020, 2101, 10),
                     families=(pv.BicopFamily.gaussian, pv.BicopFamily.indep), rotations=(0, 0), taus=(1.0, 0.0),
                     colors=('darkgreen', 'darkorange'), thresh_for_timing_diff=True, n_samples=int(1e5), ax=None):
    """
    Plot median and 5th-95th percentile range of total ice-sheet contribution (y-axis) vs time (x-axis).

    Parameters
    ----------
    projection_source : str
        The projection source for the marginal distributions. Default is 'fusion'.
    scenario : str
        The scenario for the marginal distributions. Default is 'SSP5-8.5'
    years : np.array
        Years for which to plot data. Default is np.arange(2020, 2101, 10).
    families : tuple
        Pair copula families. Default is (pv.BicopFamily.gaussian, pv.BicopFamily.indep).
    rotations : tuple
        Pair copula rotations. Default is (0, 0).
    taus: tuple
        Pair copula Kendall's tau values. Default is (1.0, 0.0).
    colors : tuple
        Colors to use when plotting. Default is ('darkgreen', 'darkorange').
    thresh_for_timing_diff : tuple, True, or None
        Thresholds to use if demonstrating the difference in timing at the 95th percentile and median.
        If True, select automatically. Default is True.
    n_samples : int
        Number of samples to generate for each copula. Default is int(1e5).
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
    # List to hold DataFrames created below
    data_dfs = []
    # For each copula, calculate EAIS+WAIS+GrIS for different years and plot median & 5th-95th percentile range
    for family, rotation, tau, color, hatch, linestyle, linewidth in zip(families, rotations, taus, colors,
                                                                         ('//', r'\\'), ('--', '-.'), (3, 2)):
        if family == 'Mixture':
            families2 = 'Mixture'  # used when calling sample_trivariate_distribution() below
            label = 'Mixture'  # label to use in legend
            if len(tau) == 2:
                label = f'{label}, $\\tau$ ~ U({tau[0]},{tau[1]})'
        else:
            families2 = (family, )*2
            label = family.name.capitalize()
            label = f'{label}, $\\tau$ = {round(tau, 3)}'
        data_df = pd.DataFrame()  # create DataFrame to hold percentile time series for this copula
        # For each year, calculate percentiles of EAIS+WAIS+GrIS
        for year in years:
            x_n3 = sample_trivariate_distribution(projection_source=projection_source, scenario=scenario, year=year,
                                                  families=families2, rotations=(rotation, )*2, taus=(tau, )*2,
                                                  n_samples=n_samples, plot=False)
            for perc in (5, 50, 95):
                data_df.loc[year, perc] = np.percentile(x_n3.sum(axis=1), perc)
        # Plot
        ax.fill_between(data_df.index, data_df[5], data_df[95], label=f'{label} (5th–95th)',  # plot
                        color=color, alpha=0.2, hatch=hatch)
        sns.lineplot(data_df[50], color=color, label=f'{label} (median)',
                     linestyle=linestyle, linewidth=linewidth, ax=ax)
        # Save percentile time series to list of DataFrames
        data_dfs.append(data_df)
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
                interp_years = np.arange(years[0], years[-1]+1, 1)  # interpolate years
                interp_perc = np.interp(interp_years, data_df.index, data_df[perc])  # interpolate data at percentile
                idx = np.abs(interp_perc - thresh).argmin()  # index closest to threshold
                year_eq_threshs.append(interp_years[idx])  # year closest to threshold
            # Calculate timing difference
            timing_diff = max(year_eq_threshs) - min(year_eq_threshs)
            if timing_diff > 0:
                # Plot line and text showing timing difference
                ax.plot(year_eq_threshs, [thresh, ]*2, color='k', marker='|')
                text_str = f'{timing_diff} yr'
                ax.text(np.mean(year_eq_threshs), thresh+0.07, text_str,
                        horizontalalignment='center', fontsize='large')
                # Plot line showing threshold
                ax.axhline(thresh, alpha=0.3, color='k', linestyle=':')
    # Customize plot
    ax.set_xlim(years[0], years[-1])
    ax.set_xlabel('Year')
    ax.set_ylabel(f'Total ice-sheet contribution, m')
    ax.yaxis.set_minor_locator(plt.MultipleLocator(0.1))
    ax.xaxis.set_minor_locator(plt.MultipleLocator(1))
    ax.tick_params(which='minor', direction='in', color='0.7', bottom=True, top=True, left=True, right=True)
    ax.legend(loc='upper left', fontsize='large')
    return ax


def fig_total_vs_time(projection_source='fusion', scenario='SSP5-8.5', years=np.arange(2020, 2101, 10),
                      families_a=(pv.BicopFamily.gaussian, pv.BicopFamily.indep), taus_a=(1.0, 0.0),
                      colors_a=('darkgreen', 'darkorange'), title_a='Perfect dependence & independence',
                      families_b=(pv.BicopFamily.joe, pv.BicopFamily.clayton), taus_b=(0.5, 0.5),
                      colors_b=('darkred', 'blue'), title_b='Two copula families',
                      thresh_for_timing_diff=(1.4, 0.2), ylim=(-0.2, 2.3), n_samples=int(1e5)):
    """
    Plot figure showing median and 5th-95th percentile range of total ice-sheet contribution (y-axis) vs time (x-axis)
    for different copulas (in two panels, a and b).

    Parameters
    ----------
    projection_source : str
        The projection source for the marginal distributions. Default is 'fusion'.
    scenario : str
        The scenario for the marginal distributions. Default is 'SSP5-8.5'
    years : np.array
        Years for which to plot data. Default is np.arange(2020, 2101, 10).
    families_a and families_b : tuple
        Pair copula families to use for panels (a) and (b).
        Default is (pv.BicopFamily.gaussian, pv.BicopFamily.indep) and (pv.BicopFamily.joe, pv.BicopFamily.clayton).
    taus_a and taus_b: tuple
        Pair copula Kendall's tau values. Default is (1.0, 0.0) and (0.5, 0.5).
    colors_a and colors_b : tuple
        Colors to use when plotting. Default is ('darkgreen', 'darkorange') and ('darkred', 'blue').
    title_a and title_b : str
        Titles for panels (a) and (b).
    thresh_for_timing_diff : tuple, True, or None
        Thresholds to use if demonstrating the difference in timing at the 95th percentile and median.
        If True, select automatically. Default is (1.4, 0.2).
    ylim : tuple
        Limits for y-axis. Default is (-0.2, 2.3).
    n_samples : int
        Number of samples to generate for each copula. Default is int(1e5).

    Returns
    -------
    fig : Figure
    axs : array of Axes
    """
    # Create figure and axes
    fig, axs = plt.subplots(1, 2, figsize=(10, 4), sharey=True, constrained_layout=True)
    # (a)
    ax = axs[0]
    _ = ax_total_vs_time(projection_source=projection_source, scenario=scenario, years=years,
                         families=families_a, rotations=(0, )*2, taus=taus_a, colors=colors_a,
                         thresh_for_timing_diff=thresh_for_timing_diff, n_samples=n_samples, ax=ax)
    ax.set_title(f'(a) {title_a}')
    # (b)
    ax = axs[1]
    _ = ax_total_vs_time(projection_source=projection_source, scenario=scenario, years=years,
                         families=families_b, rotations=(0, )*2, taus=taus_b, colors=colors_b,
                         thresh_for_timing_diff=thresh_for_timing_diff, n_samples=n_samples, ax=ax)
    ax.set_title(f'(b) {title_b}')
    ax.set_ylabel(None)
    ax.set_ylim(ylim)
    return fig, axs


# Influence of GRD fingerprints

def ax_sum_vs_gris_fingerprint(projection_source='fusion', scenario='SSP5-8.5', year=2100,
                               families=(pv.BicopFamily.gaussian, pv.BicopFamily.indep),
                               rotations=(0, 0), taus=(1.0, 0.0), colors=('darkgreen', 'darkorange'),
                               n_samples=int(1e5), ax=None):
    """
    Plot median and 5th-95th percentile range of total ice-sheet contribution (y-axis) vs GrIS GRD fingerprint (x-axis).

    Parameters
    ----------
    projection_source : str
        The projection source for the marginal distributions. Default is 'fusion'.
    scenario : str
        The scenario for the marginal distributions. Default is 'SSP5-8.5'
    year : int
        Target year for which to plot data. Default is 2100.
    families : tuple
        Pair copula families. Default is (pv.BicopFamily.gaussian, pv.BicopFamily.indep).
    rotations : tuple
        Pair copula rotations. Default is (0, 0).
    taus: tuple
        Pair copula Kendall's tau values. Default is (1.0, 0.0).
    colors : tuple
        Colors to use when plotting. Default is ('darkgreen', 'darkorange').
    n_samples : int
        Number of samples to generate for each copula. Default is int(1e5).
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
    # For each copula, calculate EAIS+WAIS+GrIS for different GrIS fingerprints and plot median & 5th-95th range
    for family, rotation, tau, color, hatch, linestyle, linewidth in zip(families, rotations, taus, colors,
                                                                         ('//', r'\\'), ('--', '-.'), (3, 2)):
        if family == 'Mixture':
            families2 = 'Mixture'  # used when calling sample_trivariate_distribution() below
            label = 'Mixture'  # label to use in legend
            if len(tau) == 2:
                label = f'{label}, $\\tau$ ~ U({tau[0]},{tau[1]})'
        else:
            families2 = (family, )*2
            label = family.name.capitalize()
            label = f'{label}, $\\tau$ = {tau}'
        # Get trivariate distribution data for global mean (ie fingerprints all 1.0)
        x_n3 = sample_trivariate_distribution(projection_source=projection_source, scenario=scenario, year=year,
                                              families=families2, rotations=(rotation, )*2, taus=(tau, )*2,
                                              n_samples=n_samples, plot=False)
        # Create DataFrame to hold percentile data across GrIS fingerprints
        data_df = pd.DataFrame()
        # Loop over GrIS fingerprints and calculate 5th, 50th, and 95th percentiles of total ice-sheet contribution
        for gris_fp in gris_fp_g:
            for perc in (5, 50, 95):
                sum_n = eais_fp * x_n3[:, 0] + wais_fp * x_n3[:, 1] + gris_fp * x_n3[:, 2]
                data_df.loc[gris_fp, perc] = np.percentile(sum_n, perc)
        # Plot data for this copula
        ax.fill_between(data_df.index, data_df[5], data_df[95], color=color, alpha=0.2, hatch=hatch,
                        label=f'{label} (5th–95th)')
        sns.lineplot(data_df[50], color=color, label=f'{label} (median)', linestyle=linestyle, linewidth=linewidth,
                     ax=ax)
    # Customize plot
    ax.legend(loc='upper left', framealpha=1, fontsize='large')
    ax.set_xlim(gris_fp_g[0], gris_fp_g[-1])
    ax.set_xlabel('Fingerprint of GrIS')
    ax.set_ylabel(f'Total ice-sheet contribution (2005–{year}), m')
    ax.yaxis.set_minor_locator(plt.MultipleLocator(0.1))
    ax.xaxis.set_minor_locator(plt.FixedLocator(gris_fp_g))
    ax.tick_params(which='minor', direction='in', color='0.7', bottom=True, top=True, left=True, right=True)
    ax.set_title('GRD fingerprints influence ice-sheet contribution to RSLC')
    # Annotations
    ax.text(1, -0.15, f'Fingerprint of EAIS = {eais_fp:.2f}\nFingerprint of WAIS = {wais_fp:.2f}',
            transform=ax.transAxes, ha='right', va='bottom')
    ax.set_ylim(ax.get_ylim())  # fix y-axis limits before plotting points near limit
    for gauge, city in [('REYKJAVIK', 'Reykjavik'), ('DUBLIN', 'Dublin'), ('TANJONG_PAGAR', 'Singapore')]:
        gris_fp = read_gauge_grd(gauge=gauge)['GrIS']
        plt.axvline(gris_fp, color='red', linestyle='--', alpha=0.5)
        ax.text(gris_fp, ax.get_ylim()[0]+0.05, city, color='red', fontsize='large',
                ha='right', va='bottom', rotation=90)
    return ax


# Name and save figures

# Counters for figures
f_num = itertools.count(1)  # main figures
e_num = itertools.count(1)  # extended data figures
s_num = itertools.count(1)  # supplementary figures
o_num = itertools.count(1)  # other figures


def name_save_fig(fig, feso='o', exts=('pdf', 'png'), fig_dir=Path('figs_d23b'), close=False):
    """
    Name & save a figure, then increase counter.

    Based on https://github.com/grandey/d22a-mcdc.

    Parameters
    ----------
    fig : Figure
        Figure to save.
    feso : str
        Figure type. Either 'f' (main), 'e' (extended data), 's' (supplement), or 'o' (other; default).
    exts : tuple
        Extensions to use. Default is ('pdf', 'png').
    fig_dir : Path
        Directory in which to save figures. Default is Path('figs_d23b').
    close : bool
        Suppress output in notebook? Default is False.

    Returns
    -------
    fig_name : str
        Name of figure.
    """
    # Name based on counter, then update counter (in preparation for next figure)
    if feso == 'f':
        fig_name = f'fig{next(f_num):02}'
    elif feso == 'e':
        fig_name = f'e{next(e_num):02}'
    elif feso == 's':
        fig_name = f's{next(s_num):02}'
    else:
        fig_name = f'o{next(o_num):02}'
    # File location based on extension(s)
    for ext in exts:
        # Sub-directory
        sub_dir = fig_dir.joinpath(f'{feso}_{ext}')
        sub_dir.mkdir(exist_ok=True)
        # Save
        fig_path = sub_dir.joinpath(f'{fig_name}.{ext}')
        fig.savefig(fig_path, bbox_inches='tight')
        # Print file name and size
        fig_size = fig_path.stat().st_size / 1024 / 1024  # bytes -> MB
        print(f'Written {fig_name}.{ext} ({fig_size:.2f} MB)')
    # Suppress output in notebook?
    if close:
        plt.close()
    return fig_name

