"""
d23b:
    Functions that support the analysis contained in the d23b-ice-dependence repository.

Author:
    Benjamin S. Grandey, 2022-2023.
"""


from functools import cache
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path
import pyvinecopulib as pv
from scipy import stats
import seaborn as sns
import warnings
import xarray as xr


# Combined Antarctic ISM ensemble

@cache
def read_p21_l23_ism_data(ref_year=2015, target_year=2100):
    """
    Read combined Antarctic ISM ensemble data from Payne et al. (2021) and Li et al. (2023).

    This function uses data from https://doi.org/10.5281/zenodo.4498331 and https://doi.org/10.5281/zenodo.7380180.

    Parameters
    ----------
    ref_year : int
        Reference year. Default is 2015 (which is the start year for Payne et al. data).
    target_year : int
        Target year for difference. Default is 2100.

    Returns
    -------
    p21_l23_df : pandas DataFrame
        A DataFrame containing WAIS and EAIS sea-level equivalents (in m), Group (P21_ISMIP6 or L23_MICI), and Notes.
    """
    # DataFrame to hold data
    p21_l23_df = pd.DataFrame(columns=['WAIS', 'EAIS', 'Group', 'Notes'])

    # Read Payne et al. data
    # Location of data
    payne_base = Path(f'data/CMIP5_CMIP6_Scalars_Paper')
    in_dir = payne_base / 'AIS' / 'Ice'
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
            ais_dict = {'Group': f'P21_ISMIP6'}
            # Get ice-sheet model institute and name
            ais_dict['Notes'] = '_'.join(in_fn.name.split('_')[-3:-1])
            # Read DataSet
            in_ds = xr.load_dataset(in_fn)
            # Calculate SLE for target year relative to reference year for WAIS and EAIS
            wais_da = in_ds[f'limnsw_region_{1}'] + in_ds[f'limnsw_region_{3}']  # include peninsula in WAIS
            eais_da = in_ds[f'limnsw_region_{2}']
            for region_name, in_da in [('WAIS', wais_da), ('EAIS', eais_da)]:
                if ref_year == 2015:
                    ais_dict[region_name] = float(in_da.sel(time=target_year)) * convert_Gt_m
                else:
                    ais_dict[region_name] = float(in_da.sel(time=target_year) - in_da.sel(time=ref_year)) * convert_Gt_m
            # Append to DataFrame
            p21_l23_df.loc[len(p21_l23_df)] = ais_dict

    # Read Li et al. data
    # Location of data
    li_base = Path(f'data')
    # Lists containing experiments of interest and CMIP6 GCMs
    exp_dict = {'CMIP6_BC_1850-2100': 'L23_MICI'}
    gcm_list = sorted([g.name for g in (li_base/'CMIP6_BC_1850-2100').glob('*') if g.is_dir()])
    # Loop over experiments
    for exp, group in exp_dict.items():
        # Loop over GCMs
        for gcm in gcm_list:
            # Create dictionary to hold data for this input file
            ais_dict = {'Group': group}
            # Get ice-sheet model institute and name
            ais_dict['Notes'] = f'{exp} {gcm}'
            # Read data
            in_fn = li_base / exp / gcm / 'fort.22'
            try:
                in_df = pd.read_fwf(in_fn, skiprows=1, index_col='time')
            except ValueError:
                in_df = pd.read_fwf(in_fn, skiprows=2, index_col='time')
            # Get SLE for target year relative to reference year for WAIS and EAIS
            for region_name, in_varname in [('WAIS', 'eofw(m)'), ('EAIS', 'eofe(m)')]:
                ais_dict[region_name] = in_df.loc[ref_year][in_varname] - in_df.loc[target_year][in_varname]
            # Append to DataFrame
            p21_l23_df.loc[len(p21_l23_df)] = ais_dict

    # Return result
    return p21_l23_df


def fig_p21_l23_ism_data(ref_year=2015, target_year=2100):
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
    bicop : pv.Bicop
        Fitted copula.
    fig : Figure
    axs : array of Axes
    """
    # Read combined Antarctic ISM ensemble data from Payne et al. (2021) and Li et al. (2023)
    p21_l23_df = read_p21_l23_ism_data(ref_year=ref_year, target_year=target_year)
    # Create Figure and Axes
    fig, axs = plt.subplots(1, 2, figsize=(8, 4), tight_layout=True)
    # (a) WAIS vs EAIS on GMSLR scale (ie sea-level equivalent)
    ax = axs[0]
    sns.scatterplot(p21_l23_df, x='EAIS', y='WAIS', hue='Group', style='Group', ax=ax)
    ax.legend(loc='lower right', framealpha=1, edgecolor='0.85')  # specify edgecolor consistent with box in (b)
    ax.set_title(f'(a) Contribution to GMSLR')
    ax.set_xlabel('EAIS, m')
    ax.set_ylabel('WAIS, m')
    # (b) Pseudo-copula data on copula scale
    ax = axs[1]
    x_n2 = np.stack([p21_l23_df['EAIS'], p21_l23_df['WAIS']], axis=1)
    u_n2 = pv.to_pseudo_obs(x_n2)
    u_df = pd.DataFrame({'EAIS': u_n2[:, 0], 'WAIS': u_n2[:, 1], 'Group': p21_l23_df['Group']})
    sns.scatterplot(u_df, x='EAIS', y='WAIS', hue='Group', style='Group', legend=False, ax=ax)
    ax.set_title(f'(b) Pseudo-copula data')
    ax.set_xlabel('EAIS')
    ax.set_ylabel('WAIS')
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])
    # Fit copula (limited to single-parameter families)
    controls = pv.FitControlsBicop(family_set=[pv.BicopFamily.indep, pv.BicopFamily.joe, pv.BicopFamily.gumbel,
                                               pv.BicopFamily.gaussian, pv.BicopFamily.frank, pv.BicopFamily.clayton])
    bicop = pv.Bicop(data=u_n2, controls=controls)  # fit
    ax.text(0.8, 0.04, f'Best fit: {bicop.family.name.capitalize()}\nwith $\\tau$ = {bicop.tau:.3f}',
            ha='center', va='bottom', bbox=dict(boxstyle='square,pad=0.5', fc='1', ec='0.85'))
    return bicop, fig, axs


# Marginals of ice-sheet component projections

@cache
def read_sea_level_qf(projection_source='fusion', component='total', scenario='SSP5-8.5', year=2100):
    """
    Read quantile function corresponding to sea-level projection (either AR6 ISMIP6, AR6 SEJ, p-box, or fusion).

    This function follows fusion_analysis_d23a.ipynb and uses data from
    https://doi.org/10.5281/zenodo.6382554.

    Parameters
    ----------
    projection_source : str
        Projection source. Options are 'ISMIP6'/'model-based' (emulated ISMIP6),
        'SEJ'/'expert-based' (Bamber et al. structured expert judgment),
        'p-box'/'bounding quantile function' (p-box bounding quantile function of ISMIP6 & SEJ),
        and 'fusion' (fusion of ISMIP6 and bounding quantile function, weighted using triangular function; default).
    component : str
        Component of global sea level change. Options are 'GrIS' (Greenland Ice Sheet),
        'EAIS' (East Antarctic Ice Sheet), 'WAIS' (West Antarctic Ice Sheet), and
        'total' (total global-mean sea level; default).
        Note: for ISMIP6, 'PEN' is also included in 'WAIS'.
    scenario : str
        Scenario. Options are 'SSP1-2.6' and 'SSP5-8.5' (default).
    year : int
        Year. Default is 2100.

    Returns
    -------
    qf_da : xarray DataArray
        DataArray of sea level change in metres for different quantiles.
    """
    # Case 1: p-box or fusion
    if projection_source in ['p-box', 'bounding quantile function', 'fusion']:
        # If p-box bounding quantile function...
        if projection_source in ['p-box', 'bounding quantile function']:
            # Call function recursively to get ISMIP6 and SEJ data
            ism_da = read_sea_level_qf(projection_source='ISMIP6', component=component, scenario=scenario, year=year)
            sej_da = read_sea_level_qf(projection_source='SEJ', component=component, scenario=scenario, year=year)
            # Initialize qf_da as copy of ism_da
            qf_da = ism_da.copy()
            # Loop over quantile probabilities
            for q, quantile in enumerate(qf_da.quantiles):
                # Get data for this quantile probability
                ism = ism_da[q].data
                sej = sej_da[q].data
                if quantile == 0.5:  # if median, use mean of ISMIP6 and SEJ
                    qf_da[q] = (ism + sej) / 2
                elif quantile < 0.5:  # if quantile < 0.5, use min
                    qf_da[q] = np.minimum(ism, sej)
                else:  # if > 0.5, use maximum
                    qf_da[q] = np.maximum(ism, sej)
        # If fusion...
        elif projection_source == 'fusion':
            # Call function recursively to get ISMIP6 and p-box bounding quantile function data
            ism_da = read_sea_level_qf(projection_source='ISMIP6', component=component, scenario=scenario, year=year)
            pbox_da = read_sea_level_qf(projection_source='p-box', component=component, scenario=scenario, year=year)
            # Weights for ISMIP6 emulator data: triangular function, with peak at median
            weights_q = 1 - np.abs(ism_da.quantiles - 0.5) * 2  # _q indicates quantile probability dimension
            # Combine ISMIP6 and bounding quantile function data using weights; rely on automatic broadcasting/alignment
            qf_da = weights_q * ism_da + (1 - weights_q) * pbox_da
            # Copy units attribute
            qf_da.attrs['units'] = ism_da.attrs['units']
        # Is result monotonic?
        if np.any((qf_da[1:].data - qf_da[:-1].data) < 0):  # allow difference to equal zero
            warnings.warn(f'read_sea_level_qf{projection_source, component, scenario, year} result not monotonic.')
        # Return result (Case 1)
        return qf_da

    # Case 2: ISMIP6 or SEJ
    # Check projection_source argument and identify corresponding projection_code and workflow_code
    elif projection_source in ['ISMIP6', 'model-based']:  # emulated ISMIP6
        projection_code = 'ismipemu'  # projection_code for individual ice-sheet components
        workflow_code = 'wf_1e'  # workflow_code for total GMSL
    elif projection_source in ['SEJ', 'expert-based']:  # Bamber et al structured expert judgment
        projection_code = 'bamber'
        workflow_code = 'wf_4'
    else:
        raise ValueError(f'Unrecognized argument value: projection_source={projection_source}')
    # Check component argument
    if component not in ['EAIS', 'WAIS', 'PEN', 'GrIS', 'total']:
        raise ValueError(f'Unrecognized argument value: component={component}')
    # Check scenario argument and identify corresponding scenario_code
    if scenario in ['SSP1-2.6', 'SSP5-8.5']:
        scenario_code = scenario.replace('-', '').replace('.', '').lower()
    else:
        raise ValueError(f'Unrecognized argument value: scenario={scenario}')
    # Input directory and file
    if component == 'total':
        in_dir = Path(f'data/ar6/global/dist_workflows/{workflow_code}/{scenario_code}')
        in_fn = in_dir / 'total-workflow.nc'
    else:
        in_dir = Path(f'data/ar6/global/dist_components')
        if component == 'GrIS':
            in_fn = in_dir / f'icesheets-ipccar6-{projection_code}icesheet-{scenario_code}_GIS_globalsl.nc'
        else:
            in_fn = in_dir / f'icesheets-ipccar6-{projection_code}icesheet-{scenario_code}_{component}_globalsl.nc'
    # Does input file exist?
    if not in_fn.exists():
        raise FileNotFoundError(in_fn)
    # Read data
    qf_da = xr.open_dataset(in_fn)['sea_level_change'].squeeze().drop_vars('locations').sel(years=year)
    # Convert units from mm to m
    if qf_da.attrs['units'] == 'mm':
        qf_da *= 1e-3
        qf_da.attrs['units'] = 'm'
    # If ISMIP6 WAIS, also include PEN (assuming perfect dependence)
    if projection_code == 'ismipemu' and component == 'WAIS':
        qf_da += read_sea_level_qf(projection_source='ISMIP6', component='PEN', scenario=scenario, year=year)
        print(f'read_sea_level_qf{projection_source, component, scenario, year}: including PEN in WAIS.')
    # Is result monotonic?
    if np.any((qf_da[1:].data - qf_da[:-1].data) < 0):
        warnings.warn(f'read_sea_level_qf{projection_source, component, scenario, year} result not monotonic.')
    # Return result (Case 2)
    return qf_da


@cache
def sample_sea_level_marginal(projection_source='fusion', component='total', scenario='SSP5-8.5', year=2100,
                              n_samples=int(1e5), plot=False):
    """
    Sample marginal distribution corresponding to sea-level projection (either AR6 ISMIP6, AR6 SEJ, p-box, or fusion).

    This function follows fusion_analysis_d23a.ipynb.

    Parameters
    ----------
    projection_source : str
        Projection source. Options are 'ISMIP6'/'model-based' (emulated ISMIP6),
        'SEJ'/'expert-based' (Bamber et al. structured expert judgment),
        'p-box'/'bounding quantile function' (p-box bounding quantile function of ISMIP6 & SEJ),
        and 'fusion' (fusion of ISMIP6 and bounding quantile function, weighted using triangular function; default).
    component : str
        Component of global sea level change. Options are 'GrIS' (Greenland Ice Sheet),
        'EAIS' (East Antarctic Ice Sheet), 'WAIS' (West Antarctic Ice Sheet), and
        'total' (total global-mean sea level; default).
        Note: for ISMIP6, 'PEN' is also included in 'WAIS'.
    scenario : str
        Scenario. Options are 'SSP1-2.6' and 'SSP5-8.5' (default).
    year : int
        Year. Default is 2100.
    n_samples : int
        Number of samples. Default is int(1e5).
    plot : bool
        If True, plot diagnostic ECDF and histogram. Default is False.

    Returns
    -------
    marginal_n : numpy array
        A one-dimensional array of randomly drawn samples.
    """
    # Read quantile function data
    qf_da = read_sea_level_qf(projection_source=projection_source, component=component, scenario=scenario, year=year)
    # Sample uniform distribution
    rng = np.random.default_rng(12345)
    uniform_n = rng.uniform(size=n_samples)
    # Transform these samples to marginal distribution samples by interpolation of quantile function
    marginal_n = qf_da.interp(quantiles=uniform_n).data
    # Plot diagnostic plots?
    if plot:
        fig, axs = plt.subplots(1, 2, figsize=(8, 4), constrained_layout=True)
        sns.ecdfplot(marginal_n, label='marginal_n ECDF', ax=axs[0])
        axs[0].plot(qf_da, qf_da['quantiles'], label='"True" quantile function', linestyle='--')
        sns.histplot(marginal_n, bins=100, label='marginal_n', ax=axs[1])
        for ax in axs:
            ax.legend()
            try:
                ax.set_xlabel(f'{component}, {qf_da.attrs["units"]}')
            except KeyError:
                ax.set_xlabel(component)
        plt.suptitle(f'{projection_source}, {component}, {scenario}, {year}, {n_samples}')
        plt.show()
    return marginal_n


def fig_ice_sheet_marginals(projection_source='fusion', scenario='SSP5-8.5', year=2100,
                            components=('EAIS', 'WAIS', 'GrIS'), n_samples=int(1e5)):
    """
    Plot figure showing marginals for specific ice-sheet components.

    Parameters
    ----------
    projection_source : str
        Projection source. Options are 'ISMIP6'/'model-based',  'SEJ'/'expert-based',
        'p-box'/'bounding quantile function', and 'fusion' (default).
    scenario : str
        Scenario. Options are 'SSP1-2.6' and 'SSP5-8.5' (default).
    year : int
        Year. Default is 2100.
    components : tuple
        Ice-sheet components. Default is ('EAIS', 'WAIS', 'GrIS').
    n_samples : int
        Number of samples. Default is int(1e5).

    Returns
    -------
    fig : Figure
    axs : array of Axes
    """
    # Create Figure and Axes
    n_axs = len(components)  # number of subplots = number of components
    fig, axs = plt.subplots(n_axs, 1, figsize=(5, 2.1*n_axs), sharex=True, sharey=True, tight_layout=True)
    # Loop over components and Axes
    for i, (component, ax) in enumerate(zip(components, axs)):
        # Get marginal samples
        marginal_n = sample_sea_level_marginal(projection_source=projection_source, component=component,
                                               scenario=scenario, year=year, n_samples=n_samples, plot=False)
        marginal_df = pd.DataFrame(marginal_n, columns=[component])
        # Plot KDE
        sns.kdeplot(marginal_df, x=component, bw_adjust=0.2, color='b', fill=True, cut=0, ax=ax)  # limit to data limits
        # Plot 5th, 50th, and 95th percentiles, using original quantile function (not samples)
        qf_da = read_sea_level_qf(projection_source=projection_source, component=component,
                                  scenario=scenario, year=year)
        y_pos = 12.3  # position of percentile whiskers is tuned for the default parameters
        ax.plot([qf_da.sel(quantiles=p) for p in (0.05, 0.95)], [y_pos, y_pos], color='g', marker='|')
        ax.plot([qf_da.sel(quantiles=0.5), ], [y_pos, ], color='g', marker='x')
        if i == (n_axs-1):  # label percentiles in final subplot
            for p in [0.05, 0.5, 0.95]:
                ax.text(qf_da.sel(quantiles=p), y_pos-0.4, f'{int(p*100)}th',
                        ha='center', va='top', color='g', rotation=90)
        # Skewness and kurtosis
        ax.text(1.9, 6.5,  # position tuned for the default parameters
                f"Skewness = {stats.skew(marginal_n):.2f}\n"
                f"Fisher's kurtosis = {stats.kurtosis(marginal_n, fisher=True):0.2f}",
                ha='right', va='center', fontsize='medium', bbox=dict(boxstyle='square,pad=0.5', fc='1', ec='0.85'))
        # Title etc
        ax.set_title(f'({chr(97+i)}) {component}')
    # x-axis label and limits
    axs[-1].set_xlabel(f'Contribution to GMSLR (2005–{year}), m')
    axs[-1].set_xlim([-0.5, 2.0])
    return fig, axs


# Sample copulas and joint distributions

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


@cache
def sample_dvine_copula(families=(pv.BicopFamily.gaussian, pv.BicopFamily.gaussian),
                        rotations=(0, 0), taus=(0.5, 0.5), n_samples=int(1e5), plot=False):
    """
    Sample truncated D-vine copula with given families, rotations, Kendall's tau values, and number of samples.

    Parameters
    ----------
    families : tuple of pv.BicopFamily
        Pair copula families. Default is (pv.BicopFamily.gaussian, pv.BicopFamily.gaussian).
    rotations : tuple of int
        Pair copula rotations. Ignored for Independence, Gaussian and Frank copulas. Default is (0, 0).
    taus : tuple of float
        Pair copula Kendall's tau values. Default is (0.5, 0.5).
    n_samples : int
        Number of samples to generate. Default is int(1e5).
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
    # Derive parameters and create bivariate/pair copulas
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
    cop = pv.Vinecop(struc, [bicops,])
    # Simulate data
    u_nm = cop.simulate(n=n_samples, seeds=[1, 2, 3, 4, 5])
    # Plot?
    if plot:
        sns.pairplot(pd.DataFrame(u_nm, columns=[f'u{n+1}' for n in range(len(families)+1)]), kind='hist')
        plt.suptitle(f'{cop.str()}', y=1.05)
        plt.show()
    return u_nm


@cache
def sample_mixture_dvine_copula(taus=((0, 1), (0, 1)), n_copulas=1000, n_samples=int(1e5), plot=False):
    """
    Sample a mixture of truncated D-vine copulas across 1-parameter families and Kendall's tau.

    Parameters
    ----------
    taus : tuple containing floats and/or tuples
        Pair copula Kendall's tau values. If float, tau is deterministic. If tuple of length 2, tau ~ U(a, b).
        If of length 4, tau ~ TN(μ, σ, clip_a, clip_b). Default is ((0, 1), (0, 1)).
    n_copulas : int
        Number of copulas to sample. Default is 1000.
    n_samples : int
        Number of samples to generate. Must be multiple of n_copulas. Default is int(1e5).
    plot : bool
        Plot the simulated data? Default is False.

    Returns
    -------
    u_nm : np.array
        An array of the simulated data, with shape (n_samples, len(taus)+1).
    """
    # Is n_samples a multiple of n_copulas?
    if n_samples % n_copulas != 0:
        raise ValueError(f'n_samples={n_samples} is not a multiple of n_copulas={n_copulas}.')
    # Initialize random number generator
    rng = np.random.default_rng(12345)
    # Sample tau; c represents index corresponding to n_copulas
    tau_c_list = []  # list to hold arrays of tau values
    for tau in taus:  # sample each pair copula independently
        if type(tau) in [float, np.float64, int]:
            tau_c = np.full(n_copulas, tau)
        elif type(tau) == tuple and len(tau) == 2:  # U(a, b)
            tau_c = rng.uniform(low=tau[0], high=tau[1], size=n_copulas)
        elif type(tau) == tuple and len(tau) == 4:  # TN(μ, σ, clip_a, clip_b)
            loc, scale, clip_a, clip_b = tau
            a = (clip_a - loc) / scale  # scipy.stats doc: [a, b] defined wrt standard normal
            b = (clip_b - loc) / scale
            tau_c = stats.truncnorm.rvs(a=a, b=b, loc=loc, scale=scale, size=n_copulas, random_state=rng)
        else:
            raise ValueError(f'tau={tau} is not a float or tuple of length 2 or 4.')
        tau_c_list.append(tau_c)
    # Sample copula families
    possible_families = [pv.BicopFamily.joe,
                         pv.BicopFamily.gumbel,
                         pv.BicopFamily.gaussian,
                         pv.BicopFamily.frank,
                         pv.BicopFamily.clayton]
    family_c_list = []  # list to hold arrays of families
    for i in range(len(taus)):  # sample each pair copula independently
        family_c = rng.choice(possible_families, size=n_copulas)
        family_c_list.append(family_c)
    # Simulate data
    u_nm = np.full((n_samples, len(taus)+1), np.nan)  # array to hold simulated data
    n_spc = n_samples // n_copulas  # number of samples per copula
    struc = pv.DVineStructure(np.arange(len(taus)+1)+1, trunc_lvl=1)  # D-vine structure
    for c in range(n_copulas):
        bicops = []  # list to hold pair copulas
        for family, tau in zip([family_c[c] for family_c in family_c_list],
                               [tau_c[c] for tau_c in tau_c_list]):
            parameters = pv.Bicop(family=family).tau_to_parameters(tau)
            bicop = pv.Bicop(family=family, parameters=parameters)
            bicops.append(bicop)
        cop = pv.Vinecop(struc, [bicops, ])  # create vine copula
        u_nm[c*n_spc:(c+1)*n_spc, :] = cop.simulate(n=n_spc, seeds=[c+1, c+2, c+3, c+4, c+5])  # simulate, varying seed
    # Plot?
    if plot:
        sns.pairplot(pd.DataFrame(u_nm, columns=[f'u{n+1}' for n in range(len(taus)+1)]), kind='hist')
        plt.suptitle('Mixture', y=1.01)
        plt.show()
    return u_nm


@cache
def sample_trivariate_distribution(projection_source='fusion', scenario='SSP5-8.5', year=2100,
                                   families=(pv.BicopFamily.gaussian, pv.BicopFamily.gaussian),
                                   rotations=(0, 0), taus=(0.5, 0.5),
                                   n_samples=int(1e5), plot=False):
    """
    Sample EAIS-WAIS-GrIS joint distribution.

    Parameters
    ----------
    projection_source : str
        Projection source. Options are 'ISMIP6'/'model-based',  'SEJ'/'expert-based',
        'p-box'/'bounding quantile function', and 'fusion' (default).
    scenario : str
        Scenario. Options are 'SSP1-2.6' and 'SSP5-8.5' (default).
    year : int
        Year. Default is 2100.
    families : 'Mixture' or tuple of pv.BicopFamily
        Pair copula families. Default is (pv.BicopFamily.gaussian, pv.BicopFamily.gaussian).
    rotations : tuple of int
        Pair copula rotations. Ignored if family is 'Mixture'. Default is (0, 0).
    taus : tuple of float or tuple
        Pair copula Kendall's tau values. If float, tau is deterministic. If tuple of length 2, tau ~ U(a, b).
        If of length 4, tau ~ TN(μ, σ, clip_a, clip_b). Tuple only valid if families is 'Mixture'.
        Default is (0.5, 0.5).
    n_samples : int
        Number of samples. Default is int(1e5).
    plot : bool
        Plot the joint distribution? Default is False.

    Returns
    -------
    x_n3 : np.array
        An array of shape (n_samples, 3), containing the samples from the joint distribution.
    """
    # Sample marginals of EAIS, WAIS, GrIS components
    components = ['EAIS', 'WAIS', 'GrIS']
    marginals = []  # empty list to hold samples for the marginals
    for component in components:
        marginal_n = sample_sea_level_marginal(projection_source=projection_source, component=component,
                                               scenario=scenario, year=year, n_samples=n_samples, plot=False)
        marginals.append(marginal_n)
    marg_n3 = np.stack(marginals, axis=1)  # marginal array with shape (n_samples, 3)
    # Sample copula
    if families == 'Mixture':
        u_n3 = sample_mixture_dvine_copula(taus=taus, n_copulas=1000, n_samples=n_samples, plot=False)
    else:
        u_n3 = sample_dvine_copula(families=families, rotations=rotations, taus=taus, n_samples=n_samples, plot=False)
    # Transform marginals of copula
    x_n3 = np.transpose(np.asarray([np.quantile(marg_n3[:, i], u_n3[:, i]) for i in range(3)]))
    # Plot?
    if plot:
        sns.pairplot(pd.DataFrame(x_n3, columns=components), kind='hist')
        plt.show()
    return x_n3


# Figure illustrating bivariate distribution, bivariate copula, and truncated vine copula

@cache
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
    axs : array of Axes
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
        ax4.imshow(plt.imread(temp_dir / 'temp_copula2.png'), extent=[x-0.9,x+0.9,0,1.8])
    ax4.set_title('(c) Truncated vine copula', fontsize='x-large')
