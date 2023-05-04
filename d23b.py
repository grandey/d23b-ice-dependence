"""
d23b:
    Functions that support the analysis contained in the d23b-ice-dependence repository.

Author:
    Benjamin S. Grandey, 2022-2023.
"""


from functools import cache
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path
import seaborn as sns
import warnings
import xarray as xr


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
                              n_samples=int(1e6), plot=False):
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
        Number of samples. Default is int(1e6).
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
                'B10']  #CNRM-ESM2-1 SSP5-8.5, standard protocol
    # Loop over experiments
    for exp in exp_list:
        # Loop over available input files
        in_fns = sorted(in_dir.glob(f'computed_limnsw_minus_ctrl_proj_AIS_*_exp{exp}.nc'))
        for in_fn in in_fns:
            # Create dictionary to hold data for this input file
            ais_dict = {'Group': f'P21_ISMIP6'}
            # Get ice-sheet model institute and name
            ais_dict['Notes'] = ('_').join(in_fn.name.split('_')[-3:-1])
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
