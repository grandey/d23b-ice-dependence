# Analysis code for _Modelling Correlation Between the Ice Sheet Components of Sea-Level Rise_ (d23b-ice-dependence)

## Usage guidelines
This repository accompanies the following manuscript:

B. S. Grandey et al. (2024),  **Modelling Correlation Between the Ice Sheet Components of Sea-Level Rise**, _in preparation_.

The manuscript serves as the primary reference.
The Zenodo archive of this repository will serve as a secondary reference.

## Workflow

### Environment
To create a _conda_ environment with the necessary software dependencies, use the [**`environment.yml`**](environment.yml) file:

```
conda env create --file environment.yml
conda activate d23b-ice-dependence
```

The analysis has been performed within this environment on _macOS 13_ (arm64).

### Input data
The analysis code requires
(i) the global-mean sea-level data samples and tide gauge location data from the [IPCC AR6 Sea Level Projections](https://doi.org/10.5281/zenodo.6382554),
(ii) the "CMIP5_CMIP6_Scalars_Paper" ISM data from [Payne et al. (2021)](https://doi.org/10.5281/zenodo.4498331),
(iii) the "CMIP6_BC_1850-2100" ISM data from [Li et al. (2023)](https://doi.org/10.5281/zenodo.7380180), and
(iv) the GRD fingerprints from the [FACTS module data](https://doi.org/10.5281/zenodo.7478192).
These can be downloaded as follows:

```
mkdir data
cd data
curl -Z "https://zenodo.org/records/6382554/files/ar6.zip?download=1" -O "https://zenodo.org/records/6382554/files/location_list.lst?download=1" -O "https://zenodo.org/records/4498331/files/CMIP5_CMIP6_Scalars_Paper.zip?download=1" -O "https://zenodo.org/records/7380180/files/CMIP6_BC_1850-2100.tar.gz?download=1" -O "https://zenodo.org/records/7478192/files/grd_fingerprints_data.tgz?download=1" -O
unzip ar6.zip
unzip CMIP5_CMIP6_Scalars_Paper.zip
tar -xvzf CMIP6_BC_1850-2100.tar.gz
mkdir grd_fingerprints_data/
tar -xvzf grd_fingerprints_data.tgz -C grd_fingerprints_data/
cd ..
```

Users of these data should refer to the respective repositories for information about the data, including required acknowledgements and citations.

### Analysis and figures
Analysis is performed using [**`analysis_d23b.ipynb`**](analysis_d23b.ipynb), which uses the functions contained in [**`d23b.py`**](d23b.py) and writes figures to [**`figs_d23b/`**](figs_d23b).

## Author
[Benjamin S. Grandey](https://grandey.github.io) (_Nanyang Technological University_), in collaboration with co-authors.

## Acknowledgements
This Research/Project is supported by the National Research Foundation, Singapore, and National Environment Agency, Singapore under the National Sea Level Programme Funding Initiative (Award No. USS-IF-2020-3).
We thank the projection authors for developing and making the sea level rise projections available, multiple funding agencies for supporting the development of the projections, and the NASA Sea Level Change Team for developing and hosting the IPCC AR6 Sea Level Projection Tool.
We thank the Climate and Cryosphere (CliC) effort, which provided support for ISMIP6 through sponsoring of workshops, hosting the ISMIP6 website and wiki, and promoted ISMIP6.
We acknowledge the World Climate Research Programme, which, through its Working Group on Coupled Modelling, coordinated and promoted CMIP5 and CMIP6.
We thank the climate modeling groups for producing and making available their model output, the Earth System Grid Federation (ESGF) for archiving the CMIP data and providing access, the University at Buffalo for ISMIP6 data distribution and upload, and the multiple funding agencies who support CMIP5 and CMIP6 and ESGF.
We thank the ISMIP6 steering committee, the ISMIP6 model selection group and ISMIP6 dataset preparation group for their continuous engagement in defining ISMIP6.
We thank Li et al. (2023) for publishing the data from their Antarctic ISM simulations.
We thank Kopp et al. (2023) for publishing the FACTS module data.
