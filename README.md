 # neurothesis, a Thesis Repository
This repository gathers the scripts used to conduct data analysis for my Bachelor Thesis Research (BTR) titled **"Unraveling the impact of aging on Parkinson’s Disease-derived Neuroepithelial Stem Cells"** in the Maastricht Science Programme (MSP).

By automating various steps and standardizing the analytical process, these scripts ensure that the analysis is reproducible and can be uniformly applied to all samples.

## Project Overview
It primarily involves MMqPCR analysis to assess the telomere length based on the Cq values. Two scripts were made based from two differnt protocols: Cawthon (2009) and Martin et al. (2024). The one that was finally used was the one based on Cawthon (2009). Additionally, there is a separate script that plotted the efficiency curves of the primers used. 

The project covers:
- Monochrome Multiplex Quantitative PCR (MMqPCR) analysis of telomere length (TL)
- Image analysis of Immunocytochemistry (ICC) stainings
- Data cleaning, visualization, and statistical analysis

## Data 
Raw data is not included in this repository due to file size and privacy restrictions. A small test file is available, and the data can be requested if needed. 

### a. MMqPCR
For the MMqPCR analysis, the data needs to be presented in a CSV file and follow the following format: 

| Column Name     | Description                                    |
|:----------------|:-----------------------------------------------|
| Sample Name     | Name of the biological sample (e.g., A13 P9)   |
| Cq_Telomere     | Cq value for the telomere amplification        |
| Cq_SCG          | Cq value for the single-copy gene amplification|
| Well (optional) | qPCR plate well position (e.g., C08)           |

Additionally, the first row was where the reference was put. This was  labeled "Reference" in the "Sample Name" column for 2^(-ΔΔCt) calculations.

The well number is ignored by the script but I kept it to keep track. 

### b. ICC 
The files analysed where .nd2/.tiff files

## Requirements
The Scripts were written in Python 3.12.5 on VSCode 
- Python 3.12.5
- Install required libraries with 'pip install -r requirements.txt'