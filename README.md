# neurothesis
markdown
# Thesis Repository: Analysis of Telomere Dynamics

This repository contains all the Python scripts and analysis tools used for my Master's Thesis titled **"Analysis of Telomere Dynamics in Human Stem Cells."**

## Project Overview

The project covers:
- Quantitative PCR analysis of telomere length (MMqPCR)
- Image analysis of ICC staining
- Data cleaning, visualization, and statistical analysis

## Requirements
- Python 3.12.5
- Install required libraries: pip install -r requirements.txt


1. What is this?
This repository gathers the scripts I employed to conduct data analysis for my Bachelor Thesis Research (BTR) titled "Unraveling the impact of aging on Parkinson’s Disease-derived Neuroepithelial Stem Cells" in the Maastricht Science Programme (MSP). By automating various steps and standardizing the analytical process, these scripts ensure that the analysis is reproducible and can be uniformly applied to all samples.

2. Topics 
It primarily involves MMqPCR analysis to assess the telomere length based on the Cq values. Two scripts were made based from two differnt protocols: Cawthon (2009) and Martin et al. (2024). The one that was finally used was the one based on Cawthon (2009). Additionally, there is a separate script that plotted the efficiency curves of the primers used. 

3. How to run the scripts
The Scripts were written in Python 3.12.5. And the required libraries can be installed using 'pip install -r requirements.txt'. 

4. Data
For the MMqPCR analysis, the data needs to be presented in a CSV file and follow the following format: 

| Column Name     | Description                                    |
|:----------------|:-----------------------------------------------|
| Sample Name     | Name of the biological sample (e.g., A13 P9)   |
| Cq_Telomere     | Cq value for the telomere amplification        |
| Cq_SCG          | Cq value for the single-copy gene amplification|
| Well (optional) | qPCR plate well position (e.g., C08)           |

Additionally, the first row was where the reference was put. This was  labeled "Reference" in the "Sample Name" column for 2^(-ΔΔCt) calculations.

The well number is ignored by the script but I kept it to keep track. 

5. Where is the data?
Raw data is not included in this repository due to file size and privacy restrictions. A small test file is available, and the data can be requested if needed. 



