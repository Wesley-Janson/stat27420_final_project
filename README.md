# STAT 27420 Final Project
## Wesley Janson and Drew Keller

This repository contains relevant files for our final project for STAT 27420: Introduction to Causality for Machine Learning. This project attempts to supplement and update "Inflation Expectations and Readiness to Spend: Cross-Sectional Evidence" by Bachmann, Berg, and Sims.

* **STAT_27420_Final_Paper.pdf**: PDF of final paper.

* **models.ipynb**: This is the main Jupyter notebook used to run the XGBoost and other corresponding models used in our paper.

* **data_utils.py**: Python script that contains helper functions used in data cleaning.

* **load_data.py**: Python script used to pull data available online, create relevant variables, and merge into main Michigan Survey of Consumers data.

* **parameters.py**: Python file denoting variable groups (confounders etc.), variable types (continuous, categorical), used in data cleaning.

* **prepare_survey_data.ipynb**: 

* **replication_test.ipynb**: 

* **data_attributes.csv**:

* **variable_analysis.xlsx**: Excel file used to keep track of causal identification for potential confounders/mediators/colliders etc.

## Subdirectories
* **figures**: Contains the figures used in the paper. Both model feature importance and cDAG figues.

* **literature**: This folder contains the MSC data codebook, along with Bachmann et al's paper and appendix, for reference.