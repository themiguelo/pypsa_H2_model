###### Work in progress #####
# Modeling and Minimization of Levelized Cost of Hydrogen (LCOH) Using PyPSA

This repository supports a master's thesis focused on modeling and minimizing the Levelized Cost of Hydrogen (LCOH) using the open-source power system analysis tool [PyPSA](https://pypsa.org/). The study evaluates how system configurations, electrolyzer loading conditions, and renewable energy integration affect the cost-efficiency of green hydrogen production.

## Overview

The model analyzes various techno-economic scenarios for hydrogen production using Python-based simulations built on PyPSA. The system represents electricity generation from renewables, electrolyzer units, and hydrogen storage. Optimization is performed using the HiGHS solver via the dual simplex method to minimize the LCOH across all scenarios.

## Repository Structure

- `my_pypsa_model/`
  - Contains modular Python scripts for each major step: data import, model setup, scenario creation, optimization, and results analysis.
  - The scripts and functions are executed via Python's `if __name__ == "__main__":` pattern
  - build_model.py: builds six networks for reference scenarios
  - build_model_sensibilities.py: builds 150 networks for sensibilities scenarios
  - analyze_results.py: build graphs and tables displayed in final thesis paper using the scenarios built in build_model.py
  - analyze_results_sensibilities: build graphs and tables displayed in final thesis paper using the scenarios built in build_model_sensibilities.py

- `data/`
  - Holds input datasets for renewable generation profiles, demand, technology parameters, and cost assumptions.

- `backup_models/`
  - Backup 29.03. 10:16

## Goals

This repository is designed to:

- Reproduce and validate thesis results
- Allow future users to extend or adapt the model to new scenarios
- Showcase a practical application of PyPSA for green hydrogen analysis






