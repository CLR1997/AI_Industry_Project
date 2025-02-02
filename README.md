# SMLE: Safe Machine Learning via Embedded Overapproximation

## Description
This repository allows for the reproduction of the findings reported in the manuscript __"SMLE: Safe Machine Learning via Embedded Overapproximation"__, submitted to __AAAI 2025 AI Alignment Track__ and currently under revision.

The paper proposes a particular neural network architecture and a dedicated training algorithm to produce models that, by design, formally guarantee the satisfaction of robustness and safety properties.

## Installation
1. Python: `pip install -r requirements.txt`
2. Gurobi: follow the instructions on https://www.gurobi.com/features/academic-named-user-license/ to download Gurobi Optimizer v11.0.0 and obtain a free academic license

## Project Structure
This repository is structured into two main folders: `core`, containing the implementation of the methodology, and `benchmarks`, containing the implementation of the experiments.

1. `core` contains:
    * `data.py` -- implements data preprocessing for the three benchmarks
    * `property.py` -- implements property generation for the three benchmarks
    * `generate.py` -- implements model training and testing
    * `metrics.py` -- implements metrics to evaluate model accuracy and property difficulty/violation
    * `model.py` -- implements ML models, in particular SMLE and its Robust Training Algorithm (Algorithm 2)
    * `optimization.py` -- implements optimization components, in particular Projection with Delayed Constraint Generation (Algorithm 1) and its subroutines, Counterexample Generation and Weight Projection, and the MAP operator.

2. `benchmarks` consists of a folder for each benchmark: `synthetic`, `forecasting` and `classification`. Each of them contains:
    * `process.py` -- trains the models and produces the logs with the corresponding raw results
    * `postprocess.py` -- retrieves the logs of the models and aggregates the results into plots and/or tables
    * `data`: contains the training/testing data used by `process.py`
    * `results`: stores the raw results produced by `process.py`
    * `aggregated_results`: stores the aggregated results produced by `postprocess.py`
    * `preprocess`: 
      * synthetic: randomly generates linear properties (`properties.pkl`) 
      * forecasting: selects time series (`all.csv`, `selected.csv`)
      * classification: no preprocessing is required

## Usage
The results presented in the paper can be reproduced, for each benchmark, as follows. 

1. Generate properties (Synthetic) / Select series (Forecasting): `python benchmarks/{benchmark}/preprocess/preprocess.py`
2. Generate raw results: `python benchmarks/{benchamrk}/process.py`
3. Generate aggregated results: `python benchmarks/{benchmark}/postprocess.py`

__NOTE__: All aggregated results (plots/tables) reported in the paper can be direclty accessed from `benchmarks/{benchmark}/aggregated_results`, without running any process. However, the raw results for each trained model are not available in `benchmarks/{benchmark}/results`, as they were deleted due to exceeding the size limit set by OpenReview. The only way to obtain these results is by re-generating them through the 3-step pipeline described above.
