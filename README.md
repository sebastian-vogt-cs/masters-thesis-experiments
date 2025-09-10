# Master's Thesis Experiments
This project contains the code for the numerical experiments in my master's thesis.
It is structured as follows:

## Python modules
- `algorithms/vkabc.py`: Contains the implementation of the VKABC and KABC algorithms from my master's thesis. Computation-heavy tasks are calculated using multiple processes in parallel.

- `drawing/bandit_drawer.py`: Contains functions that draw and save the figures in my thesis.

- `model/arm.py`: Provides classes modelling an arm of a multi-armed bandit.

## Experiment Scripts
- `multimodal_experiment.py`: Runs the multimodal experiment. The recorded data is pickled and saved to `data/`
- `same_mean_experiment.py`: Runs the same-mean experiment. The recorded data is pickled and saved to `data/`

## Jupyter Notebooks
- `multimodal_experiment.ipynb`: Loads the data from the multimodal experiment from the pickles, creates the figures for the thesis and saves them to `data/`
- `same_mean_experiment.ipynb`: Loads the data from the same mean experiment from the pickles, creates the figures for the thesis and saves them to `data/`

## Data
The pickles of the experiment data and the figures for the thesis are saved to `data/`.