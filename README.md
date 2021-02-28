# About

Deep QLearning with tensorflow2 / openai-gym

## Installation (CPU only using mkl optimized builds)

1. Download Conda from [Anaconda Individual Edition:](https://docs.anaconda.com/anaconda/install/mac-os/)

2. Setup an empty base environment in conda using

`conda create -n ${envname} -c conda-forge python==~3.8 mamba`

* This is what this command will do:
  * Add `conda-forge` to the list of approved channels.

  * Set `python 3.8` as the base python for the new environment.
  * Install the alternative [`mamba`](https://github.com/mamba-org/mamba) package manager.

* Add `mkl` to your `.condarc`'s track-features with

`conda config --prepend track_features mkl`

Install the environment requirements using

`mamba install -n ${envname} -f requirements.yaml`

Switch to the environment using

`conda activate ${envname}`
