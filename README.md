# T-wave Alternans Estimator
[![pre-commit](https://github.com/drsantos89/twave-alternans-python/actions/workflows/pre-commit.yaml/badge.svg?branch=main)](https://github.com/drsantos89/twave-alternans-python/actions/workflows/pre-commit.yaml)
[![pytest](https://github.com/drsantos89/twave-alternans-python/actions/workflows/pytest.yaml/badge.svg?branch=main)](https://github.com/drsantos89/twave-alternans-python/actions/workflows/pytest.yaml)
[![PyPI version](https://badge.fury.io/py/twaextractor.svg)](https://pypi.org/project/twaextractor/)

## Spectral Analysis of T-wave Alternans (Kscore)

The Kscore method relies on point-by-point spectral analysis of the T-wave magnitude at a frequency of 0.5 cycle per beat. Initially, each T-wave is measured at specific time form the QRS complex. Then, the fluctuations on the magnitude of each sample in T-wave vector is measured using the fast Fourier transform (FFT), generating a power spectrum for each point in the the T-wave. Finally, the power spectrum are aggregated and the final markers calculated using the two equation mention below:

![image](https://user-images.githubusercontent.com/9551412/135170075-7267bf92-079d-40e6-9084-3b7f7e38a1c6.png)

Where, `P` corresponded to the accumulated amplitude of the power spectrum, usually at a `f=0.5Hz`. `P_{N}` and `\sigma_{P_{N}}` corresponds respectively to the mean and standard deviation of the power spectrum at the noise band, usually set at `f=[0.43:0.49]Hz`. The Kscore measures the statistical significance of the alternans in relation to the noise levels while KTWAM is a direct measure of the TWA magnitude. A visual representation of the algorithm is shown in the figure below.

![image](https://user-images.githubusercontent.com/9551412/135169912-7b55fe98-3615-4ede-b6b4-8a1ff4684fe7.png)
