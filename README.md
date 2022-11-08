# spclustering: Super Paramagnetic Clustering Wrapper

This is a Python Ctypes wrapper for the original SPC algorithm (available in [Eytan Domany repository](https://github.com/eytandomany/SPC)). The SPC code was edited to work as a shared library, reduce disk usage and improve speed and interface with Python.

## How to install spclustering

```bash
pip install spclustering
```

## How to build spclustering

The original code requires gcc and makefile. For Windows I recommend [TDM-GCC](https://jmeubank.github.io/tdm-gcc/).


## How to use spclustering

```python
from spclustering import SPC
import matplotlib.pyplot as plt

rng = np.random.RandomState(0)

cl1 = rng.multivariate_normal([8, 8], [[4,0],[0,3]], size=400)
cl2 = rng.multivariate_normal([0,0], [[3,0],[0,2]], size=1000)
data = np.concatenate([cl1,cl2])

clustering = SPC(mintemp=0.02,maxtemp=0.04)
results = clustering.run(data)

#select a temperature (similar to select a distance in a dendogram)
temp = 1

plt.plot(*data[results[temp,:]==0,:].T,'.r')
plt.plot(*data[results[temp,:]==1,:].T,'.b')
plt.plot(*data[results[temp,:]>1,:].T,'.k')

```

## Limitations and Changes

- It runs with a data matrix [npoints, ndims] as input. Running the clustering from a distance matrix is not implemented.

- Only the main parameters and flags are implemented.

- Tmax (stop temperature) is not included in the temperature array (similar to Python ```range```). 

- Included *WriteEdges* boolean parameter (defaul False). If True, the \*.edges/\*.mst/\*.K file is created as in the original SPC. Added aswell *PrintParam* parameter in the legacy SW.c code.

- Reduces/removes console ouputs.

## Original SPC Citation

Blatt, M., Wiseman, S., & Domany, E. (1996). Superparamagnetic clustering of data. Physical review letters, 76(18), 3251.

```bibtex
@article{spc,
  title={Superparamagnetic clustering of data},
  author={Blatt, Marcelo and Wiseman, Shai and Domany, Eytan},
  journal={Physical review letters},
  volume={76},
  number={18},
  pages={3251},
  year={1996},
  publisher={APS}
}
```