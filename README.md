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
from spclustering import SPC, plot_temperature_plot
import matplotlib.pyplot as plt
import numpy as np
randomseed = 0
rng = np.random.RandomState(randomseed)

cl1 = rng.multivariate_normal([8, 8], [[4,0],[0,3]], size=800)
cl2 = rng.multivariate_normal([0,0], [[3,0],[0,2]], size=2000)
cl3 = rng.multivariate_normal([5,5], [[0.5,0.2],[0.2,0.6]], size=300)
cl4 = rng.multivariate_normal([-3,-2.5], [[0.5,0],[0,0.6]], size=500)
gt = [cl1,cl2,cl3,cl4]

#plot clusters
plt.figure()
for cl in gt:
    plt.plot(*cl.T,marker='.',linestyle='',markersize=3)
plt.title('Ground Truth')
plt.grid()
plt.show()


#run the algorithm. The fit method applied the cluster selection described in Waveclus 3. The method fit_WC1 is the alternative using the original Waveclus 1 temperature selection.
data = np.concatenate(gt)
clustering = SPC(mintemp=0,maxtemp=0.2,randomseed=randomseed)
labels, metadata = clustering.fit(data,min_clus=150,return_metadata=True)

#It is posible to show a temperature map using the optional output metadata
plot_temperature_plot(metadata)
plt.show()

#To show the assigned labels:
plt.figure()
for c in np.unique(labels):
    if c==0:
        plt.plot(*data[labels==c,:].T,marker='.',color='k',linestyle='',markersize=3)
    else:
        plt.plot(*data[labels==c,:].T,marker='.',linestyle='',markersize=3)
plt.grid()
plt.title('Results')
plt.show()
```

## Limitations and Changes

- It runs with a data matrix [npoints, ndims] as input. Running the clustering from a distance matrix is not implemented.

- Only the main parameters and flags are implemented.

- Tmax (stop temperature) is not included in the temperature array (similar to Python ```range```). 

- Included *WriteEdges* boolean parameter (defaul False). If True, the \*.edges/\*.mst/\*.K file is created as in the original SPC. Added aswell *PrintParam* parameter in the legacy SW.c code.

- Reduces/removes console ouputs.

## Citations
### Original SPC 
Blatt, M., Wiseman, S., & Domany, E. (1996). Superparamagnetic clustering of data. Physical review letters, 76(18), 3251.


### Cluster Selection
#### Waveclus 1 (WC1)

Quian Quiroga R, Nadasdy Z, Ben-Shaul Y. Unsupervised spike detection and sorting with wavelets and superparamagnetic clustering. Neural computation 16: 1661–1687, 2004.

#### Waveclus 3 (WC3)

Chaure FJ, Rey HG, Quian Quiroga R. A novel and fully automatic spike sorting implementation with variable number of features. J Neurophysiol , 2018. doi:10.1152/jn.00339.2018.

### Bibtex
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

@article{WC1,
	title = {Unsupervised spike detection and sorting with wavelets and superparamagnetic clustering},
	volume = {16},
	number = {8},
	journal = {Neural computation},
	author = {Quian Quiroga, R and Nadasdy, Zoltan and Ben-Shaul, Yoram},
	year = {2004},
	pages = {1661--1687},
	file = {48b6995f327440100e0d7382ff2652c17c6f.pdf:I\:\\My Drive\\zotero\\storage\\GXTC9KF8\\48b6995f327440100e0d7382ff2652c17c6f.pdf:application/pdf},
}

@article{WC3,
	title = {A novel and fully automatic spike sorting implementation with variable number of features},
	issn = {1522-1598},
	doi = {10.1152/jn.00339.2018},
	journal = {Journal of Neurophysiology},
	author = {Chaure, Fernando Julian and Rey, Hernan Gonzalo and Quian Quiroga, Rodrigo},
	month = jul,
	year = {2018},
	pmid = {29995603},
	keywords = {neurophysiology, single-neuron recordings, spike sorting, tetrode}
}
```