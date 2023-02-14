import numpy as np
from ctypes import CDLL, POINTER, cast, c_uint, c_double, c_int, c_char_p
import pathlib
import matplotlib.pyplot as plt
spclib = CDLL(str(pathlib.Path(__file__).parent.resolve()/"spclib.so"))
doublePtr = POINTER(c_double)
doublePtrPtr = POINTER(doublePtr)
uintPtr = POINTER(c_uint)
spclib.spc.argtypes = [doublePtrPtr,c_char_p, uintPtr, uintPtr]
spclib.spc.restype = c_int



class SPC():
    def __init__( self,
        mintemp=0, maxtemp=0.251,  tempstep=0.01,
        swcycles=100, nearest_neighbours=11,
        mstree=True, directedgrowth=True,
        ncl_reported=12,
        randomseed=0,
    ):
        assert mintemp<=maxtemp, "error: mintemp > maxtemp" 
        self.mintemp = mintemp
        self.maxtemp = maxtemp
        self.tempstep = tempstep
        self.swcycles = swcycles
        self.nearest_neighbours = nearest_neighbours
        self.ncl_reported = ncl_reported
        self.mstree_char = _bool2char(mstree)
        self.directedgrowth_char = _bool2char(directedgrowth)
        self.randomseed = randomseed

        self.temp_vector = np.arange(self.mintemp,self.maxtemp,self.tempstep)

        #all additional ouputs disabled
        self.output_name = 'spc_data'
        self.writelabels_char = _bool2char(False)
        self.writecorfile_char = _bool2char(False)
        self.savesuscept_char = _bool2char(False)
        self.writefpsum_char = _bool2char(False)
        self.writesizes_char = _bool2char(False)
        self.writeedges_char = _bool2char(False)

    @property
    def temp_vectr(self):
        return self.temp_vector
    
    def config_additional_outputs(self, output_name, writelabels=None,  writecorfile=None, 
        savesuscept=None, writefpsum=None, writesizes=None,writeedges=None):
        self.output_name = output_name

        if writelabels is not None: self.writelabels_char = _bool2char(writelabels)
        if writecorfile is not None: self.writecorfile_char = _bool2char(writecorfile)
        if savesuscept is not None: self.savesuscept_char = _bool2char(savesuscept)
        if writefpsum is not None: self.writefpsum_char = _bool2char(writefpsum)
        if writesizes is not None: self.writesizes_char = _bool2char(writesizes)
        if writeedges is not None: self.writeedges_char = _bool2char(writeedges)

    def run(self, data, return_sizes=False):
        dim = data.shape[1]
        data_points = data.shape[0]
        
        ntemp = self.temp_vector.shape[0]

        classes = np.ascontiguousarray(np.zeros((ntemp,data_points),dtype=c_uint, order="C"))

        new_param = (f"NumberOfPoints: {data_points}\n"
                f"OutFile: {self.output_name}\n"
                f"Dimensions: {dim}\n"
                f'MinTemp: {self.mintemp}\n'
                f'MaxTemp: {self.maxtemp}\n'
                f'TempStep: {self.tempstep}\n'
                f'SWCycles: {self.swcycles}\n'
                f'KNearestNeighbours: {self.nearest_neighbours}\n'
                f'MSTree{self.mstree_char}\n'
                f'DirectedGrowth{self.directedgrowth_char}\n'
                f'WriteLabels{self.writelabels_char}\n'
                f"ClustersReported: {self.ncl_reported}\n"
                f'WriteCorFile{self.writecorfile_char}\n'
                f"SaveSuscept{self.savesuscept_char}\n"
                f"WriteFPSum{self.writefpsum_char}\n"
                f"WriteSizes{self.writesizes_char}\n"
                f"WriteEdges{self.writeedges_char}\n"
                f"ForceRandomSeed: {self.randomseed}\n"
                "Timing~\n" 
                ) 
        
        ct_arr = np.ctypeslib.as_ctypes(data.astype(c_double, order='C',copy=False)) 
        
        doublePtrArr = doublePtr * data_points
        input_data = cast(doublePtrArr(*(cast(row, doublePtr) for row in ct_arr)), doublePtrPtr)


        if return_sizes:
            sizes = np.ascontiguousarray(np.zeros((ntemp,self.ncl_reported),dtype=c_uint, order="C"))

        res = spclib.spc(input_data, new_param.encode(),
            cast(np.ctypeslib.as_ctypes(classes),uintPtr ),
            cast(np.ctypeslib.as_ctypes(sizes), uintPtr) if return_sizes else cast(None, uintPtr))
        if res != 0:
            raise(f'spc c code returned: {res}')
        
        if return_sizes:
            return classes, sizes
        return classes

    def fit_WC1(self, data, min_clus=60, return_metadata=False):  
        classes, sizes= self.run(data, return_sizes=True)
        num_temp = sizes.shape[0]     #total number of temperatures 
        diffs =  np.diff(sizes.astype(int),axis=0)
        temp = 0  # Initial value

        for ti in range(num_temp-1):
            #Looks for changes in the cluster size of any cluster larger than min_clus.
            if any(diffs[ti,:4]>min_clus):
                temp = ti+1         

        # In case the second cluster is too small, then raise the temperature a little bit 
        if temp == 0 and sizes[temp,1] < min_clus:
            temp = 1
        nclus = sum(sizes[temp,:] > min_clus)
        c = 1 #initial class
        labels = np.zeros(classes.shape[1],dtype=int)
        for i in range(nclus):
            labels[classes[temp,:]== i] = c
            c += 1

        metadata = {'method': 'WC1'}
        if return_metadata:
            metadata['clusters_info'] = {ci+1:{'index':ci, 'itemp':temp} for ci in range(nclus)}
            metadata['sizes'] = sizes
            metadata['temperatures'] = self.temp_vector.copy()
            return labels, metadata
        return labels




    def fit_WC3(self, data, min_clus=20, elbow_min=0.4, c_ov=0.7, return_metadata=False):

        classes, sizes= self.run(data, return_sizes=True)

        maxdiff = np.max(np.diff(sizes[:,1:].astype(int),axis=0),1)
        maxdiff[maxdiff<0]=0

        main_cluster = sizes[:,0]

        prop = (main_cluster[1:]+maxdiff)/main_cluster[0:-1]
        aux = next((i for i in range(len(prop)) if prop[i]<elbow_min),np.NaN)+1 #percentaje of the rest

        # The next lines if removes the particular case where just a class is found at the 
        # lowest temperature and with just a small change the rest appears
        # all together at the next temperature
        if (not np.isnan(aux)) and self.mintemp==0 and aux==2:
            aux = next((i for i in range(len(prop)-1) if prop[i+1]<elbow_min),np.NaN)+2 #percentaje of the rest
        tree = sizes[0:-1,:]
        clus = np.zeros_like(tree).astype(bool)

        clus[tree >= min_clus]=1; #only check the ones that cross the thr
        diffs =  np.diff(tree.astype(int),axis=0)
        clus = clus  * np.vstack([np.ones_like(clus[1,:]), diffs>min_clus])

        for ii in range(clus.shape[0]):
            detect = np.nonzero(clus[ii,:])[0]
            if len(detect>0):
                clus[ii,:detect[-1]]=1

        elbow = tree.shape[0]
        if not np.isnan(aux):
            clus[aux:,:] = 0
            elbow = aux


        if return_metadata:
            metadata = {'method': 'WC3'}
            allpeaks = np.where(clus)
            metadata['method_info'] = {'elbow':elbow, 'peaks_temp':allpeaks[0], 'peaks_cl':allpeaks[1]}
        
        for ti in reversed(range(elbow)):
            detect = np.where(clus[ti,:])[0]
            for dci in detect:
                cl = classes[ti,:] == dci
                for tj in np.arange(ti-1,-1,-1):
                    toremove = np.where(clus[tj,:])[0]
                    for rj in toremove:
                        totest = classes[tj,:] == rj
                        if sum(cl * totest)/min(sum(totest),sum(cl)) >= c_ov:
                            clus[tj,rj]=0


        temp, clust_num = np.where(clus)
        c = 1 #initial class
        labels = np.zeros(classes.shape[1], dtype=int)
        for tx,cx in zip(temp, clust_num):
            labels[classes[tx,:]== cx] = c
            c += 1
    
        if return_metadata:
            metadata['clusters_info']={i+1: {'index':clust_num[i], 'itemp':ti} for i,ti in enumerate(temp)}
            metadata['sizes'] = sizes
            metadata['temperatures'] = self.temp_vector.copy()
            return labels, metadata
        return labels

    def fit(self,*args, **kwargs):
        return self.fit_WC3(*args, **kwargs)

def plot_temperature_plot(metadata,ax=None):
    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(111)
    ax.set_yscale('log')
    ax.plot(metadata['temperatures'],metadata['sizes'])
    ax.set_prop_cycle(None)
    if metadata['method'] == 'WC3':
        ax.axvline(metadata['temperatures'][metadata['method_info']['elbow']],color='k',linestyle='--')
        ax.plot(metadata['temperatures'][metadata['method_info']['peaks_temp']],
                metadata['sizes'][metadata['method_info']['peaks_temp'],metadata['method_info']['peaks_cl']],color='k',alpha=0.5,marker='x',linestyle='',markersize=8)

    for c,info in metadata['clusters_info'].items():
        ax.plot(metadata['temperatures'][info['itemp']],
        metadata['sizes'][info['itemp'],info['index']],'.',markersize=15)
    ax.set_ylabel('Cluster Sizes')
    ax.set_xlabel('Temperatures')

def _bool2char(x):
    return '|' if x else '~'