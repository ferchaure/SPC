# %%
from spclustering import SPC, plot_temperature_plot
import numpy as np
import matplotlib.pyplot as plt


def test_SPC():
    # %%
    rng = np.random.RandomState(0)
    center1 = np.array([10, 10])
    center2 = np.array([-10, -10])
    nc1 = 1500
    nc2 = 1000

    cl1 = rng.multivariate_normal(center1, [[2.5, 0.3], [0.3, 2.5]], size=nc1)
    cl2 = rng.multivariate_normal(center2, [[3, 0.9], [0.9, 4]], size=nc2)
    data = np.concatenate([cl1, cl2])

    clustering = SPC(mintemp=0.0, maxtemp=0.03)
    results, sizes = clustering.run(data, return_sizes=True)
    assert sum(results[0, :] == 0) == sizes[0, 0]
    assert sum(results[1, :] == 1) == sizes[1, 1]

    err = 0.1
    gt = np.ones(nc1+nc2)
    gt[:nc1] = 0
    assert (1-sum(results[1, :] == gt)/len(gt)) < err


def test_fit_WC1():
    rng = np.random.RandomState(5)

    n = 1000
    ncls = 5
    data_list = []
    gt_list = []
    for i in range(ncls):
        nclass = int(n/(i+1))
        center = np.array([10+7*i, 10+3*(i % 3)])
        data_list.append(rng.multivariate_normal(
            center, [[2.2, 0.4], [0.4, 2]], size=nclass))
        gt_list.append(np.ones(nclass)*i)

    data = np.concatenate(data_list)
    gt = np.concatenate(gt_list)
    clustering = SPC(mintemp=0, maxtemp=0.251,  tempstep=0.013)
    results, metadata = clustering.fit_WC1(
        data, min_clus=60, return_metadata=True)

    assert get_tprate(
        results, gt) > 0.8, 'Low performance of WC1 in toy example'

    f, ax = plt.subplots()
    plot_temperature_plot(metadata, ax=ax)
    plt.close(f)


def test_fit_WC3():
    rng = np.random.RandomState(17)
    n = 800
    ncls = 12
    data_list = []
    gt_list = []
    for i in range(ncls):
        center = rng.uniform(low=0, high=25., size=2)
        data_list.append(rng.multivariate_normal(
            center, np.array([[4.2, 0.4], [0.4, 4.5]])/(i % 3*7+1), size=n))
        gt_list.append(np.ones(n)*i)

    data = np.concatenate(data_list)
    gt = np.concatenate(gt_list)
    clustering = SPC(mintemp=0, maxtemp=0.251,  tempstep=0.013)
    # plt.scatter(data[:,0],data[:,1],c=gt,marker='.')
    results, metadata = clustering.fit_WC3(
        data, min_clus=50, return_metadata=True)

    assert get_tprate(
        results, gt) > 0.65, 'Low performance of WC3 in complex example'
    # plt.scatter(data[:,0],data[:,1],c=results,marker='.')
    f, ax = plt.subplots()
    plot_temperature_plot(metadata, ax=ax)
    plt.close(f)


def get_tprate(results, gt):
    # Calculate confusion matrix

    gt_classes = np.unique(gt)
    classes = np.unique(results)
    classes = classes[classes > 0]

    num_classes = len(classes)
    num_gt_classes = len(gt_classes)
    confusion_matrix = np.zeros((num_gt_classes, num_classes))
    for i, ci in enumerate(gt_classes):
        for j, cj in enumerate(classes):
            confusion_matrix[i, j] = np.sum((gt == ci) & (results == cj))
    tp = 0
    # assigning clusters to real classes in a greedy way
    for j, cj in enumerate(classes):
        row_ind, col_ind = np.unravel_index(
            np.argmax(confusion_matrix), confusion_matrix.shape)
        tp += confusion_matrix[row_ind, col_ind]
        confusion_matrix[row_ind, :] = -1
        confusion_matrix[:, col_ind] = -1
    return tp/len(gt)


# %%
if __name__ == '__main__':
    test_SPC()
    test_fit_WC1()
    test_fit_WC3()
