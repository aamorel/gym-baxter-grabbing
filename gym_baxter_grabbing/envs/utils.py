import numpy as np
from sklearn.cluster import KMeans
from scipy.spatial import cKDTree as KDTree
from scipy.spatial import distance

color_list = ["#000000", "#FFFF00", "#1CE6FF", "#FF34FF", "#FF4A46", "#008941", "#006FA6", "#A30059",
              "#FFDBE5", "#7A4900", "#0000A6", "#63FFAC", "#B79762", "#004D43", "#8FB0FF", "#997D87",
              "#5A0007", "#809693", "#FEFFE6", "#1B4400", "#4FC601", "#3B5DFF", "#4A3B53", "#FF2F80",
              "#61615A", "#BA0900", "#6B7900", "#00C2A0", "#FFAA92", "#FF90C9", "#B903AA", "#D16100",
              "#DDEFFF", "#000035", "#7B4F4B", "#A1C299", "#300018", "#0AA6D8", "#013349", "#00846F",
              "#372101", "#FFB500", "#C2FFED", "#A079BF", "#CC0744", "#C0B9B2", "#C2FF99", "#001E09",
              "#00489C", "#6F0062", "#0CBD66", "#EEC3FF", "#456D75", "#B77B68", "#7A87A1", "#788D66",
              "#885578", "#FAD09F", "#FF8A9A", "#D157A0", "#BEC459", "#456648", "#0086ED", "#886F4C",
              "#34362D", "#B4A8BD", "#00A6AA", "#452C2C", "#636375", "#A3C8C9", "#FF913F", "#938A81",
              "#575329", "#00FECF", "#B05B6F", "#8CD0FF", "#3B9700", "#04F757", "#C8A1A1", "#1E6E00",
              "#7900D7", "#A77500", "#6367A9", "#A05837", "#6B002C", "#772600", "#D790FF", "#9B9700",
              "#549E79", "#FFF69F", "#201625", "#72418F", "#BC23FF", "#99ADC0", "#3A2465", "#922329",
              "#5B4534", "#FDE8DC", "#404E55", "#0089A3", "#CB7E98", "#A4E804", "#324E72", "#6A3A4C",
              "#83AB58", "#001C1E", "#D1F7CE", "#004B28", "#C8D0F6", "#A3A489", "#806C66", "#222800",
              "#BF5650", "#E83000", "#66796D", "#DA007C", "#FF1A59", "#8ADBB4", "#1E0200", "#5B4E51",
              "#C895C5", "#320033", "#FF6832", "#66E1D3", "#CFCDAC", "#D0AC94", "#7ED379", "#012C58"]


class CVT():
    def __init__(self, num_centroids=7, bounds=[[-1, 1]], num_samples=100000,
                 num_replicates=1, max_iterations=20, tolerance=0.001):
        
        self.num_centroids = num_centroids
        self.bounds = bounds
        self.num_samples = num_samples
        self.num_replicates = num_replicates
        self.max_iterations = max_iterations
        self.tolerance = tolerance

        X = []
        for bound in bounds:
            X.append(np.random.uniform(low=bound[0], high=bound[1], size=self.num_samples))
        X = np.array(X)
        X = np.transpose(X)

        kmeans = KMeans(init='k-means++',
                        n_clusters=num_centroids,
                        n_init=num_replicates,
                        max_iter=max_iterations,
                        tol=tolerance,
                        verbose=0)
        
        kmeans.fit(X)

        self.centroids = kmeans.cluster_centers_
        self.k_tree = KDTree(self.centroids)

    def get_grid_index(self, sample):
        grid_index = self.k_tree.query(sample, k=1)[1]
        return grid_index


def bound(behavior, bound_behavior):
    for i in range(len(behavior)):
        if behavior[i] < bound_behavior[i][0]:
            behavior[i] = bound_behavior[i][0]
        if behavior[i] > bound_behavior[i][1]:
            behavior[i] = bound_behavior[i][1]


def normalize(behavior, bound_behavior):
    for i in range(len(behavior)):
        range_of_interval = bound_behavior[i][1] - bound_behavior[i][0]
        mean_of_interval = (bound_behavior[i][0] + bound_behavior[i][1]) / 2
        behavior[i] = (behavior[i] - mean_of_interval) / (range_of_interval / 2)


def list_l2_norm(list1, list2):
    if len(list1) != len(list2):
        raise NameError('The two lists have different length')
    else:
        dist = 0
        for i in range(len(list1)):
            dist += (list1[i] - list2[i]) ** 2
        dist = dist ** (1 / 2)
        return dist


def compute_uniformity(grid):
    P = grid[np.nonzero(grid)]
    P = P / np.sum(P)
    Q = np.ones(len(P)) / len(P)
    uniformity = 1 - distance.jensenshannon(P, Q)
    return uniformity
