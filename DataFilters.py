import numpy as np
from sklearn.neighbors import NearestNeighbors, KDTree


def no_filter(x, y, cls):
    return np.array([x[k] for k in range(x.shape[0]) if y[k] == cls])


def filter_nn(x, y, cls):
    all_minority_samples = np.array([x[k] for k in range(x.shape[0]) if y[k] == cls])
    num_all_minority_samples = all_minority_samples.shape[0]

    neighbors = NearestNeighbors(n_neighbors=5, algorithm='kd_tree').fit(x)
    distances, indices = neighbors.kneighbors(all_minority_samples)

    outliers = []
    core_points = []
    border_points = []

    for m in range(num_all_minority_samples):
        # min_sample = x[m]
        # print("Checking point", m, " - Neighbors:", indices.shape[1])
        points_with_same_class = 0

        for k in range(indices.shape[1]):
            nn_idx = indices[m][k]

            if y[m] == y[nn_idx]:
                points_with_same_class = points_with_same_class + 1

        # if points_with_same_class > 2:
        #    core_points.append(x[m])
        # print("point", m, " same class", points_with_same_class)
        if points_with_same_class == indices.shape[1]:
            core_points.append(x[m])
        elif points_with_same_class == 1:
            outliers.append(x[m])
        else:
            border_points.append(x[m])

    minority_samples = []
    # minority_samples.extend(core_points)
    if len(minority_samples) < 0.4 * num_all_minority_samples:
        minority_samples.extend(border_points)

    return np.array(minority_samples)


def filter_sphere(x, y, radius, cls):
    all_minority_samples = np.array([x[k] for k in range(x.shape[0]) if y[k] == cls])
    num_all_minority_samples = all_minority_samples.shape[0]

    tree = KDTree(x, leaf_size=10)
    indices = tree.query_radius(all_minority_samples, r=radius)

    isolated_points = []
    outliers = []
    core_points = []
    border_points = []

    for m in range(num_all_minority_samples):
        minority_sample = all_minority_samples[m]
        neighbors_in_radius = indices[m]
        num_neighbors = len(neighbors_in_radius)
        # print("Checking point", m, " pts in radius:", len(pts_in_radius))

        if num_neighbors == 1:
            isolated_points.append(minority_sample)
        else:
            points_with_same_class = 0
            # For each neighbor of the minority sample
            for k in range(num_neighbors):
                nn_idx = neighbors_in_radius[k]

                if y[nn_idx] == cls:
                    points_with_same_class = points_with_same_class + 1

            # print("point", m, " same class", points_with_same_class)
            if points_with_same_class == num_neighbors:
                core_points.append(minority_sample)
            elif points_with_same_class == 1:
                outliers.append(minority_sample)
            else:
                border_points.append(minority_sample)

    # print("Core points:", len(core_points), ", outliers:", len(outliers), ", border_pts:", + len(border_points))
    minority_samples = []
    minority_samples.extend(core_points)
    if len(minority_samples) < 0.40 * num_all_minority_samples:
        minority_samples.extend(border_points)
    if len(minority_samples) < 0.40 * num_all_minority_samples:
        minority_samples.extend(isolated_points)

    minority_samples_array = np.array(minority_samples)
    return minority_samples_array
