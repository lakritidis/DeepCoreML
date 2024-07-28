import numpy as np

import torch
from torch.distributions import Normal

from sklearn.ensemble import IsolationForest
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler, MinMaxScaler

from joblib import Parallel, delayed


class ctdCluster:
    """A cluster component.

    """
    def __init__(self, label=None, center=None, data_transformer=None, clip=False, random_state=0):
        """
        Cluster initializer.

        Args:
            label: The cluster's label
            center: The cluster's centroid
            data_transformer (string): A descriptor that defines a transformation on the cluster's data. Values:

              * '`None`' : No transformation takes place; the data is considered immutable
              * '`stds`' : Standard scaler
              * '`mms`'  : Min-Max scaler
            clip (bool): If 'True' the reconstructed data will be clipped to their original minimum and maximum values.
            random_state (int): Seed the random number generators. Use the same value for reproducible results.
        """
        self._label = label
        self._clip = clip
        self._random_state = random_state

        self._num_samples = 0
        self._data_dimensions = 0

        self.center_ = center
        self._min = None
        self._max = None

        self.class_distribution_ = None
        self.probability_distribution_ = None

        # Define the Data Transformation Model
        if data_transformer == 'stds':
            self._transformer = StandardScaler()
        elif data_transformer == 'mms':
            self._transformer = MinMaxScaler()
        else:
            self._transformer = None

    def fit(self, x, y=None, num_classes=0):
        """
            Compute cluster statistics and fit the selected Data Transformer.

        Args:
            `x` (NumPy array): The data to be transformed
            `y` (NumPy array): If the data has classes, pass them here. The ctdGAN will be trained for resampling.
            `num_classes`: The distinct number of classes in `y`.

        """
        self._num_samples = x.shape[0]
        self._data_dimensions = x.shape[1]

        mean = torch.zeros(self._data_dimensions)
        std = torch.ones(self._data_dimensions)

        # The probability distribution that models the cluster samples.
        self.probability_distribution_ = Normal(loc=mean, scale=std)

        # Min/Max column values are used for clipping.
        self._min = [np.min(x[:, i]) for i in range(self._data_dimensions)]
        self._max = [np.max(x[:, i]) for i in range(self._data_dimensions)]

        self.class_distribution_ = np.zeros(num_classes)
        if y is not None:
            unique_values = np.unique(y, return_counts=True)
            n = 0
            for uv in unique_values[0]:
                self.class_distribution_[uv] = unique_values[1][n]
                n += 1

        if self._transformer is not None:
            self._transformer.fit(x)

    def transform(self, x):
        if self._transformer is not None:
            return self._transformer.transform(x)
        else:
            return x

    def fit_transform(self, x):
        """Transform the sample vectors by applying the transformation function of `self._transformer`. In fact, this
        is a simple wrapper for the `fit_transform` function of `self._transformer`.

        `self._transformer` may implement a `Pipeline`.

        Returns:
            The transformed data.
        """
        self.fit(x)
        return self._transformer.transform(x)

    def inverse_transform(self, x):
        """
        Inverse the transformation that has been applied by `self.fit_transform()`. In fact, this is a wrapper for the
        `inverse_transform` function of `self._transformer`, followed by a filter that clips the returned values.

        Args:
            x: The input data to be reconstructed (NumPy array).

        Returns:
            The reconstructed data.
        """
        if self._transformer is not None:
            reconstructed_data = self._transformer.inverse_transform(x)
        else:
            reconstructed_data = x.copy()

        if self._clip:
            np.clip(reconstructed_data, self._min, self._max, out=reconstructed_data)

        return reconstructed_data

    def display(self):
        """
        Display useful cluster properties.
        """
        print("\t--- Cluster ", self._label, "-----------------------------------------")
        print("\t\t* Num Samples: ", self._num_samples)
        print("\t\t* Class Distribution: ", self.class_distribution_)
        print("\t\t* Center: ", self.center_)
        print("\t\t* Min values per column:", self._min)
        print("\t\t* Max values per column:", self._max)
        print("\t---------------------------------------------------------\n")

    def get_label(self):
        return self._label

    def get_center(self):
        return self.center_

    def sample(self):
        return self.probability_distribution_.rsample()

    def get_num_samples(self, c=None):
        if c is None:
            return self._num_samples
        else:
            if len(self.class_distribution_) > 0:
                return self.class_distribution_[c]
            else:
                return -1

    def set_label(self, v):
        self._label = v


class ctdClusterer:
    def __init__(self, max_clusters=10, data_transformer='None', samples_per_class=(), random_state=0):
        self._max_clusters = max_clusters
        self._random_state = random_state

        self._transformer = data_transformer
        self.num_clusters_ = 0
        self.clusters_ = []
        self.cluster_labels_ = None
        self.probability_matrix_ = None
        self._samples_per_class = samples_per_class

    def perform_clustering(self, x_train, y_train, num_classes):
        # Find the optimal number of clusters (best_k) for k-Means algorithm. Perform multiple executions and pick
        # the one that produces the minimum scaled inertia.
        mms = MinMaxScaler()
        x_scaled = mms.fit_transform(x_train)
        k_range = range(2, self._max_clusters)

        # Perform multiple k-Means executions in parallel; store the scaled inertia of each clustering in the ans array.
        scaled_inertia = Parallel(n_jobs=-1)(delayed(self._run_test_kmeans)(x_scaled, k) for k in k_range)
        best_k = 2 + np.argmin(scaled_inertia)

        # After the optimal number of clusters best_k has been determined, execute one last k-Means with best_k clusters
        self.num_clusters_ = best_k
        cluster_method = KMeans(n_clusters=self.num_clusters_, random_state=self._random_state, n_init='auto')
        self.cluster_labels_ = cluster_method.fit_predict(x_train)

        # Partition the dataset and create the appropriate Cluster objects.
        transformed_data = None
        for u in range(self.num_clusters_):
            x_u = x_train[self.cluster_labels_ == u, :]
            y_u = y_train[self.cluster_labels_ == u]

            cluster = ctdCluster(label=u, center=cluster_method.cluster_centers_[u], data_transformer=self._transformer,
                                 clip=True, random_state=self._random_state)
            cluster.fit(x_u, y_u, len(self._samples_per_class))

            x_transformed = cluster.transform(x_u)
            cluster_labels = (u * np.ones(y_u.shape[0])).reshape(-1, 1)
            class_labels = np.array(y_u).reshape(-1, 1)

            if u == 0:
                transformed_data = np.concatenate((x_transformed, cluster_labels, class_labels), axis=1)
            else:
                concat = np.concatenate((x_transformed, cluster_labels, class_labels), axis=1)
                transformed_data = np.concatenate((transformed_data, concat))

            self.clusters_.append(cluster)

        # Forge the probability matrix; Each element (i,j) stores the joint probability
        # P(class==i, cluster==j) = P(class==i) * P(cluster==j).
        if num_classes > 1:
            self.probability_matrix_ = np.zeros((num_classes, self.num_clusters_))
            for c in range(num_classes):
                class_samples = self._samples_per_class[c]
                class_probability = class_samples / y_train.shape[0]
                # print("\nClass:", c, "- Samples:", class_samples, ", Class probability:", class_probability)

                for u in range(self.num_clusters_):
                    cluster = self.clusters_[u]
                    # print("\t Cluster:", comp, "- Samples:", cluster.get_num_samples(),
                    #      "(", cluster.get_num_samples(c), "from class", c, ")")

                    # cluster_probability = cluster.get_num_samples() / num_samples
                    cluster_probability = cluster.get_num_samples(c) / self._samples_per_class[c]

                    # self._probability_matrix[c][comp] = class_probability * cluster_probability
                    self.probability_matrix_[c][u] = cluster_probability

        np.random.shuffle(transformed_data)

        return transformed_data

    def remove_majority_outliers(self, x_train, y_train):
        num_samples = x_train.shape[1]
        majority_class = np.argmax(self._samples_per_class)
        maj_samples = np.array([x_train[s, :] for s in range(num_samples) if y_train[s] == majority_class])

        # Use an Isolation Forest to detect the outliers. The predictions array marks them with -1
        outlier_detector = IsolationForest(random_state=self._random_state)
        outlier_detector.fit(maj_samples)
        predictions = outlier_detector.predict(maj_samples)

        # Copy all the minority samples to the cleaned dataset
        x_clean = np.array([x_train[s, :] for s in range(num_samples) if y_train[s] != majority_class])
        y_clean = np.array([y_train[s] for s in range(num_samples) if y_train[s] != majority_class])

        # Copy the majority samples that are not outliers to the cleaned dataset
        for s in range(maj_samples.shape[0]):
            if predictions[s] == 1:
                x_clean = np.append(x_clean, [maj_samples[s, :]], axis=0)
                y_clean = np.append(y_clean, [majority_class], axis=0)

        # (x_clean, y_clean) is the new dataset without the outliers
        # print("Clean Dataset Shape:", x_clean.shape)

    def _run_test_kmeans(self, scaled_data, k, alpha_k=0.02):
        """
        Parameters
        ----------
        scaled_data: matrix
            scaled data. rows are samples and columns are features for clustering
        k: int
            current k for applying KMeans
        alpha_k: float
            manually tuned factor that gives penalty to the number of clusters
        Returns
        -------
        scaled_inertia: float
            scaled inertia value for current k
        """

        inertia_o = np.square((scaled_data - scaled_data.mean(axis=0))).sum()

        # Fit k-means
        kmeans = KMeans(n_clusters=k, random_state=self._random_state, n_init='auto')
        kmeans.fit(scaled_data)
        scaled_inertia = kmeans.inertia_ / inertia_o + alpha_k * k

        return scaled_inertia

    def display(self):
        print("Num Clusters: ", self.num_clusters_)
        for i in range(self.num_clusters_):
            self.clusters_[i].display()

    def get_cluster(self, i):
        return self.clusters_[i]

    def get_cluster_center(self, i):
        return self.clusters_[i].center_
