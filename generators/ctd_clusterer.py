import numpy as np

import torch
from torch.distributions import Normal

from sklearn.ensemble import IsolationForest
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.compose import ColumnTransformer

from joblib import Parallel, delayed


class ctdCluster:
    """A typical cluster for ctdGAN."""
    def __init__(self, label=None, center=None, scaler=None, embedding_dim=32,
                 continuous_columns=(), discrete_columns=(), clip=False, random_state=0):
        """
        ctdCluster initializer. A typical cluster for ctdGAN.

        Args:
            label: The cluster's label
            center (tuple): The cluster's centroid
            scaler (string): A descriptor that defines a transformation on the cluster's data. Values:

              * '`None`'  : No transformation takes place; the data is considered immutable
              * '`stds`'  : Standard scaler
              * '`mms01`' : Min-Max scaler in the range (0,1)
              * '`mms11`' : Min-Max scaler in the range (-1,1) - so that data is suitable for tanh activations
            embedding_dim (int): The dimensionality of the latent space (for the probability distribution)
            continuous_columns (tuple): The continuous columns in the input data
            discrete_columns (tuple): The columns in the input data that contain categorical variables
            clip (bool): If 'True' the reconstructed data will be clipped to their original minimum and maximum values.
            random_state (int): Seed the random number generators. Use the same value for reproducible results.
        """
        self._label = label
        self._embedding_dim = embedding_dim
        self._continuous_columns = continuous_columns
        self._discrete_columns = discrete_columns
        self._clip = clip
        self._random_state = random_state

        self._num_samples = 0
        self._data_dimensions = 0

        self.center_ = center
        self._min = None
        self._max = None

        self.class_distribution_ = None
        self.probability_distribution_ = None

        # Define the Data Transformation Model for the continuous columns
        if len(continuous_columns) > 0:
            if scaler == 'stds':
                self._scaler = StandardScaler()
                # self._scaler = ColumnTransformer(
                #    remainder='passthrough',
                #    transformers=[(scaler, StandardScaler(), continuous_columns)])

            elif scaler == 'mms01':
                self._scaler = MinMaxScaler(feature_range=(0, 1))
                # self._scaler = ColumnTransformer(
                #    remainder='passthrough',
                #    transformers=[(scaler, MinMaxScaler(feature_range=(0, 1)), continuous_columns)])

            elif scaler == 'mms11':
                self._scaler = MinMaxScaler(feature_range=(-1, 1))
                # self._scaler = ColumnTransformer(
                #   remainder='passthrough', force_int_remainder_cols=True,
                #    transformers=[(scaler, MinMaxScaler(feature_range=(-1, 1)), continuous_columns)])

            else:
                self._scaler = None
        else:
            self._scaler = None

    def fit(self, x, y=None, num_classes=0):
        """
            Compute cluster statistics and fit the selected Data Transformer.

        Args:
            x (NumPy array): The data to be transformed
            y (NumPy array): If the data has classes, pass them here. The ctdGAN will be trained for resampling.
            num_classes: The distinct number of classes in `y`.
        """
        self._num_samples = x.shape[0]
        self._data_dimensions = x.shape[1]

        mean = torch.zeros(self._embedding_dim)
        std = torch.ones(self._embedding_dim)

        # The probability distribution that models the cluster samples
        self.probability_distribution_ = Normal(loc=mean, scale=std)

        # Min/Max column values are used for clipping.
        self._min = [np.min(x[:, i]) for i in self._continuous_columns]
        self._max = [np.max(x[:, i]) for i in self._continuous_columns]

        self.class_distribution_ = np.zeros(num_classes)
        if y is not None:
            unique_classes = np.unique(y, return_counts=True)
            n = 0
            for uv in unique_classes[0]:
                self.class_distribution_[uv] = unique_classes[1][n]
                n += 1

        if self._scaler is not None:
            x_cont = x[:, self._continuous_columns]
            self._scaler.fit(x_cont)

    def transform(self, x):
        if self._scaler is not None:
            # print("Before Transformation:\n", x)
            x_cont = x[:, self._continuous_columns]
            transformed = self._scaler.transform(x_cont)
            # print("After Transformation of Continuous Cols:\n", transformed)

            for d_col in self._discrete_columns:
                transformed = np.insert(transformed, d_col, x[:, d_col], axis=1)
            # print("After Transformation & concat with Discrete Cols:\n", transformed)
            return transformed
        else:
            return x

    def fit_transform(self, x):
        """Transform the sample vectors by applying the transformation function of `self._scaler`. In fact, this
        is a simple wrapper for the `fit_transform` function of `self._scaler`.

        `self._scaler` may implement a `Pipeline`.

        Returns:
            The transformed data.
        """
        self.fit(x)
        return self._scaler.transform(x)

    def inverse_transform(self, x):
        """
        Inverse the transformation that has been applied by `self.fit_transform()`. In fact, this is a wrapper for the
        `inverse_transform` function of `self._scaler`, followed by a filter that clips the returned values.

        Args:
            x: The input data to be reconstructed (NumPy array).
            x: The input data to be reconstructed (NumPy array).

        Returns:
            The reconstructed data.
        """

        if self._scaler is not None:
            x_cont = x[:, self._continuous_columns]
            reconstructed_data = self._scaler.inverse_transform(x_cont)

            if self._clip:
                np.clip(reconstructed_data, self._min, self._max, out=reconstructed_data)

            for d_col in self._discrete_columns:
                reconstructed_data = np.insert(reconstructed_data, d_col, x[:, d_col], axis=1)
        else:
            reconstructed_data = x.copy()

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
    """  ctdGAN data preprocessor.

    It partitions the real space into clusters; then, it transforms the data of each cluster.
    """
    def __init__(self, max_clusters=10, scaler=None, samples_per_class=(), embedding_dim=32,
                 continuous_columns=(), discrete_columns=(), random_state=0):
        """
        Initializer

        Args:
            max_clusters (int): The maximum number of clusters to create
            scaler (string): A descriptor that defines a transformation on the cluster's data. Values:

              * '`None`'  : No transformation takes place; the data is considered immutable
              * '`stds`'  : Standard scaler
              * '`mms01`' : Min-Max scaler in the range (0,1)
              * '`mms11`' : Min-Max scaler in the range (-1,1) - so that data is suitable for tanh activations
            embedding_dim (int): The dimensionality of the latent space (for the probability distribution)
            continuous_columns (tuple): The continuous columns in the input data
            discrete_columns (tuple): The columns in the input data that contain categorical variables
            samples_per_class (List or tuple of integers): Contains the number of samples per class
            random_state: Seed the random number generators. Use the same value for reproducible results.
        """
        self._max_clusters = max_clusters
        self._random_state = random_state
        self._embedding_dim = embedding_dim
        self._continuous_columns = continuous_columns
        self._discrete_columns = discrete_columns

        self._scaler = scaler
        self.num_clusters_ = 0
        self.clusters_ = []
        self.cluster_labels_ = None
        self.probability_matrix_ = None
        self._samples_per_class = samples_per_class

    def perform_clustering(self, x_train, y_train, num_classes):
        """

        Args:
            x_train: Training data
            y_train: The classes of the training samples
            num_classes: The number of distinct classes of the training data

        Returns:
            Transformed data
        """
        # Find the optimal number of clusters (best_k) for k-Means algorithm. Perform multiple executions and pick
        # the one that produces the minimum scaled inertia.
        mms = MinMaxScaler()
        x_scaled = mms.fit_transform(x_train)
        k_range = range(2, self._max_clusters)

        # Perform multiple k-Means executions in parallel; store the scaled inertia of each clustering in the ans array.
        scaled_inertia = Parallel(n_jobs=1)(delayed(self._run_test_kmeans)(x_scaled, k) for k in k_range)
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

            cluster = ctdCluster(label=u, center=cluster_method.cluster_centers_[u], scaler=self._scaler,
                                 clip=True, embedding_dim=self._embedding_dim,
                                 continuous_columns=self._continuous_columns, discrete_columns=self._discrete_columns,
                                 random_state=self._random_state)
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
                # class_samples = self._samples_per_class[c]
                # class_probability = class_samples / y_train.shape[0]
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
