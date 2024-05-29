import numpy as np

import torch
import torch.nn as nn

from joblib import Parallel, delayed

from torch.distributions import Normal
# from torch.distributions.multivariate_normal import MultivariateNormal

from torch.utils.data import DataLoader

from sklearn.cluster import KMeans

from sklearn.preprocessing import StandardScaler, MinMaxScaler, OneHotEncoder
# from sklearn.ensemble import IsolationForest

from .gan_discriminators import PackedDiscriminator
from .gan_generators import Generator
from .BaseGenerators import BaseGAN


class Cluster:
    """A cluster component.

    """
    def __init__(self, label, x, y, center, random_state):
        self._label = label
        self._x = x
        self._y = y
        self._center = center

        self._num_samples = self._y.shape[0]
        self._transformer = StandardScaler(with_mean=True, with_std=True)
        self._random_state = random_state

        mean = torch.from_numpy(self._center)
        std = torch.from_numpy(np.std(x, axis=0))

        mean = torch.zeros(x.shape[1])
        std = torch.ones(x.shape[1])

        self._pd = Normal(loc=mean, scale=std)

    def display(self):
        print("\t--- Cluster ", self._label, "-------------------------------")
        print("\t\t* Center: ", self._center)
        print("\t\t* Num Samples: ", self._num_samples)
        print("\t-----------------------------------------------\n")

    def fit_transform(self):
        """Apply the transformation indicated by `self._transformer`.
        `self._transformer` may implement a `Pipeline`.
        """
        self._transformer.fit(self._x)
        return self._transformer.transform(self._x)

    def inverse_transform(self, x):
        return self._transformer.inverse_transform(x)

    def get_label(self):
        return self._label

    def get_center(self):
        return self._center

    def sample(self):
        return self._pd.rsample()

    def get_num_samples(self, c=None):
        if c is None:
            return self._num_samples
        else:
            return self._y[self._y == c].shape[0]

    def set_label(self, v):
        self._label = v


class GAANv4(BaseGAN):
    """
    GMM GAN

    Conditional GANs (cGANs) conditionally generate data from a specific class. They are trained
    by providing both the Generator and the Discriminator the input feature vectors concatenated
    with their respective one-hot-encoded class labels.

    A Packed Conditional GAN (Pac cGAN) is a cGAN that accepts input samples in packs. Pac cGAN
    uses a Packed Discriminator to prevent the model from mode collapsing.
    """

    def __init__(self, embedding_dim=128, discriminator=(128, 128), generator=(256, 256), pac=10, g_activation='tanh',
                 adaptive=False, epochs=300, batch_size=32, lr=2e-4, decay=1e-6, sampling_strategy='auto',
                 max_clusters=20, projector=None, random_state=0):
        """
        Initializes a SGMM GAN.

        Args:
            embedding_dim: Size of the random sample passed to the Generator.
            discriminator: a tuple with number of neurons in each fully connected layer of the Discriminator. It
                determines the dimensionality of the output of each layer.
            generator: a tuple with number of neurons in each fully connected layer of the Generator. It
                determines the dimensionality of the output of each residual block of the Generator.
            pac: Number of samples to group together when applying the discriminator.
            adaptive: boolean value to enable/disable adaptive training.
            g_activation: The activation function of the Generator's output layer.
            epochs: Number of training epochs.
            batch_size: Number of data instances per training batch.
            lr: Learning rate parameter for the Generator/Discriminator Adam optimizers.
            decay: Weight decay parameter for the Generator/Discriminator Adam optimizers.
            max_clusters: The maximum number of clusters to create

             - `uniform`: Random sampling of class and cluster labels from a uniform distribution.
             - `prob`: Random integer sampling with probability.
             - `log-prob`: Random integer sampling with log probability.
            random_state: An integer for seeding the involved random number generators.
        """
        super().__init__(embedding_dim, discriminator, generator, pac, g_activation, adaptive, epochs, batch_size,
                         lr, decay, sampling_strategy, random_state)

        self._projector = projector

        self._max_clusters = max_clusters
        self._num_clusters = 0
        self._clusters = []

        # Class-Cluster probability vector: Given a class c, what is the likelihood that c exists in one of the clusters
        self._probability_matrix = None   # <-- (num_classes, num_clusters)

    def create_latent_spaces(self, x_train, y_train):
        """
        Refine the training set with sample filtering. It invokes `prepare` to return the preprocessed data.

        Args:
            x_train: The training data instances.
            y_train: The classes of the training data instances.

        Returns:
            A tensor with the preprocessed data.
        """
        self._n_classes = len(set(y_train))
        self._input_dim = x_train.shape[1]

        # ====== 1. Identify and remove the majority class outliers from the dataset
        # ====== Return a cleaned dataset (x_clean, y_clean)
        self._gen_samples_ratio = np.unique(y_train, return_counts=True)[1]

        '''
        majority_class = np.argmax(self._gen_samples_ratio)
        maj_samples = np.array([x_train[s, :] for s in range(num_samples) if y_train[s] == majority_class])

        # Use an Isolation Forest to detect the outliers. The predictions array marks them with -1
        clf = IsolationForest(random_state=self._random_state).fit(maj_samples)
        predictions = clf.predict(maj_samples)

        # Copy all the minority samples to the cleaned dataset
        x_clean = np.array([x_train[s, :] for s in range(num_samples) if y_train[s] != majority_class])
        y_clean = np.array([y_train[s] for s in range(num_samples) if y_train[s] != majority_class])

        # Copy the majority samples that are not outliers to the cleaned dataset
        for s in range(maj_samples.shape[0]):
            if predictions[s] == 1:
                x_clean = np.append(x_clean, [maj_samples[s, :]], axis=0)
                y_clean = np.append(y_clean, [majority_class], axis=0)
        '''

        # (x_clean, y_clean) is the new dataset without the outliers
        # print("Clean Dataset Shape:", x_clean.shape)
        mms = MinMaxScaler()
        x_scaled = mms.fit_transform(x_train)
        k_range = range(2, self._max_clusters)
        ans = Parallel(n_jobs=-1)(delayed(self.kMeansRes)(x_scaled, k) for k in k_range)
        best_k = 2 + np.argmin(ans)

        self._num_clusters = best_k
        cluster_method = KMeans(n_clusters=best_k, random_state=self._random_state, n_init='auto')
        cluster_labels = cluster_method.fit_predict(x_train)

        # Each GMM Component is a cluster; its samples adopt a Multivariate/Normal Gaussian Distribution.
        # Create the appropriate objects, distributions and probability vectors.
        self._probability_matrix = np.zeros((self._n_classes, self._num_clusters))
        for comp in range(self._num_clusters):
            x_comp = x_train[cluster_labels == comp, :]
            y_comp = y_train[cluster_labels == comp]

            cluster = Cluster(comp, x_comp, y_comp, cluster_method.cluster_centers_[comp], self._random_state)

            self._clusters.append(cluster)

        # Forge the probability matrix; Each element (i,j) stores the joint probability
        # P(class==i, cluster==j) = P(class==i) * P(cluster==j).
        for c in range(self._n_classes):
            # class_probability = self._gen_samples_ratio[c] / num_samples
            # print("class ", c, "samples:", self._gen_samples_ratio[c], "proba", class_probability)
            for comp in range(self._num_clusters):
                cluster = self._clusters[comp]
                # cluster_probability = cluster.get_num_samples() / num_samples
                cluster_probability = cluster.get_num_samples(c) / self._gen_samples_ratio[c]
                # print("\tcluster:", comp, " - total samples", cluster.get_num_samples(),
                #       "(from class ", c, "samples:", cluster.get_num_samples(c), ", proba=", cluster_probability, ")")
                # self._probability_matrix[c][comp] = class_probability * cluster_probability
                self._probability_matrix[c][comp] = cluster_probability

        # print(self._probability_matrix)

        # ====== 3. Create the training set
        cluster_encoder = OneHotEncoder()
        encoded_cluster_labels = cluster_encoder.fit_transform(cluster_labels.reshape(-1, 1)).toarray()

        class_encoder = OneHotEncoder()
        encoded_class_labels = class_encoder.fit_transform(y_train.reshape(-1, 1)).toarray()

        x_transformed = self._clusters[0].fit_transform()
        for comp in range(1, self._num_clusters):
            x_transformed = np.concatenate((x_transformed, self._clusters[comp].fit_transform()))

        # print("Class labels:", encoded_class_labels.shape, "\nCluster labels:", encoded_cluster_labels.shape,
        #       "\nData shape:", x_transformed.shape)

        # Concatenate the (transformed) feature vectors, the (one-hot-encoded) cluster labels and the (ohe) classes.
        train_data = np.concatenate((x_transformed, encoded_cluster_labels, encoded_class_labels), axis=1)

        training_data = torch.from_numpy(train_data).to(torch.float32)

        return training_data

    def sample_latent_space(self, num_samples):
        """Latent space sampler.

        Samples the latent space and returns the feature vectors and the one-hot-encoded cluster and class
        representations.

        Args:
            num_samples: The number of samples to create

        Returns:
             `latent_x:` The feature vectors of the sampled data.
             `latent_y:` Concatenated vector between the one-hot-encoded cluster and the one-hot-encoded class of
             the sampled data.
        """
        # Select num_samples random classes
        latent_classes = torch.from_numpy(np.random.randint(0, self._n_classes, num_samples)).to(torch.int64)
        latent_clusters = np.zeros(num_samples)
        latent_x = []

        for s in range(num_samples):
            lat_class = int(latent_classes[s])
            lat_cluster = np.random.choice(a=np.arange(self._num_clusters, dtype=int), size=1,
                                           replace=True, p=self._probability_matrix[lat_class])

            latent_cluster_object = self._clusters[int(lat_cluster)]
            latent_clusters[s] = latent_cluster_object.get_label()
            # print("Sample", s, "-- Class", int(latent_classes[s]), '--Cluster', int(lat_cluster), "(",
            #      latent_clusters[s], ")\n-- Prob Vec:", lat_subspace.get_cluster_probs())

            # latent_x derives by sampling the cluster's Multivariate Gaussian distribution
            z = latent_cluster_object.sample()
            latent_x.append(z)

        latent_x = torch.from_numpy(np.array(latent_x, dtype="float32"))
        latent_clusters = torch.from_numpy(latent_clusters).to(torch.int64)

        # One-hot encoded clusters and classes are concatenated together in latent_y
        latent_classes_ohe = nn.functional.one_hot(latent_classes, num_classes=self._n_classes)
        latent_clusters_ohe = nn.functional.one_hot(latent_clusters, num_classes=self._num_clusters)
        latent_y = torch.cat((latent_clusters_ohe, latent_classes_ohe), dim=1)

        return latent_x, latent_y

    def generator_loss(self, predicted_labels, real_labels, fake_x, fake_y):
        """Custom Generator loss

        The loss function of the Generator is a linear combination of the Discriminator Loss + Gaussian negative
        log likelihood between the generated samples and the reference Gaussian distribution of the corresponding
        component.

        Args:
             predicted_labels: The output of the Discriminator.
             real_labels: The real labels of the samples.
             fake_x: The output of the Generator.
             fake_y: Concatenated vector of one-hot-encoded cluster label + one-hot-encoded class label.
        """
        # The Discriminator loss: Binary Cross Entropy between predicted and real labels
        disc_loss = nn.BCELoss(reduction='mean')(predicted_labels, real_labels)
        reconstruction_error = 0

        # Convert one-hot-encoded cluster labels back to their integer values, so that we can retrieve the parameters
        # of the corresponding probability distribution
        cluster_labels = torch.argmax(fake_y[:, 0:self._num_clusters], dim=1)
        distinct_cluster_labels = []
        '''
        for cl in cluster_labels:
            clabel = int(cl)
            if clabel not in distinct_cluster_labels:
                distinct_cluster_labels.append(clabel)

                rows = [idx for idx in range(cluster_labels.shape[0]) if cluster_labels[idx] == cl]
                n_rows = len(rows)
                gen_samples = fake_x[rows, :]
                var = (gen_samples - self._clusters[clabel].get_center().to(self._device)) ** 2

                target = self.clusters[clabel].get_prob_distribution().rsample(sample_shape=[n_rows]).to(self._device)

                gaussian_loss = nn.GaussianNLLLoss(full=True, reduction='none')(gen_samples, target, var)
                for los in range(n_rows):
                    reconstruction_error += gaussian_loss[los].mean()
        '''
        gen_loss = disc_loss

        return gen_loss

    def train_batch(self, real_data):
        """
        Given a batch of input data, `train_batch` updates the Discriminator and Generator weights using the respective
        optimizers and back propagation.

        Args:
            real_data: data for cGAN training: a batch of concatenated sample vectors + one-hot-encoded class vectors.
        """

        # The loss function for GAN training - applied to both the Discriminator and Generator.
        disc_loss_function = nn.BCELoss(reduction='mean')

        # If the size of the batch does not allow an organization of the input vectors in packs of size self.pac_, then
        # abort silently and return without updating the model parameters.
        num_samples = real_data.shape[0]
        if num_samples % self.pac_ != 0:
            return 0, 0

        packed_samples = num_samples // self.pac_

        # DISCRIMINATOR TRAINING
        # Create fake samples from Generator
        self.D_optimizer_.zero_grad()

        # Sample the latent feature vectors
        # For a random class, pick a cluster from the corresponding subspace. latent_y = concat(cluster, class)
        # Then sample the MVN governing this cluster to create latent_x.
        latent_x, latent_y = self.sample_latent_space(packed_samples)
        latent_data = torch.cat((latent_x, latent_y), dim=1)

        # The Generator produces fake samples (their labels are 0)
        fake_x = self.G_(latent_data.to(self._device))
        fake_labels = torch.zeros((packed_samples, 1))

        # The real samples (coming from the dataset) with their one-hot-encoded classes are assigned labels eq. to 1.
        real_x = real_data[:, :self._input_dim]
        real_y = real_data[:, self._input_dim:(self._input_dim + self._num_clusters + self._n_classes)]
        real_labels = torch.ones((packed_samples, 1))

        # Mix (concatenate) the fake samples (from Generator) with the real ones (from the dataset).
        all_x = torch.cat((real_x.to(self._device), fake_x))
        all_y = torch.cat((real_y, latent_y)).to(self._device)
        all_labels = torch.cat((real_labels, fake_labels)).to(self._device)
        all_data = torch.cat((all_x, all_y), dim=1)

        # 7. Reshape the data to feed it to Discriminator (num_samples, dimensionality) -> (-1, pac * dimensionality)
        # The samples are packed according to self.pac parameter.
        all_data = all_data.reshape((-1, self.pac_ * (self._input_dim + self._num_clusters + self._n_classes)))

        # 8. Pass the mixed data to the Discriminator and train the Discriminator (update its weights with backprop).
        # The loss function quantifies the Discriminator's ability to classify a real/fake sample as real/fake.
        d_predictions = self.D_(all_data)
        disc_loss = disc_loss_function(d_predictions, all_labels)
        disc_loss.backward()
        self.D_optimizer_.step()

        # GENERATOR TRAINING
        self.G_optimizer_.zero_grad()

        # Sample data from the latent spaces
        latent_x, latent_y = self.sample_latent_space(num_samples)
        latent_data = torch.cat((latent_x, latent_y), dim=1)

        # Pass the latent data through the Generator to synthesize samples
        fake_x = self.G_(latent_data.to(self._device))

        fake_data = torch.cat((fake_x, latent_y.to(self._device)), dim=1)

        # Reshape the data to feed it to Discriminator ( (num_samples, dimensionality) -> ( -1, pac * dimensionality )
        fake_data = fake_data.reshape((-1, self.pac_ * (self._input_dim + self._n_classes + self._num_clusters)))

        # Compute and back propagate the Generator loss
        d_predictions = self.D_(fake_data)
        gen_loss = self.generator_loss(d_predictions, real_labels.to(self._device), fake_x, latent_y)
        gen_loss.backward()
        self.G_optimizer_.step()

        return disc_loss, gen_loss

    def train(self, x_train, y_train):
        """
        Conventional training process of a Cluster GAN. The Generator and the Discriminator are trained
        simultaneously in the traditional adversarial fashion by optimizing `loss_function`.

        Args:
            x_train: The training data instances.
            y_train: The classes of the training data instances.
        """
        # Modify the size of the batch to align with self.pac_
        factor = self._batch_size // self.pac_
        batch_size = factor * self.pac_

        # Prepare the data for training (Clustering, Computation of Probability Distributions, Transformations, etc.)
        training_data = self.create_latent_spaces(x_train, y_train)

        # ############################################
        # LIMITATION: The latent space must be of equal dimensionality as the data space
        self.embedding_dim_ = self._input_dim
        # ############################################

        train_dataloader = DataLoader(training_data, batch_size=batch_size, shuffle=True)

        self.D_ = PackedDiscriminator(self.D_Arch_, input_dim=self._input_dim + self._n_classes + self._num_clusters,
                                      pac=self.pac_).to(self._device)
        self.G_ = (Generator(self.G_Arch_, input_dim=self.embedding_dim_ + self._n_classes + self._num_clusters,
                             output_dim=self._input_dim, activation=self.gen_activation_, normalize=self.batch_norm_).
                   to(self._device))

        self.D_optimizer_ = torch.optim.Adam(self.D_.parameters(),
                                             lr=self._lr, weight_decay=self._decay, betas=(0.5, 0.9))
        self.G_optimizer_ = torch.optim.Adam(self.G_.parameters(),
                                             lr=self._lr, weight_decay=self._decay, betas=(0.5, 0.9))

        disc_loss, gen_loss = 0, 0
        for epoch in range(self._epochs):
            for real_data in train_dataloader:
                if real_data.shape[0] > 1:
                    disc_loss, gen_loss = self.train_batch(real_data)

        return disc_loss, gen_loss

    def fit(self, x_train, y_train):
        """`fit` invokes the GAN training process. `fit` renders the CGAN class compatible with `imblearn`'s interface,
        allowing its usage in over-sampling/under-sampling pipelines.

        Args:
            x_train: The training data instances.
            y_train: The classes of the training data instances.
        """
        self.train(x_train, y_train)

    def sample(self, num_samples, y=None):
        """ Create artificial samples using the GAN's Generator.

        Args:
            num_samples: The number of samples to generate.
            y: The class of the generated samples. If `None`, then samples with random classes are generated.

        Returns:
            Artificial data instances created by the Generator.
        """
        # If no specific class is required, pick classes randomly.
        if y is None:
            latent_classes = torch.from_numpy(np.random.randint(0, self._n_classes, num_samples)).to(torch.int64)
        # Otherwise, fill the classes tensor with the requested class (y) value
        else:
            latent_classes = torch.full(size=(num_samples,), fill_value=y)

        latent_classes_ohe = nn.functional.one_hot(latent_classes, num_classes=self._n_classes)

        # Select the cluster with probability self._cc_prob_vector
        latent_clusters = np.zeros(num_samples)
        latent_clusters_objs = []
        latent_x = []

        # For each sample with a specific class, pick a cluster with probability lat_subspace.get_cluster_probs().
        # Then build the latent sample vector by sampling the (Multivariate) Gaussian that describes this cluster.
        for s in range(num_samples):
            lat_class = int(latent_classes[s])
            lat_cluster = np.random.choice(a=np.arange(self._num_clusters, dtype=int), size=1,
                                           replace=True, p=self._probability_matrix[lat_class])

            latent_cluster_object = self._clusters[int(lat_cluster)]

            latent_clusters_objs.append(latent_cluster_object)
            latent_clusters[s] = latent_cluster_object.get_label()
            # print("Sample", s, "-- Class", int(latent_classes[s]), '--Cluster', int(lat_cluster), "(",
            #      latent_clusters[s], ")\n-- Prob Vec:", lat_subspace.get_cluster_probs())

            # latent_x derives by sampling the cluster's Multivariate Gaussian distribution
            z = latent_cluster_object.sample()
            latent_x.append(z)

        latent_x = torch.from_numpy(np.array(latent_x, dtype="float32"))

        latent_clusters = torch.from_numpy(latent_clusters).to(torch.int64)
        latent_clusters_ohe = nn.functional.one_hot(latent_clusters, num_classes=self._num_clusters)
        latent_y = torch.cat((latent_clusters_ohe, latent_classes_ohe), dim=1)

        # Concatenate, copy to device, and pass to generator
        latent_data = torch.cat((latent_x, latent_y), dim=1).to(self._device)

        # Generate data from the model's Generator - The feature values of the generated samples fall into the range:
        # [-1,1]: if the activation function of the output layer of the Generator is nn.Tanh().
        # [0,1]: if the activation function of the output layer of the Generator is nn.Sigmoid().
        generated_samples = self.G_(latent_data).cpu().detach().numpy()
        # print("Generated Samples:\n", generated_samples)

        # Inverse the transformation of the generated samples
        reconstructed_samples = np.zeros((num_samples, self.embedding_dim_))
        for s in range(num_samples):
            z = generated_samples[s].reshape(1, -1)
            reconstructed_samples[s] = latent_clusters_objs[s].inverse_transform(z)

        # print("Reconstructed samples\n", reconstructed_samples)
        return reconstructed_samples

    def fit_resample(self, x_train, y_train):
        """`fit_resample` alleviates the problem of class imbalance in imbalanced datasets. In particular, this
         resampling operation equalizes the number of samples from each class by oversampling the minority class.
         The function renders clusterGAN compatible with the `imblearn`'s interface, allowing its usage in
         over-sampling/under-sampling pipelines.

        Args:
            x_train: The training data instances.
            y_train: The classes of the training data instances.

        Returns:
            x_balanced: the balanced dataset samples
            y_balanced: the classes of the samples of x_balanced
        """

        # Train the GAN with the input data
        self.train(x_train, y_train)

        # auto mode: Use the GAN to equalize the number of samples per class. This is achieved by generating
        # samples of the minority classes (i.e. we perform oversampling).
        if self._sampling_strategy == 'auto':
            majority_class = np.array(self._gen_samples_ratio).argmax()
            num_majority_samples = np.max(np.array(self._gen_samples_ratio))

            x_balanced = np.copy(x_train)
            y_balanced = np.copy(y_train)

            # Perform oversampling
            for cls in range(self._n_classes):
                if cls != majority_class:
                    samples_to_generate = num_majority_samples - self._gen_samples_ratio[cls]

                    # Generate the appropriate number of samples to equalize cls with the majority class.
                    # print("\tSampling Class y:", cls, " Gen Samples ratio:", gen_samples_ratio[cls])
                    generated_samples = self.sample(samples_to_generate, cls)
                    generated_classes = np.full(samples_to_generate, cls)

                    x_balanced = np.vstack((x_balanced, generated_samples))
                    y_balanced = np.hstack((y_balanced, generated_classes))

            # Return balanced_data
            return x_balanced, y_balanced

        # synthesize_similar mode: Use the GAN to create a new dataset with identical class distribution
        # as the training set.
        elif self._sampling_strategy == 'synthesize_similar':
            i = 0
            x_synthetic = None
            y_synthetic = None

            # Perform oversampling
            for cls in range(self._n_classes):
                # print("Sampling class", cls, "Create", self._gen_samples_ratio[cls], "samples")
                samples_to_generate = self._gen_samples_ratio[cls]

                generated_samples = self.sample(samples_to_generate, cls)
                generated_classes = np.full(samples_to_generate, cls)

                if i == 0:
                    x_synthetic = generated_samples
                    y_synthetic = generated_classes
                else:
                    x_synthetic = np.vstack((x_synthetic, generated_samples))
                    y_synthetic = np.hstack((y_synthetic, generated_classes))
                i += 1

            # Return balanced_data
            return x_synthetic, y_synthetic


    def kMeansRes(self, scaled_data, k, alpha_k=0.02):
        '''
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
        '''

        inertia_o = np.square((scaled_data - scaled_data.mean(axis=0))).sum()
        # fit k-means
        kmeans = KMeans(n_clusters=k, random_state=self._random_state, n_init='auto').fit(scaled_data)
        scaled_inertia = kmeans.inertia_ / inertia_o + alpha_k * k
        return scaled_inertia

