import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
import torch.nn as nn

from joblib import Parallel, delayed

from torch.distributions import Normal
from torch.utils.data import DataLoader

from sklearn.cluster import KMeans

from sklearn.preprocessing import StandardScaler, MinMaxScaler
# from sklearn.ensemble import IsolationForest

from .gan_discriminators import PackedDiscriminator
from .gan_generators import ctGenerator
from .BaseGenerators import BaseGAN
from .DataTransformers import DataTransformer

import DeepCoreML.paths as paths


class Cluster:
    """A cluster component.

    """
    def __init__(self, label, x, y, center, random_state, cap=False):
        """
        Cluster initializer.

        Args:
            label:
            x:  (NumPy array)
            y:  (NumPy array)
            center:
            random_state:
            cap:
        """
        self._label = label
        self._x = x
        self._y = y
        self._center = center
        self._cap = cap

        self._num_samples = self._y.shape[0]
        self._data_dimensions = self._x.shape[1]
        self._random_state = random_state

        # Each cluster has its own data transformer; here, the transformer is a StandardScaler with mean=0, std=1.
        self._transformer = StandardScaler(with_mean=True, with_std=True)
        self._transformer.fit(self._x)

        mean = torch.zeros(self._data_dimensions)
        std = torch.ones(self._data_dimensions)

        # mean = self._transformer.mean_
        # var = self._transformer.var_  <-- Compute sigma from var

        self._pd = Normal(loc=mean, scale=std)
        self._min = [np.min(self._x[:, i]) for i in range(self._data_dimensions)]
        self._max = [np.max(self._x[:, i]) for i in range(self._data_dimensions)]

        # print("Cluster Data:\n", self._x)
        # print("Minimum values per column:", self._min)
        # print("Maximum values per column:", self._max, "\n\n")

    def display(self):
        print("\t--- Cluster ", self._label, "-------------------------------")
        print("\t\t* Center: ", self._center)
        print("\t\t* Num Samples: ", self._num_samples)
        print("\t\t* Class Distribution: ", np.unique(self._y, return_counts=True)[1])
        print("\t-----------------------------------------------\n")

    def fit_transform(self):
        """Apply the transformation function of `self._transformer`. In fact, this is a simple wrapper for the
        `transform` function of `self._transformer`.

        `self._transformer` may implement a `Pipeline`.

        Returns:
            The transformed data.
        """
        self._transformer.fit(self._x)
        return self._transformer.transform(self._x)

    def inverse_transform(self, x):
        """
        Inverse the transformation that has been applied by `self._transformer`. In fact, this is a wrapper for the
        `inverse_transform` function of `self._transformer`, followed by a filter that places a cap for the minimum
        and maximum returned values.

        Args:
            x: The input data to be reconstructed (NumPy array).

        Returns:
            The reconstructed data.
        """
        reconstructed_data = self._transformer.inverse_transform(x)

        # if self._cap:

        return reconstructed_data

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


class ctdGAN(BaseGAN):
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
        self._transformer = None

        # Class-Cluster probability vector: Given a class c, what is the likelihood that c exists in one of the clusters
        self._probability_matrix = None   # <-- (num_classes, num_clusters)

    def _apply_activate(self, data):
        """Apply proper activation function to the output of the generator."""
        data_t = []
        st = 0
        for column_info in self._transformer.output_info_list:
            for span_info in column_info:
                if span_info.activation_fn == 'tanh':
                    ed = st + span_info.dim
                    data_t.append(torch.tanh(data[:, st:ed]))
                    st = ed
                elif span_info.activation_fn == 'softmax':
                    ed = st + span_info.dim
                    transformed = torch.softmax(data[:, st:ed], dim=1)
                    data_t.append(transformed)
                    st = ed
                else:
                    raise ValueError(f'Unexpected activation function {span_info.activation_fn}.')

        return torch.cat(data_t, dim=1)

    def k_means_res(self, scaled_data, k, alpha_k=0.02):
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

    def create_latent_spaces(self, x_train, y_train, discrete_columns):
        """
        Refine the training set with sample filtering. It invokes `prepare` to return the preprocessed data.

        Args:
            x_train: The training data instances.
            y_train: The classes of the training data instances.
            discrete_columns: The discrete columns in the dataset.

        Returns:
            A tensor with the preprocessed data.
        """
        self._n_classes = len(set(y_train))
        self._input_dim = x_train.shape[1]

        # ====== 1. Identify and remove the majority class outliers from the dataset
        # ====== Return a cleaned dataset (x_clean, y_clean)
        self._gen_samples_ratio = np.unique(y_train, return_counts=True)[1]

        '''
        num_samples = x_train.shape[1]
        majority_class = np.argmax(self._gen_samples_ratio)
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
        '''

        # Find the optimal number of clusters (best_k) for k-Means algorithm. Perform multiple executions and pick
        # the one that produces the minimum scaled inertia.
        mms = MinMaxScaler()
        x_scaled = mms.fit_transform(x_train)
        k_range = range(2, self._max_clusters)

        # Perform multiple k-Means executions in parallel; store the scaled inertia of each clustering in the ans array.
        ans = Parallel(n_jobs=-1)(delayed(self.k_means_res)(x_scaled, k) for k in k_range)
        best_k = 2 + np.argmin(ans)

        # After the optimal number of clusters best_k has been determined, execute one last k-Means with best_k clusters
        self._num_clusters = best_k
        cluster_method = KMeans(n_clusters=best_k, random_state=self._random_state, n_init='auto')
        cluster_labels = cluster_method.fit_predict(x_train)

        # Partition the dataset and create the appropriate Cluster objects.
        for comp in range(self._num_clusters):
            x_comp = x_train[cluster_labels == comp, :]
            y_comp = y_train[cluster_labels == comp]

            cluster = Cluster(comp, x_comp, y_comp, cluster_method.cluster_centers_[comp], self._random_state)
            # cluster.display()

            self._clusters.append(cluster)

        # Forge the probability matrix; Each element (i,j) stores the joint probability
        # P(class==i, cluster==j) = P(class==i) * P(cluster==j).
        self._probability_matrix = np.zeros((self._n_classes, self._num_clusters))
        for c in range(self._n_classes):
            # class_probability = self._gen_samples_ratio[c] / num_samples
            # print("class ", c, "samples:", self._gen_samples_ratio[c], "proba", class_probability)
            for comp in range(self._num_clusters):
                cluster = self._clusters[comp]
                # cluster_probability = cluster.get_num_samples() / num_samples
                cluster_probability = cluster.get_num_samples(c) / self._gen_samples_ratio[c]
                # print("\t Cluster:", comp, " - total samples", cluster.get_num_samples(),
                #       "(from class ", c, "samples:", cluster.get_num_samples(c), ", proba=", cluster_probability, ")")
                # self._probability_matrix[c][comp] = class_probability * cluster_probability
                self._probability_matrix[c][comp] = cluster_probability

        # ====== 3. Create the training set
        # Normalize the feature vectors - Each cluster has its own normalizer/scaler.
        x_trans = self._clusters[0].fit_transform()
        for comp in range(1, self._num_clusters):
            x_trans = np.concatenate((x_trans, self._clusters[comp].fit_transform()))

        train_data = np.concatenate((x_trans, cluster_labels.reshape(-1, 1), y_train.reshape(-1, 1)), axis=1)
        discrete_columns.append(self._input_dim)
        discrete_columns.append(self._input_dim + 1)

        self._transformer = DataTransformer(cont_normalizer='none')
        self._transformer.fit(train_data, discrete_columns)

        ret_data = self._transformer.transform(train_data)

        return ret_data

    def sample_latent_space(self, num_samples):
        """Latent space sampler.

        Samples the latent space and returns the latent feature vectors z and the one-hot-encoded cluster and class
        representations.

        Args:
            num_samples: The number of latent data instances.

        Returns:
             `latent_vectors:` The feature vectors of the latent data.
             `latent_clusters_ohe:` One-hot-encoded latent clusters.
             `latent_classes_ohe:` One-hot-encoded latent classes.
        """
        # Select num_samples random classes and clusters.
        latent_classes = torch.from_numpy(np.random.randint(0, self._n_classes, num_samples)).to(torch.int64)
        latent_clusters = torch.from_numpy(np.random.randint(0, self._num_clusters, num_samples)).to(torch.int64)
        latent_vectors = []
        for s in range(num_samples):
            latent_cluster_object = self._clusters[latent_clusters[s]]
            z = latent_cluster_object.sample()
            latent_vectors.append(z)

        # Convert the matrix with the latent vectors to a pytorch tensor:
        latent_vectors = torch.from_numpy(np.array(latent_vectors, dtype="float32"))

        # One-hot encoded clusters and classes:
        latent_classes_ohe = nn.functional.one_hot(latent_classes, num_classes=self._n_classes)
        latent_clusters_ohe = nn.functional.one_hot(latent_clusters, num_classes=self._num_clusters)

        return latent_vectors, latent_clusters_ohe, latent_classes_ohe

    @staticmethod
    def plot_losses(losses, store_losses):
        """
        Plot the Discriminator loss and the Generator loss values vs. the training epoch.
        Args:
            losses: A list of tuples (iteration, epoch, Discriminator loss, Generator loss) recorded during training.
            store_losses: The file path to store the plot.
        """
        columns = ["Iteration", "Epoch", "Discriminator Loss", "Generator Loss"]
        df = pd.DataFrame(losses, columns=columns)
        df.to_csv(store_losses + "ctdGAN_losses.csv", sep=";", decimal='.', index=False)

        # plot = df.plot(x="Iteration", y=["Discriminator Loss", "Generator Loss"], ylim=(0, 1))
        plot = df.plot(x="Iteration", y=["Discriminator Loss", "Generator Loss"])
        fig = plot.get_figure()
        fig.savefig(store_losses + "GAN_losses.png")

        plt.show()
    def generator_loss(self, predicted_labels, real_labels, real_data, latent_data, generated_data):
        """Custom Generator loss.

        The loss function of the Generator is a linear combination of the Discriminator Loss + Gaussian negative
        log likelihood between the generated samples and the reference Gaussian distribution of the corresponding
        component.

        Args:
            predicted_labels: The output of the Discriminator.
            real_labels: The real labels of the samples.
            real_data:
            latent_data:
            generated_data:
        """
        # Real data
        real_vectors = real_data[:, 0: self._input_dim]
        real_clusters = real_data[:, self._input_dim: (self._input_dim + self._num_clusters)]

        # Latent data
        # latent_vectors = latent_data[:, 0: self._input_dim]
        latent_clusters = latent_data[:, self._input_dim: (self._input_dim + self._num_clusters)]

        # Generated data
        generated_vectors = generated_data[:, 0:self._input_dim]
        generated_clusters = generated_data[:, self._input_dim:(self._input_dim + self._num_clusters)]

        # The Discriminator loss: Binary Cross Entropy between predicted and real labels
        disc_loss = nn.BCELoss(reduction='mean')(predicted_labels, real_labels)

        # The reconstruction error - How effective is the Generator in creating high-fidelity data
        # reconstruction_error = nn.CrossEntropyLoss(reduction='mean')(real_vectors, generated_vectors)
        reconstruction_error = nn.L1Loss(reduction='mean')(real_vectors, generated_vectors)

        # clustering loss
        clustering_loss = nn.CrossEntropyLoss(reduction='mean')(real_clusters, latent_clusters)

        # print("Real clusters:", real_clusters)
        # print("Generated Clusters:", generated_clusters)

        # Convert one-hot-encoded cluster labels back to their integer values, so that we can retrieve the parameters
        # of the corresponding probability distribution
        # cluster_labels = torch.argmax(fake_y[:, 0:self._num_clusters], dim=1)
        # distinct_cluster_labels = []
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
        # _drc_loss
        # gen_loss = disc_loss + reconstruction_error + clustering_loss

        # _dr_loss
        # gen_loss = disc_loss + reconstruction_error

        # _d_loss
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

        real_data = real_data.to(torch.float).to(self._device)

        # Sample the latent feature vectors
        # For a random class, pick a cluster from the corresponding subspace. Then sample the distribution that governs
        # this cluster to create latent vectors.
        # print("Data shape:", str(real_data.shape), "- Clusters:", str(self._num_clusters),
        #       "- Classes:", str(self._n_classes))
        latent_vectors, latent_clusters, latent_classes = self.sample_latent_space(packed_samples)
        latent_data = torch.cat((latent_vectors, latent_clusters, latent_classes), dim=1).to(self._device)

        # The Generator produces fake samples (their labels are 0)
        generated_data = self._apply_activate(self.G_(latent_data))
        generated_labels = torch.zeros((packed_samples, 1)).to(self._device)

        # The real samples (coming from the dataset) with their one-hot-encoded classes are assigned labels eq. to 1.
        real_labels = torch.ones((packed_samples, 1)).to(self._device)

        # Mix (concatenate) the fake samples (from Generator) with the real ones (from the dataset).
        all_labels = torch.cat((real_labels, generated_labels))
        all_data = torch.cat((real_data, generated_data), dim=1)

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
        latent_vectors, latent_clusters, latent_classes = self.sample_latent_space(packed_samples)
        latent_data = torch.cat((latent_vectors, latent_clusters, latent_classes), dim=1).to(self._device)

        # Pass the latent data through the Generator to synthesize samples
        generated_data = self._apply_activate(self.G_(latent_data))

        # Reshape the data to feed it to Discriminator ( (num_samples, dimensionality) -> ( -1, pac * dimensionality )
        generated_data = generated_data.reshape((-1,
                                                 self.pac_ * (self._input_dim + self._n_classes + self._num_clusters)))

        # Compute and back propagate the Generator loss
        d_predictions = self.D_(generated_data)
        gen_loss = self.generator_loss(d_predictions, real_labels, real_data, latent_data, generated_data)
        gen_loss.backward()
        self.G_optimizer_.step()

        return disc_loss, gen_loss

    def train(self, x_train, y_train, store_losses=None):
        """
        Conventional training process of a Cluster GAN. The Generator and the Discriminator are trained
        simultaneously in the traditional adversarial fashion by optimizing `loss_function`.

        Args:
            x_train: The training data instances (NumPy array).
            y_train: The classes of the training data instances (NumPy array).
            store_losses: The file path where the values of the Discriminator and Generator loss functions are stored.
        """
        # Modify the size of the batch to align with self.pac_
        factor = self._batch_size // self.pac_
        batch_size = factor * self.pac_

        # Prepare the data for training (Clustering, Computation of Probability Distributions, Transformations, etc.)
        training_data = self.create_latent_spaces(x_train, y_train, discrete_columns=[])

        # ############################################
        # LIMITATION: The latent space must be of equal dimensionality as the data space
        self.embedding_dim_ = self._input_dim
        # ############################################

        train_dataloader = DataLoader(training_data, batch_size=batch_size, shuffle=True)

        real_space_dimensions = self._input_dim + self._num_clusters + self._n_classes
        latent_space_dimensions = self.embedding_dim_ + self._num_clusters + self._n_classes

        self.D_ = PackedDiscriminator(self.D_Arch_, input_dim=real_space_dimensions, pac=self.pac_).to(self._device)

        self.G_ = ctGenerator(embedding_dim=latent_space_dimensions, architecture=self.G_Arch_,
                              data_dim=real_space_dimensions).to(self._device)

        self.D_optimizer_ = torch.optim.Adam(self.D_.parameters(),
                                             lr=self._lr, weight_decay=self._decay, betas=(0.5, 0.9))
        self.G_optimizer_ = torch.optim.Adam(self.G_.parameters(),
                                             lr=self._lr, weight_decay=self._decay, betas=(0.5, 0.9))

        losses = []
        it = 0
        for epoch in range(self._epochs):
            for real_data in train_dataloader:
                if real_data.shape[0] > 1:
                    disc_loss, gen_loss = self.train_batch(real_data)

                    if store_losses is not None:
                        it += 1
                        losses.append((it, epoch + 1, disc_loss.item(), gen_loss.item()))

        if store_losses is not None:
            self.plot_losses(losses, store_losses)

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

        # Generate data from the model's Generator - The activation function depends on the variable type:
        # - Hyperbolic tangent for continuous variables
        # - Softmax for discrete variables
        generated_data = self._apply_activate(self.G_(latent_data))

        # Throw away the generated cluster and generated class.
        generated_samples = generated_data[:, 0:self.embedding_dim_].cpu().detach().numpy()

        # Inverse the transformation of the generated samples.
        reconstructed_samples = np.zeros((num_samples, self.embedding_dim_))
        for s in range(num_samples):
            z = generated_samples[s].reshape(1, -1)
            reconstructed_samples[s] = latent_clusters_objs[s].inverse_transform(z)

        # print("Reconstructed samples\n", reconstructed_samples)
        return reconstructed_samples

    def fit_resample(self, x_train, y_train):
        """`fit_resample` alleviates the problem of class imbalance in imbalanced datasets. The function renders sbGAN
        compatible with the `imblearn`'s interface, allowing its usage in over-sampling/under-sampling pipelines.

        In the `fit` part, the input dataset is used to train sbGAN. In the `resample` part, sbGAN is employed to
        generate synthetic data according to the value of `self._sampling_strategy`:

        - 'auto': the model balances the dataset by oversampling the minority classes.
        - dict: a dictionary that indicates the number of samples to be generated from each class

        Args:
            x_train: The training data instances.
            y_train: The classes of the training data instances.

        Returns:
            x_resampled: The training data instances + the generated data instances.
            y_resampled: The classes of the training data instances + the classes of the generated data instances.
        """

        # Train the GAN with the input data
        self.train(x_train, y_train, paths.output_path_loss)

        x_resampled = np.copy(x_train)
        y_resampled = np.copy(y_train)

        # auto mode: Use sbGAN to equalize the number of samples per class. This is achieved by generating samples
        # of the minority classes (i.e. we perform oversampling).
        if self._sampling_strategy == 'auto':
            majority_class = np.array(self._gen_samples_ratio).argmax()
            num_majority_samples = np.max(np.array(self._gen_samples_ratio))

            # Perform oversampling
            for cls in range(self._n_classes):
                if cls != majority_class:
                    samples_to_generate = num_majority_samples - self._gen_samples_ratio[cls]

                    if samples_to_generate > 1:
                        # Generate the appropriate number of samples to equalize cls with the majority class.
                        # print("\tSampling Class y:", cls, " Gen Samples ratio:", gen_samples_ratio[cls])
                        generated_samples = self.sample(samples_to_generate, cls)
                        generated_classes = np.full(samples_to_generate, cls)

                        x_resampled = np.vstack((x_resampled, generated_samples))
                        y_resampled = np.hstack((y_resampled, generated_classes))

        # dictionary mode: the keys correspond to the targeted classes. The values correspond to the desired number of
        # samples for each targeted class.
        elif isinstance(self._sampling_strategy, dict):
            for cls in self._sampling_strategy:
                samples_to_generate = self._sampling_strategy[cls]

                # Generate the appropriate number of samples to equalize cls with the majority class.
                # print("\tSampling Class y:", cls, " Gen Samples ratio:", gen_samples_ratio[cls])
                generated_samples = self.sample(samples_to_generate, cls)
                generated_classes = np.full(samples_to_generate, cls)

                x_resampled = np.vstack((x_resampled, generated_samples))
                y_resampled = np.hstack((y_resampled, generated_classes))

        return x_resampled, y_resampled
