# Conditional GAN Implementation
import numpy as np

import torch
import torch.nn as nn

from torch.utils.data import DataLoader

from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.ensemble import IsolationForest

from .gan_discriminators import PackedDiscriminator
from .gan_generators import Generator
from .BaseGenerators import BaseGAN


class GAANv1(BaseGAN):
    """
    Safe-Borderline GAN

    Conditional GANs (cGANs) conditionally generate data from a specific class. They are trained
    by providing both the Generator and the Discriminator the input feature vectors concatenated
    with their respective one-hot-encoded class labels.

    A Packed Conditional GAN (Pac cGAN) is a cGAN that accepts input samples in packs. Pac cGAN
    uses a Packed Discriminator to prevent the model from mode collapsing.
    """

    def __init__(self, embedding_dim=128, discriminator=(128, 128), generator=(256, 256), pac=10, g_activation='tanh',
                 adaptive=False, epochs=300, batch_size=32, lr=2e-4, decay=1e-6, sampling_strategy='auto',
                 max_clusters=20, cond_vector='uniform', projector=None, random_state=0):
        """
        Initializes a Safe-Borderline Conditional GAN.

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
            cond_vector: How to draw cluster labels for the Generator:

             - `uniform`: Random sampling of class and cluster labels from a uniform distribution.
             - `prob`: Random integer sampling with probability.
             - `log-prob`: Random integer sampling with log probability.
            random_state: An integer for seeding the involved random number generators.
        """
        super().__init__(embedding_dim, discriminator, generator, pac, g_activation, adaptive, epochs, batch_size,
                         lr, decay, sampling_strategy, random_state)

        self._real_samples_ratio = None
        self._projector = projector

        self._max_clusters = max_clusters
        self._n_clusters = 0
        self._cond_vector = cond_vector

        # Sample Probability Vectors (classes, clusters): During GAN sampling, when requesting samples of class y to be
        # generated, what is the most suitable cluster to draw samples from.
        self._spv = None

        # Sample Probability Vectors (classes, clusters): During GAN training given a random latent sample with class y,
        # what is the most suitable cluster
        self._tpv = None

    def cluster_prepare(self, x_train, y_train):
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
        num_samples = x_train.shape[0]

        # ====== 1. Identify and remove the majority class outliers from the dataset
        # ====== Return a cleaned dataset (x_clean, y_clean)
        self._gen_samples_ratio = np.unique(y_train, return_counts=True)[1]
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

        self._real_samples_ratio = np.unique(y_clean, return_counts=True)[1]

        # (x_clean, y_clean) is the new dataset without the outliers
        print("Clean Dataset Shape:", x_clean.shape)

        # ====== 2. Clustering step
        # ====== Use a Gaussian Mixture Model (GMM) to cluster the cleaned samples
        n_components = range(1, self._max_clusters + 1)
        covariance_type = ['spherical', 'tied', 'diag', 'full']
        score = []
        for cov in covariance_type:
            for n_comp in n_components:
                gmm = GaussianMixture(n_components=n_comp, covariance_type=cov, random_state=self._random_state)
                gmm.fit(x_clean)
                score.append((cov, n_comp, gmm.bic(x_clean)))

        # Best covariance_type and best number of clusters
        best = min(score, key=lambda tup: tup[2])
        self._n_clusters = best[1]  # <-- Best number of clusters
        best_cov = best[0]          # <-- Best covariance type of GMM
        print(best)

        gmm = GaussianMixture(n_components=self._n_clusters, covariance_type=best_cov, random_state=self._random_state)
        gmm.fit(x_clean)
        cluster_labels = gmm.predict(x_clean)

        # One-hot-encode the Cluster labels for the conditional vector of clusterGAN
        cluster_encoder = OneHotEncoder()
        encoded_cluster_labels = cluster_encoder.fit_transform(cluster_labels.reshape(-1, 1)).toarray()
        # print(encoded_cluster_labels)

        # ====== 3. Feature Transformation Step
        # ====== First standardize, then project the cleaned samples to a latent space of max variance by using PCA.
        # ====== What is the connection between the best covariance type of GMM and the kernel function of PCA??
        if self._projector is None:
            self._transformer = Pipeline([('scaler', StandardScaler(with_mean=True, with_std=True))])
        elif self._projector == 'pca' or self._projector == 'PCA':
            self._transformer = Pipeline([
                    ('scaler', StandardScaler(with_mean=True, with_std=True)),
                    ('pca', PCA(n_components=self._input_dim, random_state=self._random_state))
                ])

        self._transformer.fit(x_clean)
        x_clean_projected = self._transformer.transform(x_clean)

        # ====== 4. Cluster structure analysis step
        # ====== Get class distributions per cluster: e.g., for a cluster with 10 samples from class 0 and 20 samples
        # ====== from class 1, output lcd=[10, 20]. self._cluster_class_distribution stores one lcd list per cluster.
        self._tpv = np.zeros((self._n_classes, self._n_clusters))
        self._spv = np.zeros((self._n_clusters, self._n_classes))

        for cluster in range(self._n_clusters):
            y_cluster = y_clean[cluster_labels == cluster]
            cluster_samples = y_cluster.shape[0]
            for c in range(self._n_classes):
                ccd = len(y_cluster[y_cluster == c])
                self._spv[cluster][c] = ccd / cluster_samples
                self._tpv[c][cluster] = ccd / self._real_samples_ratio[c]
        # print("SPV", self._spv)
        # print("TPV:", self._tpv)

        # ====== 5. Finalize the training set
        class_encoder = OneHotEncoder()
        encoded_class_labels = class_encoder.fit_transform(y_clean.reshape(-1, 1)).toarray()

        # Concatenate everything (cleaned/scaled/projected feature vectors, ohe_clusters, ohe_classes)
        train_data = np.concatenate((x_clean_projected, encoded_cluster_labels, encoded_class_labels), axis=1)
        training_data = torch.from_numpy(train_data).to(torch.float32)

        # Class specific training data
        self._samples_per_class = []
        for y in range(self._n_classes):
            x_class_data = np.array([x_clean_projected[r, :]
                                     for r in range(encoded_class_labels.shape[0]) if encoded_class_labels[r, y] == 1])
            x_class_data = torch.from_numpy(x_class_data).to(torch.float32).to(self._device)

            self._samples_per_class.append(x_class_data)

        # print("Training Data:\n", training_data)
        # print("Shapes:\n\tClusters:", encoded_cluster_labels.shape, "\n\tClasses:", encoded_class_labels.shape,
        #      "\n\tFeatures:", x_clean_projected.shape, "\n\tTraining set:", training_data.shape)

        return training_data

    def train_batch(self, real_data):
        """
        Given a batch of input data, `train_batch` updates the Discriminator and Generator weights using the respective
        optimizers and back propagation.

        Args:
            real_data: data for cGAN training: a batch of concatenated sample vectors + one-hot-encoded class vectors.
        """

        # The loss function for GAN training - applied to both the Discriminator and Generator.
        loss_function = nn.BCELoss()

        # If the size of the batch does not allow an organization of the input vectors in packs of size self.pac_, then
        # abort silently and return without updating the model parameters.
        num_samples = real_data.shape[0]
        if num_samples % self.pac_ != 0:
            return 0, 0

        packed_samples = num_samples // self.pac_

        # DISCRIMINATOR TRAINING
        # Create fake samples from Generator
        self.D_optimizer_.zero_grad()

        # 1. Sample the feature vectors (latext_x) from a normal distribution.
        # 2. Assign one-hot-encoded random clusters - Find the respective class probability vectors
        # 3. Pass the fake data (samples + classes) to the Generator
        if self._cond_vector == 'uniform':
            latent_classes = torch.from_numpy(np.random.randint(0, self._n_classes, num_samples)).to(torch.int64)
            latent_clusters = torch.from_numpy(np.random.randint(0, self._n_clusters, num_samples)).to(torch.int64)
        else:
            latent_clusters = torch.from_numpy(np.random.randint(0, self._n_clusters, num_samples)).to(torch.int64)
            latent_classes = np.zeros(num_samples)
            for s in range(num_samples):
                latent_classes[s] = np.random.choice(a=np.arange(self._n_classes, dtype=int), size=1,
                                                     replace=True, p=self._spv[latent_clusters[s]])
                # print("\tSample:", s, ", Class:", latent_classes[s], ", Cluster", latent_clusters[s])
            latent_classes = torch.from_numpy(latent_classes).to(torch.int64)

        latent_clusters_ohe = nn.functional.one_hot(latent_clusters, num_classes=self._n_clusters)
        latent_classes_ohe = nn.functional.one_hot(latent_classes, num_classes=self._n_classes)

        latent_x = torch.randn((num_samples, self.embedding_dim_))
        latent_y = torch.cat((latent_clusters_ohe, latent_classes_ohe), dim=1)
        latent_data = torch.cat((latent_x, latent_y), dim=1)

        # 4. The Generator produces fake samples (their labels are 0)
        fake_x = self.G_(latent_data.to(self._device))
        fake_labels = torch.zeros((packed_samples, 1))

        # 5. The real samples (coming from the dataset) with their one-hot-encoded classes are assigned labels eq. to 1.
        real_x = real_data[:, 0:self._input_dim]
        real_y = real_data[:, self._input_dim:(self._input_dim + self._n_clusters + self._n_classes)]
        real_labels = torch.ones((packed_samples, 1))
        # print(real_x.shape, real_y.shape)

        # 6. Mix (concatenate) the fake samples (from Generator) with the real ones (from the dataset).
        all_x = torch.cat((real_x.to(self._device), fake_x))
        all_y = torch.cat((real_y, latent_y)).to(self._device)
        all_labels = torch.cat((real_labels, fake_labels)).to(self._device)
        all_data = torch.cat((all_x, all_y), dim=1)

        # 7. Reshape the data to feed it to Discriminator (num_samples, dimensionality) -> (-1, pac * dimensionality)
        # The samples are packed according to self.pac parameter.
        all_data = all_data.reshape((-1, self.pac_ * (self._input_dim + self._n_clusters + self._n_classes)))

        # 8. Pass the mixed data to the Discriminator and train the Discriminator (update its weights with backprop).
        # The loss function quantifies the Discriminator's ability to classify a real/fake sample as real/fake.
        d_predictions = self.D_(all_data)
        disc_loss = loss_function(d_predictions, all_labels)
        disc_loss.backward()
        self.D_optimizer_.step()

        # GENERATOR TRAINING
        self.G_optimizer_.zero_grad()

        if self._cond_vector == 'uniform':
            latent_classes = torch.from_numpy(np.random.randint(0, self._n_classes, num_samples)).to(torch.int64)
            latent_clusters = torch.from_numpy(np.random.randint(0, self._n_clusters, num_samples)).to(torch.int64)
        else:
            latent_clusters = torch.from_numpy(np.random.randint(0, self._n_clusters, num_samples)).to(torch.int64)
            latent_classes = np.zeros(num_samples)
            for s in range(num_samples):
                latent_classes[s] = np.random.choice(a=np.arange(self._n_classes, dtype=int), size=1,
                                                     replace=True, p=self._spv[latent_clusters[s]])
                # print("\tSample:", s, ", Class:", latent_classes[s], ", Cluster", latent_clusters[s])
            latent_classes = torch.from_numpy(latent_classes).to(torch.int64)

        latent_clusters_ohe = nn.functional.one_hot(latent_clusters, num_classes=self._n_clusters)
        latent_classes_ohe = nn.functional.one_hot(latent_classes, num_classes=self._n_classes)

        latent_x = torch.randn((num_samples, self.embedding_dim_))
        latent_y = torch.cat((latent_clusters_ohe, latent_classes_ohe), dim=1)
        latent_data = torch.cat((latent_x, latent_y), dim=1)

        fake_x = self.G_(latent_data.to(self._device))

        all_data = torch.cat((fake_x, latent_y.to(self._device)), dim=1)

        # Reshape the data to feed it to Discriminator ( (num_samples, dimensionality) -> ( -1, pac * dimensionality )
        all_data = all_data.reshape((-1, self.pac_ * (self._input_dim + self._n_classes + self._n_clusters)))

        d_predictions = self.D_(all_data)

        gen_loss = loss_function(d_predictions, real_labels.to(self._device))
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

        # select_prepare: implemented in BaseGenerators.py
        training_data = self.cluster_prepare(x_train, y_train)

        train_dataloader = DataLoader(training_data, batch_size=batch_size, shuffle=True)

        self.D_ = PackedDiscriminator(self.D_Arch_, input_dim=self._input_dim + self._n_classes + self._n_clusters,
                                      pac=self.pac_).to(self._device)
        self.G_ = (Generator(self.G_Arch_, input_dim=self.embedding_dim_ + self._n_classes + self._n_clusters,
                             output_dim=self._input_dim, activation=self.gen_activation_, normalize=self.batch_norm_).
                   to(self._device))

        self.D_optimizer_ = torch.optim.Adam(self.D_.parameters(),
                                             lr=self._lr, weight_decay=self._decay, betas=(0.5, 0.9))
        self.G_optimizer_ = torch.optim.Adam(self.G_.parameters(),
                                             lr=self._lr, weight_decay=self._decay, betas=(0.5, 0.9))

        disc_loss, gen_loss = 0, 0
        for epoch in range(self._epochs):
            for n, real_data in enumerate(train_dataloader):
                if real_data.shape[0] > 1:
                    disc_loss, gen_loss = self.train_batch(real_data)

                # if epoch % 10 == 0 and n >= x_train.shape[0] // batch_size:
                #    print(f"Epoch: {epoch} Loss D.: {disc_loss} Loss G.: {gen_loss}")

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

        # Completely random selection of cluster and latent sample vectors
        if self._cond_vector == 'uniform':
            latent_clusters = torch.from_numpy(np.random.randint(0, self._n_clusters, num_samples)).to(torch.int64)
            latent_x = torch.randn((num_samples, self.embedding_dim_))

        # Select the cluster with probability self._cc_prob_vector
        else:
            latent_clusters = np.zeros(num_samples)

            # For each sample with a specific class, pick a cluster with probability self._cc_prob_vector[s_class].
            # Then build the latent sample vector by sampling the (Multivariate) Gaussian that describes this cluster.
            for s in range(num_samples):
                s_class = latent_classes[s]
                s_cluster = np.random.choice(a=np.arange(self._n_clusters, dtype=int), size=1, replace=True,
                                             p=self._tpv[s_class])[0]
                latent_clusters[s] = s_cluster

            latent_clusters = torch.from_numpy(latent_clusters).to(torch.int64)
            latent_x = torch.randn((num_samples, self.embedding_dim_))

        latent_clusters_ohe = nn.functional.one_hot(latent_clusters, num_classes=self._n_clusters)
        latent_y = torch.cat((latent_clusters_ohe, latent_classes_ohe), dim=1)

        # Concatenate, copy to device, and pass to generator
        latent_data = torch.cat((latent_x, latent_y), dim=1).to(self._device)

        # Generate data from the model's Generator - The feature values of the generated samples fall into the range:
        # [-1,1]: if the activation function of the output layer of the Generator is nn.Tanh().
        # [0,1]: if the activation function of the output layer of the Generator is nn.Sigmoid().
        generated_samples = self.G_(latent_data).cpu().detach().numpy()
        # print("Generated Samples:\n", generated_samples)

        reconstructed_samples = self._transformer.inverse_transform(generated_samples)

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
