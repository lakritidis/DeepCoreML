import numpy as np

import torch
import torch.nn as nn

from torch.utils.data import DataLoader

from DeepCoreML.TabularTransformer import TabularTransformer
from .gan_discriminators import Critic
from .gan_generators import ctGenerator
from .BaseGenerators import BaseGAN
from .ctd_clusterer import ctdClusterer

import DeepCoreML.paths as paths


class ctdGAN(BaseGAN):
    """
    ctdGAN implementation

    ctdGAN conditionally generates tabular data for confronting class imbalance in machine learning tasks. The model
    is trained by embedding both cluster and class labels into the features vectors, and by penalizing incorrect
    cluster and class predictions.
    """

    def __init__(self, embedding_dim=128, discriminator=(128, 128), generator=(256, 256), epochs=300, batch_size=32,
                 pac=10, lr=2e-4, decay=1e-6, sampling_strategy='auto', max_clusters=20, random_state=0):
        """
        ctdGAN initializer

        Args:
            embedding_dim (int): Size of the random sample passed to the Generator.
            discriminator (tuple): a tuple with number of neurons for each fully connected layer of the model's Critic.
                It determines the dimensionality of the output of each layer.
            generator (tuple): a tuple with number of neurons for each fully connected layer of the model's Generator.
                It determines the dimensionality of the output of each residual block of the Generator.
            epochs (int): The number of training epochs.
            batch_size (int): The number of data instances per training batch.
            pac (int): The number of samples to group together when applying the Critic.
            lr (real): The value of the learning rate parameter for the Generator/Critic Adam optimizers.
            decay (real): The value of the weight decay parameter for the Generator/Critic Adam optimizers.
            sampling_strategy (string or dictionary): How the algorithm generates samples:

              * 'auto': the model balances the dataset by oversampling the minority classes.
              * dict: a dictionary that indicates the number of samples to be generated from each class.
            max_clusters (int): The maximum number of clusters to create.
            random_state (int): Seed the random number generators. Use the same value for reproducible results.
        """
        super().__init__(embedding_dim, discriminator, generator, pac, None, False, epochs,
                         batch_size, lr, decay, sampling_strategy, random_state)

        # clustered_transformer performs clustering and (optionally), data transformation
        self._clustered_transformer = None

        # discrete_transformer performs dataset-wise one-hot-encoding of the categorical columns
        self._discrete_transformer = None

        self._max_clusters = max_clusters
        self._n_clusters = 0

    def _apply_activate(self, data):
        """Apply proper activation function to the output of the generator."""
        data_t = []
        st = 0
        for column_info in self._discrete_transformer.output_info_list:
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

    def cluster_transform(self, x_train, y_train, discrete_columns):
        """
        Perform clustering and (optionally) transform the data in the generated clusters.

        Args:
            x_train: The training data instances.
            y_train: The classes of the training data instances.
            discrete_columns: The discrete columns in the dataset. The last two columns indicate the cluster and class.

        Returns:
            A tensor with the preprocessed data.
        """
        self._n_classes = len(set(y_train))
        self._input_dim = x_train.shape[1]

        # ====== Initialize and deploy the Clustered Transformer object that: i) partitions the real space, and
        # ====== ii) performs data transformations (scaling, PCA, outlier detection, etc.)
        self._samples_per_class = np.unique(y_train, return_counts=True)[1]

        self._clustered_transformer = ctdClusterer(max_clusters=self._max_clusters, data_transformer='stds',
                                                   samples_per_class=self._samples_per_class,
                                                   random_state=self._random_state)

        train_data = self._clustered_transformer.perform_clustering(x_train, y_train, self._n_classes)
        self._n_clusters = self._clustered_transformer.num_clusters_

        # print("Training Data:\n", train_data)
        # ====== Append the cluster and class labels to the collection of the discrete columns
        discrete_columns.append(self._input_dim)
        discrete_columns.append(self._input_dim + 1)

        # ====== Transform the discrete columns only; the continuous columns have been scaled at cluster-level.
        self._discrete_transformer = TabularTransformer(cont_normalizer='none', clip=True)
        self._discrete_transformer.fit(train_data, discrete_columns)

        ret_data = self._discrete_transformer.transform(train_data)
        # print("Returning Data:\n", ret_data)
        # inp = input('STATEMENT')

        return ret_data

    def sample_latent_space(self, num_samples):
        """Latent space sampler

        Samples the latent space and returns the latent feature vectors z, the one-hot-encoded cluster labels and
        the one-hot-encoded class labels.

        Args:
            num_samples: The number of latent data instances.

        Returns:
             * `latent_vectors`      :  The feature vectors of the latent data.
             * `latent_clusters_ohe` :  One-hot-encoded latent clusters.
             * `latent_classes_ohe:` :  One-hot-encoded latent classes.
        """
        # Select num_samples random classes and clusters.
        latent_classes = torch.from_numpy(np.random.randint(0, self._n_classes, num_samples)).to(torch.int64)
        latent_clusters = torch.from_numpy(np.random.randint(0, self._n_clusters, num_samples)).to(torch.int64)
        latent_vectors = []
        for s in range(num_samples):
            latent_cluster_object = self._clustered_transformer.get_cluster(latent_clusters[s])
            z = latent_cluster_object.sample()
            latent_vectors.append(z)

        # Convert the matrix with the latent vectors to a pytorch tensor:
        latent_vectors = torch.from_numpy(np.array(latent_vectors, dtype="float32"))

        # One-hot encoded clusters and classes:
        latent_classes_ohe = nn.functional.one_hot(latent_classes, num_classes=self._n_classes)
        latent_clusters_ohe = nn.functional.one_hot(latent_clusters, num_classes=self._n_clusters)

        return latent_vectors, latent_clusters_ohe, latent_classes_ohe

    def generator_loss(self, critic_output, real_data, latent_data, generated_data):
        """Generator loss function

        The loss function of the Generator combines the Discriminator Loss, and the mis-predictions of
        the cluster and class labels.

        Args:
            critic_output: The output of the Discriminator.
            real_data: The real data in the batch.
            latent_data: The data where the Generator sampled from.
            generated_data: The data generated by the Generator.
        """
        num_generated_samples = generated_data.size()[0]

        # cluster_st_idx = self._discrete_transformer.output_dimensions - self._n_clusters - self._n_classes
        # cluster_ed_idx = cluster_st_idx + self._n_clusters
        # print("Cluster st:", cluster_st_idx, ", Cluster ed:", cluster_ed_idx)

        # class_st_idx = self._discrete_transformer.output_dimensions - self._n_classes
        # class_ed_idx = class_st_idx + self._n_classes
        # print("Class st:", class_st_idx, ", Class ed:", class_ed_idx)

        # Loss for the discrete columns - Including the class & cluster labels.
        # print("Real data shape", real_data.shape, " - Latent data shape", latent_data.shape)
        discrete_loss = []
        st_idx = 0
        for column_metadata in self._discrete_transformer.output_info_list:
            for span_info in column_metadata:
                if len(column_metadata) != 1 or span_info.activation_fn != 'softmax':
                    # Continuous column
                    st_idx += span_info.dim
                else:
                    ed_idx = st_idx + span_info.dim

                    # print("\tStart:", st_idx, ", End: ", ed_idx)
                    # print("\t== Latent Data:\n", latent_data[:, st_idx:ed_idx])
                    # print("\t== Generated Data:\n", generated_data[:, st_idx:ed_idx])

                    tmp = nn.functional.cross_entropy(
                        generated_data[:, st_idx:ed_idx],
                        torch.argmax(latent_data[:, st_idx:ed_idx], dim=1), reduction='none')

                    discrete_loss.append(tmp)
                    st_idx = ed_idx

        cond_loss = torch.stack(discrete_loss, dim=1)
        discrete_columns_loss = cond_loss.sum() / num_generated_samples
        # print("discrete_columns_loss = ", discrete_columns_loss)

        # Real data
        real_vectors = real_data[:, 0: self._input_dim]
        real_clusters = real_data[:, self._input_dim: (self._input_dim + self._n_clusters)]

        # Latent data
        # latent_vectors = latent_data[:, 0: self._input_dim]
        latent_clusters = latent_data[:, self._input_dim: (self._input_dim + self._n_clusters)]

        # Generated data
        generated_vectors = generated_data[:, 0:self._input_dim]
        generated_clusters = generated_data[:, self._input_dim:(self._input_dim + self._n_clusters)]

        # Mis-classification error: Binary Cross Entropy between predicted (from the Discriminator) and real labels.
        # classification_error = nn.BCELoss(reduction='mean')(predicted_labels, real_labels)

        # Mis-clustering error: Difference between the generated and latent cluster labels.
        clustering_error = 0
        for i in range(num_generated_samples):
            u_l = torch.argmax(latent_clusters[i, :])
            u_g = torch.argmax(generated_clusters[i, :])

            if u_l != u_g:
                latent_centroid = self._clustered_transformer.get_cluster_center(u_l)
                # clustering_error += (generated_vectors[i] - latent_centroid) ** 2

        clustering_error /= num_generated_samples

        gen_loss = -torch.mean(critic_output) + discrete_columns_loss

        return gen_loss

    def discriminator_loss(self, real_data, generated_data):
        """Discriminator (Critic) loss function

        The loss function of the Critic measures the difference in the quality of the real and generated data.

        Args:
            real_data: The real data in the batch.
            generated_data: The data generated by the Generator.
        """

        real_data_q = torch.mean(self.D_(real_data))
        generated_data_q = torch.mean(self.D_(generated_data))

        return -(real_data_q - generated_data_q)

    def train_batch(self, real_data):
        """
        Given a batch of input data, `train_batch` updates the Critic and Generator weights using the respective
        optimizers and back propagation.

        Args:
            real_data: data for ctdGAN training: a batch of concatenated sample vectors + one-hot-encoded class vectors.
        """

        # If the size of the batch does not allow an organization of the input vectors in packs of size self.pac_, then
        # abort silently and return without updating the model parameters.
        num_samples = real_data.shape[0]
        if num_samples % self.pac_ != 0:
            return 0, 0

        packed_samples = num_samples // self.pac_

        # DISCRIMINATOR TRAINING
        real_data = real_data.to(torch.float).to(self._device)

        # Take samples from the latent space
        latent_vectors, latent_clusters, latent_classes = self.sample_latent_space(packed_samples)
        latent_data = torch.cat((latent_vectors, latent_clusters, latent_classes), dim=1).to(self._device)

        # Pass the latent samples through the Generator and generate fake samples.
        # Apply a dual activation function: tanh for the continuous columns, softmax for the discrete ones.
        generated_data = self._apply_activate(self.G_(latent_data))

        # Pass the mixed data to the Discriminator and train the Discriminator (update its weights with backprop).
        # The loss function quantifies the Discriminator's ability to classify a real/fake sample as real/fake.
        pen = self.D_.calc_gradient_penalty(real_data, generated_data, self._device, self.pac_)
        disc_loss = self.discriminator_loss(real_data, generated_data)

        self.D_optimizer_.zero_grad(set_to_none=False)
        pen.backward(retain_graph=True)
        disc_loss.backward()
        self.D_optimizer_.step()

        # GENERATOR TRAINING
        # Sample data from the latent spaces
        latent_vectors, latent_clusters, latent_classes = self.sample_latent_space(packed_samples)
        latent_data = torch.cat((latent_vectors, latent_clusters, latent_classes), dim=1).to(self._device)

        # Pass the latent data through the Generator to synthesize samples
        generated_data = self._apply_activate(self.G_(latent_data))

        # Reshape the data to feed it to Discriminator ( (num_samples, dimensionality) -> ( -1, pac * dimensionality )
        generated_data = generated_data.reshape((-1, self.pac_ * self._discrete_transformer.output_dimensions))

        # Compute and back propagate the Generator loss
        d_predictions = self.D_(generated_data)
        gen_loss = self.generator_loss(d_predictions, real_data, latent_data, generated_data)

        self.G_optimizer_.zero_grad(set_to_none=False)
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
        training_data = self.cluster_transform(x_train, y_train, discrete_columns=[])

        # ############################################
        # LIMITATION: The latent space must be of equal dimensionality as the data space
        self.embedding_dim_ = self._input_dim
        # ############################################

        train_dataloader = DataLoader(training_data, batch_size=batch_size, shuffle=True)

        # real_space_dimensions = self._input_dim + self._n_clusters + self._n_classes
        real_space_dimensions = self._discrete_transformer.output_dimensions
        latent_space_dimensions = self.embedding_dim_ + self._n_clusters + self._n_classes

        # self.D_ = PackedDiscriminator(self.D_Arch_, input_dim=real_space_dimensions, pac=self.pac_).to(self._device)
        self.D_ = Critic(input_dim=real_space_dimensions, discriminator_dim=self.D_Arch_,
                         pac=self.pac_).to(self._device)

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
        # Then build the latent sample vector by sampling the (Multivariate) Gaussian that describes this cluster.)
        for s in range(num_samples):
            lat_class = int(latent_classes[s])

            # Access the cluster probability matrix: Given a class, find a suitable cluster to sample from.
            p_matrix = self._clustered_transformer.probability_matrix_[lat_class]

            lat_cluster = np.random.choice(a=np.arange(self._n_clusters, dtype=int), size=1, replace=True, p=p_matrix)
            latent_cluster_object = self._clustered_transformer.get_cluster(int(lat_cluster))
            latent_clusters_objs.append(latent_cluster_object)
            latent_clusters[s] = latent_cluster_object.get_label()
            # print("Sample", s, "-- Class", int(latent_classes[s]), '--Cluster', int(lat_cluster), "(",
            #      latent_clusters[s], ")\n-- Prob Vec:", lat_subspace.get_cluster_probs())

            # latent_x derives by sampling the cluster's Multivariate Gaussian distribution
            z = latent_cluster_object.sample()
            latent_x.append(z)

        latent_x = torch.from_numpy(np.array(latent_x, dtype="float32"))

        latent_clusters = torch.from_numpy(latent_clusters).to(torch.int64)
        latent_clusters_ohe = nn.functional.one_hot(latent_clusters, num_classes=self._n_clusters)
        latent_y = torch.cat((latent_clusters_ohe, latent_classes_ohe), dim=1)

        # Concatenate, copy to device, and pass to generator
        latent_data = torch.cat((latent_x, latent_y), dim=1).to(self._device)

        # Generate data from the model's Generator - The activation function depends on the variable type:
        # - Hyperbolic tangent for continuous variables
        # - Softmax for discrete variables
        generated_data = self._apply_activate(self.G_(latent_data)).cpu().detach().numpy()

        # Throw away the generated cluster and generated class.
        # generated_samples = generated_data[:, 0:self.embedding_dim_].cpu().detach().numpy()
        generated_samples = self._discrete_transformer.inverse_transform(generated_data)[:, 0:self.embedding_dim_]

        # Inverse the transformation of the generated samples. First inverse the transformation of the continuous
        # variables that have been encoded according to the cluster the sample belongs.
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
        # self.train(x_train, y_train, store_losses=paths.output_path_loss)
        self.train(x_train, y_train, store_losses=None)

        x_resampled = np.copy(x_train)
        y_resampled = np.copy(y_train)

        # auto mode: Use sbGAN to equalize the number of samples per class. This is achieved by generating samples
        # of the minority classes (i.e. we perform oversampling).
        if self._sampling_strategy == 'auto':
            majority_class = np.array(self._samples_per_class).argmax()
            num_majority_samples = np.max(np.array(self._samples_per_class))

            # Perform oversampling
            for cls in range(self._n_classes):
                if cls != majority_class:
                    samples_to_generate = num_majority_samples - self._samples_per_class[cls]

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
