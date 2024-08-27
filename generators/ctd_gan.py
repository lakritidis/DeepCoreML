import numpy as np
import pandas as pd

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

    def __init__(self, discriminator=(128, 128), generator=(256, 256), embedding_dim=128, epochs=300, batch_size=32,
                 scaler='mms11', pac=1, lr=2e-4, decay=1e-6, sampling_strategy='auto', max_clusters=20, random_state=0):
        """
        ctdGAN initializer

        Args:
            discriminator (tuple): a tuple with number of neurons for each fully connected layer of the model's Critic.
                It determines the dimensionality of the output of each layer.
            generator (tuple): a tuple with number of neurons for each fully connected layer of the model's Generator.
                It determines the dimensionality of the output of each residual block of the Generator.
            embedding_dim (int): Size of the random sample passed to the Generator.
            epochs (int): The number of training epochs.
            batch_size (int): The number of data instances per training batch.
            scaler (string): A descriptor that defines a transformation on the cluster's data. Values:

              * '`None`'  : No transformation takes place; the data is considered immutable
              * '`stds`'  : Standard scaler
              * '`mms01`' : Min-Max scaler in the range (0,1)
              * '`mms11`' : Min-Max scaler in the range (-1,1) - so that data is suitable for tanh activations
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

        if scaler != 'mms11' and scaler != 'mms01' and scaler != 'stds':
            self._scaler = 'mms11'
        else:
            self._scaler = scaler

        # clustered_transformer performs clustering and (optionally), data transformation
        self._clustered_transformer = None

        # discrete_transformer performs dataset-wise one-hot-encoding of the categorical columns
        self._discrete_transformer = None

        self._max_clusters = max_clusters
        self._n_clusters = 0
        self._categorical_columns = []

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

    def cluster_transform(self, x_train, y_train, categorical_columns):
        """
        Perform clustering and (optionally) transform the data in the generated clusters.

        Args:
            x_train: The training data instances.
            y_train: The classes of the training data instances.
            categorical_columns: The discrete columns in the dataset. The last 2 columns indicate the cluster and class.

        Returns:
            A tensor with the preprocessed data.
        """
        self._categorical_columns = list(categorical_columns)
        self._n_classes = len(set(y_train))
        self._input_dim = x_train.shape[1]
        continuous_columns = [c for c in range(self._input_dim) if c not in self._categorical_columns]

        # ====== Initialize and deploy the Clustered Transformer object that: i) partitions the real space, and
        # ====== ii) performs data transformations (scaling, PCA, outlier detection, etc.)
        self._samples_per_class = np.unique(y_train, return_counts=True)[1]

        self._clustered_transformer = ctdClusterer(max_clusters=self._max_clusters, scaler=self._scaler,
                                                   samples_per_class=self._samples_per_class,
                                                   continuous_columns=tuple(continuous_columns),
                                                   discrete_columns=tuple(self._categorical_columns),
                                                   embedding_dim=self.embedding_dim_, random_state=self._random_state)

        train_data = self._clustered_transformer.perform_clustering(x_train, y_train, self._n_classes)
        self._n_clusters = self._clustered_transformer.num_clusters_

        # ====== Append the cluster and class labels to the collection of discrete columns
        self._categorical_columns.append(self._input_dim)
        self._categorical_columns.append(self._input_dim + 1)

        # ====== Transform the discrete columns only; the continuous columns have been scaled at cluster-level.
        self._discrete_transformer = TabularTransformer(cont_normalizer='none', clip=False)
        self._discrete_transformer.fit(train_data, self._categorical_columns)

        ret_data = self._discrete_transformer.transform(train_data)

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
        num_cols = len(self._discrete_transformer.output_info_list)
        col = 0

        # === Discrete variables - Simple random one-hot-encoded integers
        latent_disc = []
        latent_clusters = []
        for column_metadata in self._discrete_transformer.output_info_list:
            for span_info in column_metadata:
                col += 1
                if span_info.activation_fn == 'softmax':
                    col_length = span_info.dim

                    # random_numbers = torch.from_numpy(np.random.randint(0, col_length, num_samples)).to(torch.int64)
                    random_numbers = torch.randint(low=0, high=col_length, size=(1, num_samples))[0]
                    z_disc = nn.functional.one_hot(random_numbers, num_classes=col_length)

                    if col == num_cols - 1:
                        latent_clusters = random_numbers.to(torch.int64)
                    latent_disc.append(z_disc)

        latent_disc = torch.hstack(latent_disc).to(self._device)

        # === Continuous variables - Sample from the corresponding Normal distribution of each cluster.
        latent_cont = []
        for s in range(num_samples):
            latent_cluster_object = self._clustered_transformer.get_cluster(latent_clusters[s])
            z_cont = latent_cluster_object.sample()
            latent_cont.append(z_cont)

        # Convert the list of the latent vectors (tensors) to a pytorch tensor:
        latent_cont = torch.stack(latent_cont).to(self._device)

        return latent_cont, latent_disc

    def generator_loss(self, critic_output, latent_disc_data, generated_data):
        """Generator loss function

        The loss function of the Generator combines the Discriminator Loss, and the mis-predictions of
        the cluster and class labels.

        Args:
            critic_output: The output of the Discriminator.
            latent_disc_data: The discrete latent data where the Generator sampled from.
            generated_data: The data generated by the Generator.
        """
        num_generated_samples = generated_data.size()[0]

        cluster_st_idx = self._discrete_transformer.output_dimensions - self._n_clusters - self._n_classes
        cluster_ed_idx = cluster_st_idx + self._n_clusters
        # print("Cluster st:", cluster_st_idx, ", Cluster ed:", cluster_ed_idx)

        class_st_idx = self._discrete_transformer.output_dimensions - self._n_classes
        class_ed_idx = class_st_idx + self._n_classes
        # print("Class st:", class_st_idx, ", Class ed:", class_ed_idx)

        # Loss for the discrete columns - Including the class & cluster labels.
        # print("Real data shape", real_data.shape, " - Latent data shape", latent_data.shape)
        discrete_loss = []
        st_idx = 0
        st_disc_idx = 0
        for column_metadata in self._discrete_transformer.output_info_list:
            for span_info in column_metadata:
                if len(column_metadata) != 1 or span_info.activation_fn != 'softmax':
                    # Continuous column
                    st_idx += span_info.dim
                else:
                    ed_idx = st_idx + span_info.dim
                    ed_disc_idx = st_disc_idx + span_info.dim

                    # print("\t== Latent Discrete Data:\nStart:", st_disc_idx, ", End:", ed_disc_idx, "\nData:\n",
                    #       latent_disc_data[:, st_disc_idx:ed_disc_idx])
                    # print("\t== Generated Discrete Data:\nStart:", st_idx, ", End:", ed_idx, "\nData:\n",
                    #       generated_data[:, st_idx:ed_idx])

                    gen_d = generated_data[:, st_idx:ed_idx]
                    gen_c = torch.argmax(gen_d, dim=1)

                    lat_d = latent_disc_data[:, st_disc_idx:ed_disc_idx].to(dtype=torch.float)
                    lat_c = torch.argmax(lat_d, dim=1)

                    # Penalize mis-clusters more heavily and in a dynamic manner
                    if st_idx == cluster_st_idx and ed_idx == cluster_ed_idx:
                        # print("Lat Clusters:\n", lat_d, "(", lat_c, ")", "\nGen Clusters:\n", gen_d, "(", gen_c, ")")

                        mis_clustering = np.sum([1 for i in range(num_generated_samples) if gen_c[i] != lat_c[i]])
                        beta = 1.0

                        if self._n_clusters == 2:
                            tmp = beta * nn.functional.binary_cross_entropy(gen_d, lat_d, reduction='none')[:, 0]
                        else:
                            tmp = beta * nn.functional.cross_entropy(gen_d, lat_c, reduction='none')
                        # tmp = torch.zeros((num_generated_samples, )).to(self._device)
                        # print(beta)

                    # Penalize mis-classifications more heavily
                    elif st_idx == class_st_idx and ed_idx == class_ed_idx:
                        mis_classified = np.sum([1 for i in range(num_generated_samples) if gen_c[i] != lat_c[i]])

                        # gamma = len(self._categorical_columns) * (1.0 + mis_classified / num_generated_samples)
                        gamma = len(self._categorical_columns)
                        if self._n_classes == 2:
                            tmp = gamma * nn.functional.binary_cross_entropy(gen_d, lat_d, reduction='none')[:, 0]
                        else:
                            tmp = gamma * nn.functional.cross_entropy(gen_d, lat_c, reduction='none')
                    else:
                        tmp = nn.functional.cross_entropy(gen_d, lat_c, reduction='none')

                    discrete_loss.append(tmp)

                    st_idx = ed_idx
                    st_disc_idx = ed_disc_idx

        cond_loss = torch.stack(discrete_loss, dim=1)
        discrete_columns_loss = cond_loss.sum() / num_generated_samples
        # print("discrete_columns_loss = ", discrete_columns_loss)

        gen_loss = -torch.mean(critic_output) + discrete_columns_loss

        return gen_loss

    def discriminator_loss(self, real_data, generated_data):
        """Discriminator (Critic) loss function

        The loss function of the Critic measures the difference in the quality of the real and generated data.

        Args:
            real_data: The real data in the batch.
            generated_data: The data generated by the Generator.
        """

        real_data_quality = torch.mean(self.D_(real_data))
        generated_data_quality = torch.mean(self.D_(generated_data))

        return -(real_data_quality - generated_data_quality)

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

        real_data = real_data.to(torch.float).to(self._device)

        # DISCRIMINATOR TRAINING
        with torch.no_grad():
            # Take samples from the latent space
            latent_cont, latent_disc = self.sample_latent_space(packed_samples)
            latent_data = torch.cat((latent_cont, latent_disc), dim=1)

        # Pass the latent samples through the Generator and generate fake samples.
        # Apply a dual activation function: tanh for the continuous columns, softmax for the discrete ones.
        generated_data = self._apply_activate(self.G_(latent_data))

        # Pass the mixed data to the Discriminator and train the Discriminator (update its weights with backprop).
        # The loss function quantifies the Discriminator's ability to classify a real/fake sample as real/fake.
        pen = self.D_.calc_gradient_penalty(real_data, generated_data, self._device, self.pac_)
        disc_loss = self.discriminator_loss(real_data, generated_data)

        self.D_optimizer_.zero_grad(set_to_none=True)
        pen.backward(retain_graph=True)
        disc_loss.backward()
        self.D_optimizer_.step()

        # GENERATOR TRAINING
        # Sample data from the latent spaces
        with torch.no_grad():
            latent_cont, latent_disc = self.sample_latent_space(packed_samples)
            latent_data = torch.cat((latent_cont, latent_disc), dim=1)

        # Pass the latent data through the Generator to synthesize samples
        generated_data = self._apply_activate(self.G_(latent_data))

        # Reshape the data to feed it to Discriminator ( (num_samples, dimensionality) -> ( -1, pac * dimensionality )
        # generated_data = generated_data.reshape((-1, self.pac_ * self._discrete_transformer.output_dimensions))

        # Compute and back propagate the Generator loss
        d_predictions = self.D_(generated_data)
        gen_loss = self.generator_loss(d_predictions, latent_disc, generated_data)

        self.G_optimizer_.zero_grad(set_to_none=True)
        gen_loss.backward()
        self.G_optimizer_.step()

        return disc_loss, gen_loss

    def train(self, x_train, y_train, categorical_columns=(), store_losses=None):
        """
        Conventional training process of a Cluster GAN. The Generator and the Discriminator are trained
        simultaneously in the traditional adversarial fashion by optimizing `loss_function`.

        Args:
            x_train: The training data instances (NumPy array).
            y_train: The classes of the training data instances (NumPy array).
            categorical_columns: The columns to be considered as categorical
            store_losses: The file path where the values of the Discriminator and Generator loss functions are stored.
        """

        # Modify the size of the batch to align with self.pac_
        factor = self._batch_size // self.pac_
        batch_size = factor * self.pac_

        # Prepare the data for training (Clustering, Computation of Probability Distributions, Transformations, etc.)
        training_data = self.cluster_transform(x_train, y_train, categorical_columns=categorical_columns)
        train_dataloader = DataLoader(training_data, batch_size=batch_size, shuffle=True)

        real_space_dimensions = self._discrete_transformer.output_dimensions
        latent_space_dimensions = self.embedding_dim_ + self._discrete_transformer.ohe_dimensions

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
        """`fit` invokes the GAN training process. `fit` renders the ctdGAN class compatible with `imblearn`'s
        interface, allowing its usage in over-sampling/under-sampling pipelines.

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
        num_cols = len(self._discrete_transformer.output_info_list)
        column_transform_info_list = self._discrete_transformer.get_column_transform_info_list()

        # If no specific class is required, pick classes randomly.
        if y is None:
            latent_classes = np.random.randint(low=0, high=self._n_classes, size=num_samples)
        # Otherwise, fill the classes tensor with the requested class (y) value
        else:
            latent_classes = np.full(shape=num_samples, fill_value=y)

        latent_clusters = np.zeros(shape=num_samples)

        # Select random integer values for the discrete variables. These values will be later one-hot-encoded.
        latent_disc = []
        latent_cont = []
        col = 0
        column_labels = []
        for column_metadata in self._discrete_transformer.output_info_list:
            col = col + 1
            for span_info in column_metadata:
                if span_info.activation_fn == 'softmax':
                    column_labels.append(str(col - 1))
                    col_length = span_info.dim

                    if col < num_cols - 1:
                        random_discrete_vals = np.random.randint(low=0, high=col_length, size=num_samples)
                        latent_disc.append(random_discrete_vals)

        # For each sample with a specific class, pick a cluster with probability lat_subspace.get_cluster_probs().
        # In the same time, sample the probability distribution of each cluster to get the latent representation of
        # the continuous variables.
        latent_clusters_objs = []
        for s in range(num_samples):
            lat_class = int(latent_classes[s])
            p_matrix = self._clustered_transformer.probability_matrix_[lat_class]

            # Select the cluster with probability self._cc_prob_vector
            latent_clusters[s] = np.random.choice(
                a=np.arange(self._n_clusters, dtype=int), size=None, replace=True, p=p_matrix)

            latent_cluster_object = self._clustered_transformer.get_cluster(int(latent_clusters[s]))
            latent_clusters_objs.append(latent_cluster_object)

            latent_cont.append(latent_cluster_object.sample())

        # Put all discrete variables together into the same matrix (including the class and cluster labels)
        latent_disc.append(latent_clusters)
        latent_disc.append(latent_classes)
        latent_disc = pd.DataFrame(np.stack(latent_disc, axis=1), columns=column_labels)

        # Now one-hot-encode the discrete variables by using the OneHotEncoders that were used during training
        latent_disc_ohe = []
        for column_transform_info in column_transform_info_list:
            if column_transform_info.column_type != 'continuous':
                column_name = column_transform_info.column_name
                data = latent_disc[[column_name]]
                one_hot_data = self._discrete_transformer.transform_discrete(column_transform_info, data)
                latent_disc_ohe.append(one_hot_data)

        # Create the discrete and continuous tensors.
        latent_disc_ohe = torch.tensor(np.hstack(latent_disc_ohe))
        latent_cont = torch.tensor(np.vstack(latent_cont))

        # Concatenate the continuous with the discrete variables
        latent_data = torch.cat((latent_cont, latent_disc_ohe), dim=1).to(self._device)
        # print(latent_classes)
        # print("Latent:", latent_data)

        # Generate data from the model's Generator - The activation function depends on the variable type:
        # - Hyperbolic tangent for continuous variables
        # - Softmax for discrete variables
        generated_data = self._apply_activate(self.G_(latent_data)).cpu().detach().numpy()
        generated_samples = self._discrete_transformer.inverse_transform(generated_data)

        # Inverse the transformation of the generated samples. First inverse the transformation of the continuous
        # variables that have been encoded according to the cluster the sample belongs.
        reconstructed_samples = []
        correct_classes, correct_clusters = 0, 0
        for s in range(num_samples):
            z = generated_samples[s].reshape(1, -1)

            generated_class = z[0, z.shape[1]-1]
            generated_cluster = z[0, z.shape[1]-2]

            if generated_class == y:
                correct_classes += 1
            if generated_cluster == latent_clusters[s]:
                correct_clusters += 1

            reconstructed_sample = latent_clusters_objs[s].inverse_transform(z)
            # print("Sample", s, "- Gen:", z, " ===>", reconstructed_sample)

            reconstructed_samples.append(reconstructed_sample)

        reconstructed_samples = np.vstack(reconstructed_samples)
        # print("Reconstructed samples\n", reconstructed_samples)

        print("Correct Clusters: ", correct_clusters, " == ", "Correct Classes:", correct_classes)

        return reconstructed_samples

    def fit_resample(self, x_train, y_train, categorical_columns=()):
        """`fit_resample` alleviates the problem of class imbalance in imbalanced datasets. The function renders sbGAN
        compatible with the `imblearn`'s interface, allowing its usage in over-sampling/under-sampling pipelines.

        In the `fit` part, the input dataset is used to train sbGAN. In the `resample` part, sbGAN is employed to
        generate synthetic data according to the value of `self._sampling_strategy`:

        - 'auto': the model balances the dataset by oversampling the minority classes.
        - dict: a dictionary that indicates the number of samples to be generated from each class

        Args:
            x_train: The training data instances.
            y_train: The classes of the training data instances.
            categorical_columns: The columns to be considered as categorical

        Returns:
            x_resampled: The training data instances + the generated data instances.
            y_resampled: The classes of the training data instances + the classes of the generated data instances.
        """

        # Train the GAN with the input data
        self.train(x_train, y_train, categorical_columns=categorical_columns, store_losses=paths.output_path_loss)
        # self.train(x_train, y_train, categorical_columns=categorical_columns, store_losses=None)

        x_resampled = np.copy(x_train)
        y_resampled = np.copy(y_train)

        # auto mode: Use ctdGAN to equalize the number of samples per class. This is achieved by generating samples
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
