import numpy as np

import torch

from sklearn.preprocessing import OneHotEncoder


class BaseGenerator:
    """`BaseGenerator` provides the base class for all generative models.

    Args:
        epochs: Number of training epochs.
        batch_size: Number of data instances per training batch.
        random_state: An integer for seeding the involved random number generators.
    """
    def __init__(self, epochs, batch_size, random_state):
        self._input_dim = 0                     # Input data dimensionality
        self._n_classes = 0                     # Number of classes in the input dataset
        self._random_state = random_state       # An integer to seed the random number generators

        self._gen_samples_ratio = None          # Array [number of samples to generate per class]
        self._samples_per_class = None          # Array [ [x_train_per_class] ]

        self._epochs = epochs                   # Number of training epochs
        self._batch_size = batch_size           # Number of data instances per training batch

        self._device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class BaseGAN(BaseGenerator):
    """`BaseGAN` provides the base class for all GAN subclasses. It inherits from `BaseGenerator`.

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
        sampling_mode:
         - `balance`: perform oversampling on the minority classes to establish class imbalance in the dataset
         - `reproduce`: create a new dataset with the same class distribution as the input dataset
        random_state: An integer for seeding the involved random number generators.
    """
    def __init__(self, embedding_dim, discriminator, generator, pac, g_activation, adaptive, epochs, batch_size,
                 lr, decay, sampling_mode, random_state):

        super().__init__(epochs, batch_size, random_state)

        self.embedding_dim_ = embedding_dim     # Size of the random sample passed to the Generator.
        self.gen_activation_ = g_activation     # The activation function of the Generator's output layer
        self.batch_norm_ = True                 # Sets/unsets 1D-Batch Normalization in the Generator
        self.adaptive_ = adaptive               # Enables/Disables GAN adaptive training
        self.test_classifier_ = None            # Optional classification model for evaluating the GAN's effectiveness
        self.pac_ = pac                         # Number of samples to group together when applying the discriminator.
        self._lr = lr                           # Learning rate param for the Generator/Discriminator Adam optimizers.
        self._decay = decay                     # Weight decay param for the Generator/Discriminator Adam optimizers.
        self._transformer = None                # Input data transformer (normalizers)
        self._sampling_mode = sampling_mode     # Used in `fit_resample`: Given an input dataset, how GAN generates data

        # Discriminator parameters (object, architecture, optimizer)
        self.D_ = None
        self.D_Arch_ = discriminator
        self.D_optimizer_ = None

        # Generator parameters (object, architecture, optimizer)
        self.G_ = None
        self.G_Arch_ = generator
        self.G_optimizer_ = None

    def display_models(self):
        """Display the Generator and Discriminator objects."""
        self.D_.display()
        self.G_.display()

    def display_hyperparameters(self):
        info = ["BaseGenerator hyperparameters\n---------------------------------------------------",
                "\tEpochs" + str(self._epochs),
                "\tBatch Size" + str(self._batch_size),
                ]
        print(info)

    def prepare(self, x_train, y_train):
        """
        Data preparation function. Several auxiliary structures are built here (e.g. samples-per-class tensors, etc.) .

        Args:
            x_train: The training data instances.
            y_train: The classes of the training data instances.

        Returns:
            A tensor with the preprocessed data.
        """
        class_encoder = OneHotEncoder()
        y_train = class_encoder.fit_transform(y_train.reshape(-1, 1)).toarray()

        train_data = np.concatenate((x_train, y_train), axis=1)
        training_data = torch.from_numpy(train_data).to(torch.float32)

        self._input_dim = x_train.shape[1]
        self._n_classes = y_train.shape[1]

        # Determine how to sample the conditional GAN in smart training
        self._gen_samples_ratio = [int(sum(y_train[:, c])) for c in range(self._n_classes)]

        # Class specific training data for smart training (KL/JS divergence)
        self._samples_per_class = []
        for y in range(self._n_classes):
            x_class_data = np.array([x_train[r, :] for r in range(y_train.shape[0]) if y_train[r, y] == 1])
            x_class_data = torch.from_numpy(x_class_data).to(torch.float32).to(self._device)

            self._samples_per_class.append(x_class_data)

        return training_data

    def synthesize_dataset(self):
        i = 0
        x_synthetic, y_synthetic = None, None
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

        return x_synthetic, y_synthetic
