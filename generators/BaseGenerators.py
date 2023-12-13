import numpy as np

import torch
import torch.nn as nn

from sklearn.preprocessing import OneHotEncoder


class BaseGenerator:
    """`BaseGenerator` provides the base class for all generative models.

    Args:
        epochs: Number of training epochs.
        batch_size: Number of data instances per training batch.
        random_state: An integer for seeding the involved random number generators.
    """
    def __init__(self, epochs, batch_size, random_state):
        self.input_dim_ = 0                     # Input data dimensionality
        self.n_classes_ = 0                     # Number of classes in the input dataset
        self.random_state_ = random_state       # An integer to seed the random number generators
        self.gen_samples_ratio_ = None          # Array [number of samples to generate per class]
        self.x_train_per_class_ = None          # Array [ [x_train_per_class] ]

        self._epochs = epochs                   # Number of training epochs
        self._batch_size = batch_size           # Number of data instances per training batch

        self.device_ = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


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
        random_state: An integer for seeding the involved random number generators.
    """
    def __init__(self, embedding_dim, discriminator, generator, pac, adaptive, g_activation, epochs, batch_size,
                 lr, decay, random_state):

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

        self.input_dim_ = x_train.shape[1]
        self.n_classes_ = y_train.shape[1]

        # Determine how to sample the conditional GAN in smart training
        self.gen_samples_ratio_ = [int(sum(y_train[:, c])) for c in range(self.n_classes_)]
        # gen_samples_ratio.reverse()

        # Class specific training data for smart training (KL/JS divergence)
        self.x_train_per_class_ = []
        for y in range(self.n_classes_):
            x_class_data = np.array([x_train[r, :] for r in range(y_train.shape[0]) if y_train[r, y] == 1])
            x_class_data = torch.from_numpy(x_class_data).to(torch.float32).to(self.device_)

            self.x_train_per_class_.append(x_class_data)

        return training_data

    # Use GAN's Generator to create artificial samples i) either from a specific class, ii) or from a random class.
    def sample(self, num_samples, y=None):
        """ Create artificial samples using the GAN's Generator.

        Args:
            num_samples: The number of samples to generate.
            y: The class of the generated samples. If `None`, then samples with random classes are generated.

        Returns:
            Artificial data instances created by the Generator.
        """
        if y is None:
            latent_classes = torch.from_numpy(np.random.randint(0, self.n_classes_, num_samples)).to(torch.int64)
            latent_y = nn.functional.one_hot(latent_classes, num_classes=self.n_classes_)
        else:
            latent_y = nn.functional.one_hot(torch.full(size=(num_samples,), fill_value=y), num_classes=self.n_classes_)

        latent_x = torch.randn((num_samples, self.embedding_dim_))

        # concatenate, copy to device, and pass to generator
        latent_data = torch.cat((latent_x, latent_y), dim=1).to(self.device_)

        # Generate data from the model's Generator - The feature values of the generated samples fall into the range:
        # [-1,1]: if the activation function of the output layer of the Generator is nn.Tanh().
        # [0,1]: if the activation function of the output layer of the Generator is nn.Sigmoid().

        generated_samples = self.G_(latent_data).cpu().detach().numpy()
        # print("Generated Samples:\n", generated_samples)
        reconstructed_samples = self._transformer.inverse_transform(generated_samples)
        # print("Reconstructed samples\n", reconstructed_samples)
        return reconstructed_samples

    def base_fit_resample(self, x_train, y_train):
        generated_data = [None for _ in range(self.n_classes_)]

        majority_class = np.array(self.gen_samples_ratio_).argmax()
        num_majority_samples = np.max(np.array(self.gen_samples_ratio_))

        x_over_train = np.copy(x_train)
        y_over_train = np.copy(y_train)

        for cls in range(self.n_classes_):
            if cls != majority_class:
                samples_to_generate = num_majority_samples - self.gen_samples_ratio_[cls]

                # print("\tSampling Class y:", y, " Gen Samples ratio:", gen_samples_ratio[y])
                generated_data[cls] = self.sample(samples_to_generate, cls)

                min_classes = np.full(samples_to_generate, cls)

                x_over_train = np.vstack((x_over_train, generated_data[cls]))
                y_over_train = np.hstack((y_over_train, min_classes))

        # balanced_data = np.hstack((x_over_train, y_over_train.reshape((-1, 1))))
        # return balanced_data

        return x_over_train, y_over_train
