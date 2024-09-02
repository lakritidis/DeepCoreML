# BaseGenerator object
# Generative models and Over-sampling methods inherit from this class.

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

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
        sampling_strategy:
         - `auto`: perform oversampling on the minority classes to establish class imbalance in the dataset
         - `reproduce`: create a new dataset with the same class distribution as the input dataset
        random_state: An integer for seeding the involved random number generators.
    """
    def __init__(self, embedding_dim, discriminator, generator, pac, g_activation, adaptive, epochs, batch_size,
                 lr, decay, sampling_strategy, random_state):

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
        self._sampling_strategy = sampling_strategy  # Used in `fit_resample`: How GAN generates data

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

    @staticmethod
    def plot_losses(losses, store_losses):
        """
        Plot the Discriminator loss and the Generator loss values vs. the training epoch.
        Args:
            losses: A list of tuples (iteration, epoch, Discriminator loss, Generator loss) recorded during training.
            store_losses: The file path to store the plot.
        """
        custom_params = {'axes.facecolor': '#f0f0f0', 'axes.edgecolor': 'black', 'grid.color': '#ffffff',
                         'legend.labelspacing': 0.5, 'legend.fontsize': 15.0,
                         'axes.spines.right': False, 'axes.spines.top': False}
        sns.set_theme(rc=custom_params, font_scale=1.6)
        colors = ['#0072b0', '#ffa868']

        columns = ['Iteration', 'Epoch', 'Critic Loss it', 'Generator Loss it']
        df = pd.DataFrame(losses, columns=columns)
        df.to_csv(store_losses + 'ctdGAN_losses.csv', sep=';', decimal='.', index=False)

        df_mean = pd.DataFrame()
        df_mean['Critic Loss'] = df.groupby('Epoch')['Critic Loss it'].mean()
        df_mean['Generator Loss'] = df.groupby('Epoch')['Generator Loss it'].mean()
        df_mean['Epoch'] = df_mean.index + 1
        df_mean.to_csv(store_losses + 'ctdGAN_mean_losses.csv', sep=';', decimal='.', index=False)

        # plot = df.plot(x="Iteration", y=["Discriminator Loss", "Generator Loss"], ylim=(0, 1))
        plot = df_mean.plot(x='Epoch', y=['Critic Loss', 'Generator Loss'], kind='line', color=colors,
                            xlabel='Epoch', ylabel='Loss', title="ecoli3")

        fig = plot.get_figure()
        # fig.savefig(store_losses + 'GAN_losses.png')
        fig.savefig(store_losses + 'GAN_losses.pdf', format='pdf', bbox_inches='tight')

        plt.show()

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

        # Determine the class distribution of the input data samples.
        self._gen_samples_ratio = [int(sum(y_train[:, c])) for c in range(self._n_classes)]

        # Class specific training data.
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
