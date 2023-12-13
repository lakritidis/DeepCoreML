import warnings

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from sklearn.preprocessing import OneHotEncoder

from .DataTransformers import DataTransformer
# from .gan_discriminators import PackedDiscriminator
from .gan_discriminators import ctDiscriminator
from .gan_generators import ctGenerator
from .BaseGenerators import BaseGAN

np.set_printoptions(threshold=np.inf, linewidth=250, precision=3, suppress=True,)


class DataSampler(object):
    """DataSampler samples the conditional vector and corresponding data for CTGAN."""

    def __init__(self, data, output_info, log_frequency):
        self._data = data

        def is_discrete_column(column_info):
            return len(column_info) == 1 and column_info[0].activation_fn == 'softmax'

        n_discrete_columns = sum([1 for column_info in output_info if is_discrete_column(column_info)])

        self._discrete_column_matrix_st = np.zeros(n_discrete_columns, dtype='int32')

        # Store the row id for each category in each discrete column. For example _rid_by_cat_cols[a][b]
        # is a list of all rows with the a-th discrete column equal value b.
        self._rid_by_cat_cols = []

        # Compute _rid_by_cat_cols
        st = 0
        for column_info in output_info:
            if is_discrete_column(column_info):
                span_info = column_info[0]
                ed = st + span_info.dim

                rid_by_cat = []
                for j in range(span_info.dim):
                    rid_by_cat.append(np.nonzero(data[:, st + j])[0])
                self._rid_by_cat_cols.append(rid_by_cat)
                st = ed
            else:
                st += sum([span_info.dim for span_info in column_info])
        assert st == data.shape[1]

        # Prepare an interval matrix for efficiently sample conditional vector
        max_category = max([column_info[0].dim for column_info in output_info if is_discrete_column(column_info)],
                           default=0)

        self._discrete_column_cond_st = np.zeros(n_discrete_columns, dtype='int32')
        self._discrete_column_n_category = np.zeros(n_discrete_columns, dtype='int32')
        self._discrete_column_category_prob = np.zeros((n_discrete_columns, max_category))
        self._n_discrete_columns = n_discrete_columns
        self._n_categories = sum([column_info[0].dim for column_info in output_info if is_discrete_column(column_info)])

        st = 0
        current_id = 0
        current_cond_st = 0
        for column_info in output_info:
            if is_discrete_column(column_info):
                span_info = column_info[0]
                ed = st + span_info.dim
                category_freq = np.sum(data[:, st:ed], axis=0)
                if log_frequency:
                    category_freq = np.log(category_freq + 1)
                category_prob = category_freq / np.sum(category_freq)
                self._discrete_column_category_prob[current_id, :span_info.dim] = category_prob
                self._discrete_column_cond_st[current_id] = current_cond_st
                self._discrete_column_n_category[current_id] = span_info.dim
                current_cond_st += span_info.dim
                current_id += 1
                st = ed
            else:
                st += sum([span_info.dim for span_info in column_info])

    def _random_choice_prob_index(self, discrete_column_id):
        probs = self._discrete_column_category_prob[discrete_column_id]
        r = np.expand_dims(np.random.rand(probs.shape[0]), axis=1)
        return (probs.cumsum(axis=1) > r).argmax(axis=1)

    def sample_condvec(self, batch_size):
        """Generate the conditional vector for training.

        Returns:
            cond: The conditional vector (batch x categories).
            mask: A one-hot vector indicating the selected discrete column (batch x discrete columns).
            discrete column id: Integer representation of mask (batch).
            category_id_in_col: Selected category in the selected discrete column (batch).
        """
        if self._n_discrete_columns == 0:
            return None

        discrete_column_id = np.random.choice(np.arange(self._n_discrete_columns), batch_size)

        cond = np.zeros((batch_size, self._n_categories), dtype='float32')
        mask = np.zeros((batch_size, self._n_discrete_columns), dtype='float32')
        mask[np.arange(batch_size), discrete_column_id] = 1
        category_id_in_col = self._random_choice_prob_index(discrete_column_id)
        category_id = (self._discrete_column_cond_st[discrete_column_id] + category_id_in_col)
        cond[np.arange(batch_size), category_id] = 1

        # print(cond)
        return cond, mask, discrete_column_id, category_id_in_col

    def sample_original_condvec(self, batch):
        """Generate the conditional vector for generation use original frequency."""
        if self._n_discrete_columns == 0:
            return None

        cond = np.zeros((batch, self._n_categories), dtype='float32')

        for i in range(batch):
            row_idx = np.random.randint(0, len(self._data))
            col_idx = np.random.randint(0, self._n_discrete_columns)
            matrix_st = self._discrete_column_matrix_st[col_idx]
            matrix_ed = matrix_st + self._discrete_column_n_category[col_idx]
            pick = np.argmax(self._data[row_idx, matrix_st:matrix_ed])
            cond[i, pick + self._discrete_column_cond_st[col_idx]] = 1

        return cond

    def sample_data(self, n, col, opt):
        """Sample data from original training data satisfying the sampled conditional vector.

        Returns:
            n rows of matrix data.
        """
        if col is None:
            idx = np.random.randint(len(self._data), size=n)
            return self._data[idx]

        idx = []
        for c, o in zip(col, opt):
            idx.append(np.random.choice(self._rid_by_cat_cols[c][o]))

        return self._data[idx]

    def dim_cond_vec(self):
        """Return the total number of categories."""
        return self._n_categories

    def generate_cond_from_condition_column_info(self, condition_info, batch):
        """Generate the condition vector."""
        vec = np.zeros((batch, self._n_categories), dtype='float32')
        id_ = self._discrete_column_matrix_st[condition_info['discrete_column_id']]
        id_ += condition_info['value_id']
        vec[:, id_] = 1
        return vec


class ctGAN(BaseGAN):
    """Conditional Table GAN Synthesizer.

    This is the core class of the CTGAN project, where the different components
    are orchestrated together.
    For more details about the process, please check the [Modeling Tabular data using
    Conditional GAN](https://arxiv.org/abs/1907.00503) paper.

    Args:
        embedding_dim (int): Size of the random sample passed to the Generator. Defaults to 128.
        generator (tuple or list of ints): Size of the output samples for each one of the Residuals. A Residual
            Layer will be created for each one of the values provided. Defaults to (256, 256).
        discriminator (tuple or list of ints): Size of the output samples for each one of the Discriminator Layers.
            A Linear Layer will be created for each one of the values provided. Defaults to (256, 256).
        lr: Learning rate parameter for the Generator/Discriminator Adam optimizers.
        decay: Weight decay parameter for the Generator/Discriminator Adam optimizers.
        batch_size (int):
            Number of data samples to process in each step.
        discriminator_steps (int):
            Number of discriminator updates to do for each generator update. From the WGAN paper:
            https://arxiv.org/abs/1701.07875. WGAN paper default is 5.
            Default used is 1 to match original CTGAN implementation.
        log_frequency (boolean):
            Whether to use log frequency of categorical levels in conditional sampling. Defaults to ``True``.
        verbose (boolean):
            Whether to have print statements for progress results. Defaults to ``False``.
        epochs (int):
            Number of training epochs. Defaults to 300.
        pac (int):
            Number of samples to group together when applying the discriminator. Defaults to 10.
    """
    def __init__(self, embedding_dim=128, generator=(256, 256), discriminator=(256, 256), pac=10, adaptive=False,
                 g_activation=None, epochs=300, batch_size=32, lr=2e-4, decay=1e-6, discriminator_steps=1,
                 log_frequency=True, verbose=False, random_state=42):

        super().__init__(embedding_dim, discriminator, generator, pac, adaptive, g_activation, epochs, batch_size,
                         lr, decay, random_state)

        assert batch_size % 2 == 0

        self._discriminator_steps = discriminator_steps
        self._log_frequency = log_frequency
        self._verbose = verbose

        self._data_sampler = None

    @staticmethod
    def _gumbel_softmax(logits, tau=1.0, hard=False, eps=1e-10, dim=-1):
        """Deals with the instability of the gumbel_softmax for older versions of torch.

        For more details about the issue:
        https://drive.google.com/file/d/1AA5wPfZ1kquaRtVruCd6BiYZGcDeNxyP/view?usp=sharing

        Args:
            logits (array(…, num_features)): Unnormalized log probabilities
            tau: Non-negative scalar temperature/
            hard (bool): If True, the returned samples will be discretized as one-hot vectors,
                but will be differentiated as if it is the soft sample in autograd
            dim (int): A dimension along which softmax will be computed. Default: -1.

        Returns:
            Sampled tensor of same shape as logits from the Gumbel-Softmax distribution.
        """
        for _ in range(10):
            transformed = nn.functional.gumbel_softmax(logits, tau=tau, hard=hard, eps=eps, dim=dim)
            if not torch.isnan(transformed).any():
                return transformed

        raise ValueError('gumbel_softmax returning NaN.')

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
                    transformed = self._gumbel_softmax(data[:, st:ed], tau=0.2)
                    data_t.append(transformed)
                    st = ed
                else:
                    raise ValueError(f'Unexpected activation function {span_info.activation_fn}.')

        return torch.cat(data_t, dim=1)

    def _cond_loss(self, data, c, m):
        """Compute the cross entropy loss on the fixed discrete column."""
        loss = []
        st = 0
        st_c = 0
        for column_info in self._transformer.output_info_list:
            for span_info in column_info:
                if len(column_info) != 1 or span_info.activation_fn != 'softmax':
                    # not discrete column
                    st += span_info.dim
                else:
                    ed = st + span_info.dim
                    ed_c = st_c + span_info.dim
                    tmp = nn.functional.cross_entropy(data[:, st:ed], torch.argmax(c[:, st_c:ed_c], dim=1),
                                                      reduction='none')
                    loss.append(tmp)
                    st = ed
                    st_c = ed_c

        loss = torch.stack(loss, dim=1)  # noqa: PD013

        return (loss * m).sum() / data.size()[0]

    def _validate_discrete_columns(self, train_data, discrete_columns):
        """Check whether ``discrete_columns`` exists in ``train_data``.

        Args:
            train_data (numpy.ndarray or pandas.DataFrame):
                Training Data. It must be a 2-dimensional numpy array or a pandas.DataFrame.
            discrete_columns (list-like):
                List of discrete columns to be used to generate the Conditional Vector. If ``train_data`` is a
                Numpy array, this list should contain the integer indices of the columns. Otherwise, if it is
                a ``pandas.DataFrame``, this list should contain the column names.
        """
        if isinstance(train_data, pd.DataFrame):
            invalid_columns = set(discrete_columns) - set(train_data.columns)
        elif isinstance(train_data, np.ndarray):
            invalid_columns = []
            for column in discrete_columns:
                if column < 0 or column >= train_data.shape[1]:
                    invalid_columns.append(column)
        else:
            raise TypeError('``train_data`` should be either pd.DataFrame or np.array.')

        if invalid_columns:
            raise ValueError(f'Invalid columns found: {invalid_columns}')

    def train(self, train_data, discrete_columns=(), epochs=None):
        """Fit the CTGAN Synthesizer models to the training data.

        Args:
            train_data (numpy.ndarray or pandas.DataFrame):
                Training Data. It must be a 2-dimensional numpy array or a pandas.DataFrame.
            discrete_columns (list-like):
                List of discrete columns to be used to generate the Conditional Vector. If ``train_data`` is a
                numpy array, this list should contain the integer indices of the columns. Otherwise, if it is
                a ``pandas.DataFrame``, this list should contain the column names.
            epochs: deprecated
        """
        self._validate_discrete_columns(train_data, discrete_columns)

        if epochs is None:
            epochs = self._epochs
        else:
            warnings.warn(('`epochs` argument in `fit` method has been deprecated and will be removed '
                           'in a future version. Please pass `epochs` to the constructor instead'), DeprecationWarning)

        self._transformer = DataTransformer(cont_normalizer='gm')
        self._transformer.fit(train_data, discrete_columns)

        # print(self._transformer._column_transform_info_list)
        # print(train_data.shape, "\n", train_data)

        # TRAINING DATA
        train_data = self._transformer.transform(train_data)
        # print(train_data.shape, "\n", train_data)

        self._data_sampler = DataSampler(train_data, self._transformer.output_info_list, self._log_frequency)

        data_dim = self._transformer.output_dimensions

        # CtGAN components: ctGenerator & ctDiscriminator
        self.G_ = ctGenerator(self.embedding_dim_ + self._data_sampler.dim_cond_vec(), self.G_Arch_,
                              data_dim).to(self.device_)
        # self.D_ = PackedDiscriminator(self.D_Arch_, input_dim=data_dim + self._data_sampler.dim_cond_vec(),
        #                              pac=self.pac_, p=0.5, negative_slope=0.2).to(self.device_)

        self.D_ = ctDiscriminator(data_dim + self._data_sampler.dim_cond_vec(), self.D_Arch_,
                                  pac=self.pac_).to(self.device_)

        self.D_optimizer_ = torch.optim.Adam(self.D_.parameters(),
                                             lr=self._lr, weight_decay=self._decay, betas=(0.5, 0.9))
        self.G_optimizer_ = torch.optim.Adam(self.G_.parameters(),
                                             lr=self._lr, weight_decay=self._decay, betas=(0.5, 0.9))

        mean = torch.zeros(self._batch_size, self.embedding_dim_, device=self.device_)
        std = mean + 1

        steps_per_epoch = max(len(train_data) // self._batch_size, 1)
        loss_d = loss_g = 0
        c2 = 0
        for i in range(epochs):
            for id_ in range(steps_per_epoch):

                for n in range(self._discriminator_steps):
                    fakez = torch.normal(mean=mean, std=std)

                    condvec = self._data_sampler.sample_condvec(self._batch_size)
                    if condvec is None:
                        c1, m1, col, opt = None, None, None, None
                        real = self._data_sampler.sample_data(self._batch_size, col, opt)
                    else:
                        c1, m1, col, opt = condvec
                        c1 = torch.from_numpy(c1).to(self.device_)
                        m1 = torch.from_numpy(m1).to(self.device_)
                        fakez = torch.cat([fakez, c1], dim=1)

                        perm = np.arange(self._batch_size)
                        np.random.shuffle(perm)
                        real = self._data_sampler.sample_data(self._batch_size, col[perm], opt[perm])
                        c2 = c1[perm]

                    fake = self.G_(fakez)
                    fakeact = self._apply_activate(fake)

                    real = torch.from_numpy(real.astype('float32')).to(self.device_)

                    if c1 is not None:
                        fake_cat = torch.cat([fakeact, c1], dim=1)
                        real_cat = torch.cat([real, c2], dim=1)
                    else:
                        real_cat = real
                        fake_cat = fakeact

                    y_fake = self.D_(fake_cat)
                    y_real = self.D_(real_cat)

                    pen = self.D_.calc_gradient_penalty(real_cat, fake_cat, self.device_, self.pac_)
                    loss_d = -(torch.mean(y_real) - torch.mean(y_fake))

                    self.D_optimizer_.zero_grad(set_to_none=False)
                    pen.backward(retain_graph=True)
                    loss_d.backward()
                    self.D_optimizer_.step()

                fakez = torch.normal(mean=mean, std=std)
                condvec = self._data_sampler.sample_condvec(self._batch_size)

                if condvec is None:
                    c1, m1, col, opt = None, None, None, None
                else:
                    c1, m1, col, opt = condvec
                    c1 = torch.from_numpy(c1).to(self.device_)
                    m1 = torch.from_numpy(m1).to(self.device_)
                    fakez = torch.cat([fakez, c1], dim=1)

                fake = self.G_(fakez)
                fakeact = self._apply_activate(fake)

                if c1 is not None:
                    y_fake = self.D_(torch.cat([fakeact, c1], dim=1))
                else:
                    y_fake = self.D_(fakeact)

                if condvec is None:
                    cross_entropy = 0
                else:
                    cross_entropy = self._cond_loss(fake, c1, m1)

                loss_g = -torch.mean(y_fake) + cross_entropy

                self.G_optimizer_.zero_grad(set_to_none=False)
                loss_g.backward()
                self.G_optimizer_.step()

            if self._verbose:
                print(f'Epoch {i+1}, Loss G: {loss_g.detach().cpu(): .4f},'
                      f'Loss D: {loss_d.detach().cpu(): .4f}', flush=True)

    def sample(self, n, condition_column=None, condition_value=None):
        """Sample data similar to the training data.

        Choosing a condition_column and condition_value will increase the probability of the
        discrete condition_value happening in the condition_column.

        Args:
            n (int): Number of rows to sample.
            condition_column: Name of a discrete column.
            condition_value: Name of the category in the condition_column which we wish to increase the
                probability of happening.

        Returns:
            numpy.ndarray or pandas.DataFrame
        """
        if condition_column is not None and condition_value is not None:
            condition_info = self._transformer.convert_column_name_value_to_id(condition_column, condition_value)
            global_condition_vec = self._data_sampler.generate_cond_from_condition_column_info(
                condition_info, self._batch_size)
        else:
            global_condition_vec = None

        steps = n // self._batch_size + 1
        data = []
        for i in range(steps):
            mean = torch.zeros(self._batch_size, self.embedding_dim_)
            std = mean + 1
            fakez = torch.normal(mean=mean, std=std).to(self.device_)

            if global_condition_vec is not None:
                condvec = global_condition_vec.copy()
            else:
                condvec = self._data_sampler.sample_original_condvec(self._batch_size)

            if condvec is None:
                pass
            else:
                c1 = condvec
                c1 = torch.from_numpy(c1).to(self.device_)
                fakez = torch.cat([fakez, c1], dim=1)

            fake = self.G_(fakez)
            # print("fake:\n", fake)
            fakeact = self._apply_activate(fake)
            # print("fakeact:\n", fakeact)
            data.append(fakeact.detach().cpu().numpy())

        data = np.concatenate(data, axis=0)
        data = data[:n]

        return self._transformer.inverse_transform(data)

    def set_device(self, device):
        """Set the `device` to be used ('GPU' or 'CPU)."""
        self.device_ = device
        if self.G_ is not None:
            self.G_.to(self.device_)

    def fit_resample(self, x_train, y_train):
        # Create the ctGAN training data
        training_data = np.concatenate((x_train, y_train.reshape((-1, 1))), axis=1)

        self.input_dim_ = x_train.shape[1]

        # Train the ctGAN
        self.train(training_data, discrete_columns=(self.input_dim_,))

        # One-hot-encode the class labels; Get the number of classes and the number of samples to generate per class.
        class_encoder = OneHotEncoder()
        y_encoded = class_encoder.fit_transform(y_train.reshape(-1, 1)).toarray()
        self.n_classes_ = y_encoded.shape[1]
        self.gen_samples_ratio_ = [int(sum(y_encoded[:, c])) for c in range(self.n_classes_)]

        majority_class = np.array(self.gen_samples_ratio_).argmax()
        num_majority_samples = np.max(np.array(self.gen_samples_ratio_))

        # X and Y data to return
        x_over_train = np.copy(x_train)
        y_over_train = np.copy(y_train)

        generated_data = [None for _ in range(self.n_classes_)]
        for cls in range(self.n_classes_):
            if cls != majority_class:
                samples_to_generate = num_majority_samples - self.gen_samples_ratio_[cls]

                # print("\tSampling Class y:", y, " Gen Samples ratio:", gen_samples_ratio[y])
                # generated_data[cls] = self.sample(samples_to_generate, cls).cpu().detach()
                generated_data[cls] = self.sample(n=samples_to_generate, condition_column=str(self.input_dim_),
                                                  condition_value=cls)[:, 0:self.input_dim_]

                min_classes = np.full(samples_to_generate, cls)

                x_over_train = np.vstack((x_over_train, generated_data[cls]))
                y_over_train = np.hstack((y_over_train, min_classes))

        # balanced_data = np.hstack((x_over_train, y_over_train.reshape((-1, 1))))
        # return balanced_data

        return x_over_train, y_over_train
