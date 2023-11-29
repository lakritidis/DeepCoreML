import numpy as np
import pandas as pd

from sklearn.decomposition import PCA, IncrementalPCA, KernelPCA, SparsePCA, TruncatedSVD, NMF
from sklearn.feature_selection import mutual_info_classif

import torch
import torch.nn as nn


class BaseReducer:
    """
    Base class for dimensionality reduction
    """
    def __init__(self, latent_dimensionality, random_state, **kwargs):
        self.seed_ = random_state
        self.latent_dim_ = latent_dimensionality
        self.reducer_ = None
        super().__init__(**kwargs)

    def display(self):
        print(self.reducer_)

    def get_reducer(self):
        return self.reducer_

    def get_property(self, property_name):
        obj_vars = vars(self.reducer_)
        return obj_vars[property_name]

    def get_latent_dimensionality(self):
        return self.latent_dim_


class SkLearnReducer(BaseReducer):
    """
    Base class for dimensionality reduction
    """
    def __init__(self, reducer, latent_dimensionality, random_state=0, **kwargs):
        super().__init__(latent_dimensionality, random_state)
        self.reducer_ = reducer

    def fit(self, x, y=None):
        self.reducer_.fit(x, y)

    def transform(self, x):
        x_prime = self.reducer_.transform(x)
        return x_prime

    def fit_transform(self, x, y):
        self.reducer_.fit(x, y)
        x_prime = self.reducer_.transform(x)
        return x_prime


# PCA Reducer
# https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html
class PCAReducer(SkLearnReducer):
    def __init__(self, n_components=16, random_state=0, **kwargs):
        super().__init__(
            PCA(n_components=n_components, random_state=random_state, **kwargs),
            n_components, random_state, **kwargs)


# Incremental PCA Reducer
# https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.IncrementalPCA.html
class IncrementalPCAReducer(SkLearnReducer):
    def __init__(self, n_components=16, random_state=0, **kwargs):
        super().__init__(
            IncrementalPCA(n_components=n_components, **kwargs),
            n_components, random_state, **kwargs)


# Sparse PCA Reducer
# https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.SparsePCA.html
class SparsePCAReducer(SkLearnReducer):
    def __init__(self, n_components=16, random_state=0, **kwargs):

        super().__init__(
            SparsePCA(n_components=n_components, random_state=random_state, **kwargs),
            n_components, random_state, **kwargs)


# Kernel PCA Reducer
# https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.KernelPCA.html
class KernelPCAReducer(SkLearnReducer):
    def __init__(self, n_components=16, random_state=0, **kwargs):

        super().__init__(
            KernelPCA(n_components=n_components, random_state=random_state, **kwargs),
            n_components, random_state, **kwargs)


# Truncated Singular Value Decomposition Reducer:
# https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.TruncatedSVD.html
class TSVDReducer(SkLearnReducer):
    def __init__(self, n_components=16, random_state=0, **kwargs):
        super().__init__(
            TruncatedSVD(n_components=n_components, random_state=random_state, **kwargs),
            n_components, random_state, **kwargs)


# NonNegative Matrix Factorization Reducer:
# https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.NMF.html
class NMFReducer(SkLearnReducer):
    def __init__(self, n_components=16, random_state=0, **kwargs):
        super().__init__(
            NMF(n_components=n_components, random_state=random_state, **kwargs),
            n_components, random_state, **kwargs)


# Dimensionality reduction with Autoencoders
class AutoencoderReducer(nn.Module):
    def __init__(self, encoder=(256, 128), decoder=(128, 256), input_dimensionality=512, n_components=128,
                 training_epochs=20, batch_size=32, random_state=0):

        super().__init__()

        self.latent_dimensionality_ = n_components
        self.input_dimensionality_ = input_dimensionality
        self.seed_ = random_state
        self.device_ = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.epochs_ = training_epochs
        self.batch_size_ = batch_size

        seq_encoder = []
        dim = self.input_dimensionality_
        for lay in encoder:
            seq_encoder += [nn.Linear(dim, lay), nn.Dropout(0.2), nn.LeakyReLU(0.2)]
            dim = lay

        seq_encoder += [nn.Linear(dim, self.latent_dimensionality_), nn.Sigmoid()]
        self.encoder_ = nn.Sequential(*seq_encoder).to(self.device_)

        seq_decoder = []
        dim = self.latent_dimensionality_
        for lay in decoder:
            seq_decoder += [nn.Linear(dim, lay), nn.Dropout(0.2), nn.LeakyReLU(0.2)]
            dim = lay

        seq_decoder += [nn.Linear(dim, self.input_dimensionality_)]
        self.decoder_ = nn.Sequential(*seq_decoder).to(self.device_)

    def forward(self, x):
        out = self.decoder_(self.encoder_(x))
        return out

    def display(self):
        print("Encoder")
        print(self.encoder_)
        print("Decoder")
        print(self.decoder_)

    def fit(self, x, y=None):
        if isinstance(x, np.ndarray):
            training_data = torch.from_numpy(x).to(torch.float32).to(self.device_)
        elif isinstance(x, pd.DataFrame):
            training_data = torch.tensor(x.values).to(torch.float32).to(self.device_)
        else:
            print(x.toarray())
            training_data = torch.from_numpy(x.toarray()).to(torch.float32).to(self.device_)

        loader = torch.utils.data.DataLoader(dataset=training_data, batch_size=self.batch_size_, shuffle=True)
        loss_function = torch.nn.MSELoss()

        optimizer = torch.optim.Adam(self.parameters(), lr=1e-1, weight_decay=1e-8)

        loss = 0
        for epoch in range(self.epochs_):
            for data in loader:
                # Output of Autoencoder
                reconstructed = self(data)

                # Calculating the loss function
                loss = loss_function(reconstructed, data)

                # The gradients are set to zero, and then, the gradient is computed and stored.
                # .step() performs parameter update
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            print("Epoch ", epoch, " loss =", loss)

    def transform(self, x):
        if isinstance(x, np.ndarray):
            test_data = torch.from_numpy(x).to(torch.float32).to(self.device_)
        elif isinstance(x, pd.DataFrame):
            test_data = torch.tensor(x.values).to(torch.float32).to(self.device_)
        else:
            test_data = torch.from_numpy(x.toarray()).to(torch.float32).to(self.device_)

        reduced_data = self.encoder_(test_data)

        return reduced_data.cpu().detach().numpy()

    def fit_transform(self, x, y=None):
        self.fit(x, y)
        return self.transform(x)


# Dimensionality reduction with Feature selection
class FeatureSelector(BaseReducer):
    def __init__(self, n_components=128, random_state=0, **kwargs):
        super().__init__(n_components, random_state)
        self.best_features_ = []

    def fit(self, features, labels):

        feature_scores = {}
        num_features = features.shape[1]
        for f in range(num_features):
            feat = features[:, f].reshape(-1, 1)
            score = mutual_info_classif(feat, labels, discrete_features=[False], random_state=self.seed_)
            feature_scores[f] = score[0]

        value_key_pairs = ((value, key) for (key, value) in feature_scores.items())
        sorted_feature_scores = sorted(value_key_pairs, reverse=True)

        self.best_features_ = [value for (key, value) in sorted_feature_scores]
        self.best_features_ = self.best_features_[: self.latent_dim_]

    def transform(self, data):
        num_features = data.shape[1]
        delete_columns = [feat for feat in range(num_features) if feat not in self.best_features_]

        return np.delete(data, delete_columns, axis=1)

    def fit_transform(self, features, labels):
        self.fit(features, labels)
        return self.transform(features)
