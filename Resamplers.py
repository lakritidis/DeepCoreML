import numpy as np
import pandas as pd

from imblearn.over_sampling import RandomOverSampler
from imblearn.over_sampling import SMOTE
from imblearn.over_sampling import BorderlineSMOTE
from imblearn.over_sampling import SVMSMOTE
from imblearn.over_sampling import KMeansSMOTE
from imblearn.over_sampling import ADASYN

from generators.c_gan import cGAN
from generators.sb_gan import sbGAN
from generators.ct_gan import ctGAN
from generators.ctd_gan import ctdGAN
from generators.cbr import CBR

from sdv.single_table import GaussianCopulaSynthesizer
from sdv.single_table import CTGANSynthesizer
from sdv.single_table import TVAESynthesizer


class BaseResampler:
    """Resampler base wrapper class.

    Used to wrap the ImbLearn models, C-GAN and SB-GAN"""
    def __init__(self, name, model, random_state):
        self.name_ = name
        self._model = model
        self._random_state = random_state

    def fit(self, x, y):
        self._model.fit(x, y)
        return self

    def fit_resample(self, dataset, training_set_rows):
        x_train = dataset.x_[training_set_rows]
        y_train = dataset.y_[training_set_rows]

        x_bal, y_bal = self._model.fit_resample(x_train, y_train)

        return x_bal, y_bal


class CTResampler(BaseResampler):
    """Resampler wrapper class - inherits from BaseResampler.

    Used to wrap ctGAN (GitHub version) and ctdGAN"""
    def __init__(self, name, model, random_state):
        super().__init__(name, model, random_state)

    def fit_resample(self, dataset, training_set_rows):
        x_train = dataset.x_[training_set_rows]
        y_train = dataset.y_[training_set_rows]

        x_bal, y_bal = self._model.fit_resample(x_train, y_train)

        return x_bal, y_bal


class SDVResampler(BaseResampler):
    """Resampler wrapper class - inherits from BaseResampler.

    Used to wrap the Synthetic Data Vault (SDV) models"""
    def __init__(self, name, model, random_state):
        super().__init__(name, model, random_state)

    def fit_resample(self, dataset, training_set_rows):
        x_train = dataset.x_[training_set_rows]
        y_train = dataset.y_[training_set_rows]

        # if training_set_rows is None or dataset is None:
        #    fit_data = x_train
        # else:
        #    fit_data = dataset.df_.iloc[training_set_rows, :]

        # Fit the SDV model
        fit_data = dataset.df_.iloc[training_set_rows, :]
        self._model.fit(fit_data)

        # Perform Sampling until the dataset is balanced
        gen_samples_ratio = np.unique(y_train, return_counts=True)[1]

        majority_class = np.array(gen_samples_ratio).argmax()
        num_majority_samples = np.max(np.array(gen_samples_ratio))

        x_balanced = np.copy(x_train)
        y_balanced = np.copy(y_train)

        # Perform oversampling
        for cls in range(dataset.num_classes):
            if cls != majority_class:
                samples_to_generate = num_majority_samples - gen_samples_ratio[cls]

                generated_samples = None

                # Generate the appropriate number of samples to equalize cls with the majority class.
                # print("\tSampling Class y:", cls, " Gen Samples ratio:", gen_samples_ratio[cls])
                reference_data = pd.DataFrame(data={str(dataset.class_column): [cls] * samples_to_generate})
                generated_samples = self._model.sample_remaining_columns(
                    max_tries_per_batch=500, known_columns=reference_data).iloc[:, 0:dataset.dimensionality]

                if generated_samples is not None:
                    # print(generated_samples.shape)
                    generated_classes = np.full(samples_to_generate, cls)

                    x_balanced = np.vstack((x_balanced, generated_samples))
                    y_balanced = np.hstack((y_balanced, generated_classes))

        return x_balanced, y_balanced


class TestSynthesizers:
    """
    An object that contains a collection of data over-sampling and under-sampling techniques.
    """
    def __init__(self, metadata, sampling_strategy='auto', random_state=0, **kwargs):
        """
        An object that contains a collection of data over-sampling and under-sampling techniques.

        Args:
            random_state: Control the randomization of the algorithm.
            sampling_strategy: how the member samplers generate/remove/replace samples.

             - If a float is passed, it corresponds to the desired ratio of the number of samples in the minority class over the number of samples in the majority class after resampling. float is only available for binary classification. An error is raised for multi-class classification.
             - If a string is passed, specify the class targeted by the resampling. The number of samples in the different classes will be equalized. Possible choices are:

               * 'minority': resample only the minority class;
               * 'not minority': resample all classes but the minority class;
               * 'not majority': resample all classes but the majority class;
               * 'all': resample all classes;
               * 'auto': equivalent to 'not majority'.

             - If a dictionary is passed, the keys correspond to the targeted classes. The values correspond to the desired number of samples for each targeted class.
             - When callable, function taking y and returns a dict. The keys correspond to the targeted classes. The values correspond to the desired number of samples for each class.`

            kwargs: extra arguments
        """
        disc = (256, 256)
        gen = (256, 256)
        emb_dim = 32
        knn = 10
        rad = 1
        pac = 1
        epochs = 300
        batch_size = 32
        max_clusters = 10

        # Prepare the column descriptors for the SDV models
        dp_cols = {}
        for k in metadata.columns.keys():
            dp_cols[k] = {'type': metadata.columns[k]['sdtype']}

        # Random over-sampler
        ros = RandomOverSampler(sampling_strategy=sampling_strategy, random_state=random_state)

        # Synthetic Minority Oversampling Technique (SMOTE)
        smote = SMOTE(sampling_strategy=sampling_strategy, random_state=random_state)

        # Borderline SMOTE
        b_smote = BorderlineSMOTE(sampling_strategy=sampling_strategy, random_state=random_state)

        # SMOTE with Support Vector Machine
        svm_smote = SVMSMOTE(sampling_strategy=sampling_strategy, random_state=random_state)

        # SMOTE-based clustered over-sampler
        km_smote = KMeansSMOTE(sampling_strategy=sampling_strategy, cluster_balance_threshold='auto',
                               random_state=random_state)

        # Adaptive Synthetic Sampling (ADASYN)
        adasyn = ADASYN(sampling_strategy=sampling_strategy, random_state=random_state)

        # Cluster-Based Resampler (CBR)
        cbr = CBR(sampling_strategy=sampling_strategy, cluster_estimator='hac', cluster_resampler='cs', verbose=False,
                  k_neighbors=1, min_distance_factor=0.01, random_state=random_state)

        # Conditional Generative Adversarial Network (C-GAN)
        c_gan = cGAN(embedding_dim=emb_dim, discriminator=disc, generator=gen, pac=pac, epochs=epochs,
                     batch_size=batch_size, random_state=random_state)

        # Safe/Borderline Generative Adversarial Network (SB-GAN)
        sb_gan = sbGAN(embedding_dim=emb_dim, discriminator=disc, generator=gen, pac=pac, epochs=epochs,
                       batch_size=batch_size, method='knn', k=knn, r=rad, random_state=random_state)

        # This ctGAN is from the GitHub implementation
        ctgan_1 = ctGAN(embedding_dim=emb_dim, discriminator=disc, generator=gen, pac=pac, epochs=epochs, verbose=False,
                        batch_size=batch_size, discriminator_steps=1, log_frequency=True, random_state=random_state)

        # And this ctGAN is from the Synthetic Data Vault - Default Discriminator (256, 256) - Generator (256, 256)
        ctgan = CTGANSynthesizer(metadata, enforce_min_max_values=False, enforce_rounding=False, epochs=epochs,
                                 verbose=False)

        # Tabular Variational Autoencoder (TVAE)
        t_vae = TVAESynthesizer(metadata, enforce_min_max_values=False, enforce_rounding=False, epochs=1000)

        # Gaussian Copula
        g_cop = GaussianCopulaSynthesizer(metadata, enforce_min_max_values=False, enforce_rounding=False)

        # CTD Generative Adversarial Network (ctdGAN)
        ctd_gan = ctdGAN(embedding_dim=emb_dim, discriminator=disc, generator=gen, epochs=epochs, batch_size=batch_size,
                         pac=pac, scaler='mms11', max_clusters=max_clusters, random_state=random_state)

        # All over-samplers.
        self.over_samplers_all_ = (
            BaseResampler(name="None", model=None, random_state=random_state),
            BaseResampler(name="ROS", model=ros, random_state=random_state),
            BaseResampler(name="SMOTE", model=smote, random_state=random_state),
            BaseResampler(name="BorderSMOTE", model=b_smote, random_state=random_state),
            BaseResampler(name="SVM-SMOTE", model=svm_smote, random_state=random_state),
            BaseResampler(name="KMeans SMOTE", model=km_smote, random_state=random_state),
            BaseResampler(name="ADASYN", model=adasyn, random_state=random_state),
            BaseResampler(name="CBR", model=cbr, random_state=random_state),
            BaseResampler(name="C-GAN", model=c_gan, random_state=random_state),
            BaseResampler(name="SB-GAN", model=sb_gan, random_state=random_state),
            SDVResampler(name="CTGAN", model=ctgan, random_state=random_state),
            SDVResampler(name="TVAE", model=t_vae, random_state=random_state),
            SDVResampler(name="GCOP", model=g_cop, random_state=random_state),
            CTResampler("ctGAN", model=ctgan_1, random_state=random_state),
            CTResampler("ctdGAN", model=ctd_gan,random_state=random_state),
        )

        self.over_samplers_ = (
            SDVResampler(name="CTGAN", model=ctgan, random_state=random_state),
            SDVResampler(name="TVAE", model=t_vae, random_state=random_state),
            SDVResampler(name="GCOP", model=g_cop, random_state=random_state),
        )

        self.num_over_samplers_ = len(self.over_samplers_)