from imblearn.over_sampling import RandomOverSampler
from imblearn.over_sampling import SMOTE
from imblearn.over_sampling import BorderlineSMOTE
from imblearn.over_sampling import SVMSMOTE
from imblearn.over_sampling import KMeansSMOTE
from imblearn.over_sampling import ADASYN

from imblearn.under_sampling import RandomUnderSampler
from imblearn.under_sampling import ClusterCentroids

from sklearn.cluster import MiniBatchKMeans

from generators.c_gan import cGAN
from generators.sb_gan import sbGAN
from generators.ct_gan import ctGAN
from generators.ctd_gan import ctdGAN
from generators.cbr import CBR

from sdv.single_table import GaussianCopulaSynthesizer
from sdv.single_table import CTGANSynthesizer
from sdv.single_table import TVAESynthesizer

from dp_cgans import DP_CGAN


class BaseSampler:
    def __init__(self, name, short_name, model, **kwargs):
        """
        A data resampling technique for mitigating class imbalance. It may be an over-sampling or an under-sampling
        technique implementing fit(x, y) and fit_resample(x, y).

        Args:
            name: Name of the selected resampling technique.
            short_name: Short name of the selected resampling technique.
            model: an over-sampling or under-sampling technique implementing fit(x, y) and fit_resample(x, y).
            kwargs: extra arguments
        """

        self.name_ = name
        self.short_name_ = short_name
        self.sampler_ = model
        super().__init__(**kwargs)

    def fit(self, x, y):
        """
        Check inputs and statistics of the sampler.

        Args:
            x: features of the dataset.
            y: classes of the dataset.
        """
        if self.sampler_ is not None:
            self.sampler_.fit(x, y)

    def fit_resample(self, x_train, y_train, original_dataset=None, train_idx=None):
        """
        Resample the dataset.

        Args:
            x_train: features of the dataset.
            y_train: classes of the dataset.
            original_dataset: An object that represents the original input dataset - to be passed to the Synthetic
                Data Vault models.
            train_idx: The rows of the training examples in the original dataset.
        """
        if self.sampler_ is not None:
            # If the sampler has the fit_resample method (imblearn methods, DeepCoreML Models (CBR, GANs), etc.) we
            # just call it.
            sampler_fit_resample_method = getattr(self.sampler_, "fit_resample", None)
            if callable(sampler_fit_resample_method):
                return sampler_fit_resample_method(x_train, y_train)

            # otherwise, we apply the transformation to the dataset itself:
            else:
                if train_idx is None or original_dataset is None:
                    fit_data = x_train
                else:
                    fit_data = original_dataset.df_.iloc[train_idx, :]

                self.sampler_.fit(fit_data)
                return original_dataset.balance(self, train_idx)

        else:
            return x_train, y_train


class DataSamplers:
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
        clus = MiniBatchKMeans(n_clusters=2, init='k-means++', n_init='auto')

        # Good for KC1 disc = (12, 6) gen = (32, 32)
        disc = (256, 256)
        gen = (256, 256)
        emb_dim = 32
        knn = 10
        rad = 1
        pac = 1
        epochs = 300
        batch_size = 32
        max_clusters = 10
        act = 'tanh'

        dp_cols = {}
        for k in metadata.columns.keys():
            dp_cols[k] = {'type': metadata.columns[k]['sdtype']}

        # Experimental over-samplers.
        self.over_samplers_ = (
            # BaseSampler("TVAE", "TVAE",
            #            TVAESynthesizer(metadata, enforce_min_max_values=True, enforce_rounding=False, epochs=epochs)),

            # BaseSampler("Conditional GAN", "CGAN",
            #             cGAN(embedding_dim=emb_dim, discriminator=disc, generator=gen, pac=pac, adaptive=False,
            #                 g_activation=act, epochs=epochs, batch_size=batch_size, random_state=random_state)),

            #BaseSampler("Safe-Borderline GAN (KNN)", "SBGAN",
            #            sbGAN(embedding_dim=emb_dim, discriminator=disc, generator=gen, pac=pac, adaptive=False,
            #                  g_activation=act, epochs=epochs, batch_size=batch_size, method='knn', k=knn, r=rad,
            #                  random_state=random_state)),

            # BaseSampler("CTGAN", "CTGAN",
            #            CTGANSynthesizer(
            #                metadata, enforce_min_max_values=False, enforce_rounding=False, epochs=epochs,
            #                verbose=False)),

            #BaseSampler("ctdGANBase_mms11", "ctdGANBase_mms11",
            #            ctdGAN(embedding_dim=emb_dim, discriminator=disc, generator=gen, batch_size=batch_size,
            #                   scaler='mms11', epochs=epochs, pac=pac, max_clusters=max_clusters,
            #                   random_state=random_state)),

            #BaseSampler("ctdGANBase_mms11", "ctdGANBase_mms11",
            #            ctdGAN(embedding_dim=emb_dim, discriminator=disc, generator=gen, batch_size=batch_size,
            #                   scaler='stds', epochs=epochs, pac=pac, max_clusters=max_clusters,
            #                   random_state=random_state)),

            BaseSampler("DPCGAN", "DPCGAN",
                        DP_CGAN(field_types=dp_cols, generator_dim=(128, 128, 128), discriminator_dim=(128, 128, 128),
                                epochs=epochs, batch_size=batch_size, log_frequency=True, verbose=False, pac=1,
                                generator_lr=2e-4, discriminator_lr=2e-4, discriminator_steps=1, private=False,)),
        )

        # All over-samplers.
        self.over_samplers_all_ = (
            BaseSampler("None", "None", None),

            BaseSampler("Random Oversampling", "ROS",
                        RandomOverSampler(sampling_strategy=sampling_strategy, random_state=random_state)),

            BaseSampler("SMOTE", "SMOTE",
                        SMOTE(sampling_strategy=sampling_strategy, random_state=random_state)),

            BaseSampler("Borderline SMOTE", "B-SMOTE",
                        BorderlineSMOTE(sampling_strategy=sampling_strategy, random_state=random_state)),

            BaseSampler("SMOTE SVM", "SVM-SMOTE",
                        SVMSMOTE(sampling_strategy=sampling_strategy, random_state=random_state)),

            # BaseSampler("KMeans SMOTE", "KMN-SMOTE",
            #            KMeansSMOTE(sampling_strategy=sampling_strategy,
            #                        cluster_balance_threshold='auto', random_state=random_state)),

            BaseSampler("ADASYN", "ADASYN",
                        ADASYN(sampling_strategy=sampling_strategy, random_state=random_state)),

            BaseSampler("CBR", "CBR",
                        CBR(cluster_estimator='hac', cluster_resampler='cs', verbose=False, k_neighbors=1,
                            min_distance_factor=3, sampling_strategy=sampling_strategy, random_state=random_state)),

            BaseSampler("Conditional GAN", "CGAN",
                        cGAN(embedding_dim=emb_dim, discriminator=disc, generator=gen, pac=pac, adaptive=False,
                             g_activation=act, epochs=epochs, batch_size=batch_size, random_state=random_state)),

            BaseSampler("Safe-Borderline GAN (KNN)", "SBGAN",
                        sbGAN(embedding_dim=emb_dim, discriminator=disc, generator=gen, pac=pac, adaptive=False,
                              g_activation=act, epochs=epochs, batch_size=batch_size, method='knn', k=knn, r=rad,
                              random_state=random_state)),

            # This ctGAN is from the GitHub implementation
            BaseSampler("ctGAN", "ctGAN",
                        ctGAN(embedding_dim=emb_dim, discriminator=disc, generator=gen, pac=pac, adaptive=False,
                              g_activation=None, epochs=epochs, batch_size=batch_size, discriminator_steps=1,
                              log_frequency=True, verbose=False, random_state=random_state)),

            # And this ctGAN is from the Synthetic Data Vault library
            # Default Discriminator (256, 256) - Generator (256, 256)
            BaseSampler("CTGAN", "CTGAN",
                        CTGANSynthesizer(
                            metadata, enforce_min_max_values=False, enforce_rounding=False, epochs=epochs,
                            verbose=False)),

            BaseSampler("Gaussian Copula", "GCOP",
                        GaussianCopulaSynthesizer(metadata, enforce_min_max_values=False, enforce_rounding=False)),

            BaseSampler("ctdGANBase", "ctdGANBase",
                        ctdGAN(embedding_dim=emb_dim, discriminator=disc, generator=gen, batch_size=batch_size,
                               scaler='mms11', epochs=epochs, pac=pac, max_clusters=max_clusters,
                               random_state=random_state)),
        )

        self.under_samplers_ = (
            BaseSampler("None", "None", None),
            BaseSampler("Random Oversampling", "RUS",
                        RandomUnderSampler(sampling_strategy=sampling_strategy, replacement=True,
                                           random_state=random_state)),
            BaseSampler("Cluster Centroids", "CCUS",
                        ClusterCentroids(sampling_strategy=sampling_strategy, random_state=random_state))
        )

        self.num_over_samplers_ = len(self.over_samplers_)
        self.num_under_samplers_ = len(self.under_samplers_)

        super().__init__(**kwargs)
