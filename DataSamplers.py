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


class BaseSampler:
    def __init__(self, name, short_name, model, **kwargs):
        """
        A data resampling technique for mitigating class imbalance. It may be an over-sampling or an under-sampling
        technique implementing fit(x, y) and fit_resample(x, y).

        :param name: Name of the selected resampling technique.
        :param short_name: Short name of the selected resampling technique.
        :param model: an over-sampling or under-sampling technique implementing fit(x, y) and fit_resample(x, y).
        :param kwargs:
        """

        self.name_ = name
        self.short_name_ = short_name
        self.sampler_ = model
        super().__init__(**kwargs)

    def fit(self, x, y):
        """
        Check inputs and statistics of the sampler.

        :param x: features of the dataset
        :param y: classes of the dataset
        """
        if self.sampler_ is not None:
            self.sampler_.fit(x, y)

    def fit_resample(self, x, y):
        """
        Resample the dataset.

        :param x: features of the dataset
        :param y: classes of the dataset
        """
        if self.sampler_ is not None:
            return self.sampler_.fit_resample(x, y)
        else:
            return x


class DataSamplers:
    """
    Array of data over-sampling and inder-sampling techniques.
    """
    def __init__(self, sampling_strategy='auto', random_state=0, **kwargs):
        """
        :param sampling_strategy: float, str, dict or callable, default=auto
        Sampling information to resample the data set.
         * When float, it corresponds to the desired ratio of the number of samples in the minority class over the
           number of samples in the majority class after resampling. float is only available for binary classification.
           An error is raised for multi-class classification.
         * When str, specify the class targeted by the resampling. The number of samples in the different classes will
           be equalized. Possible choices are:
           * 'minority': resample only the minority class;
           * 'not minority': resample all classes but the minority class;
           * 'not majority': resample all classes but the majority class;
           * 'all': resample all classes;
           * 'auto': equivalent to 'not majority'.
         * When dict, the keys correspond to the targeted classes. The values correspond to the desired number of
           samples for each targeted class.
         * When callable, function taking y and returns a dict. The keys correspond to the targeted classes. The values
           correspond to the desired number of samples for each class.`
        :param random_state: Control the randomization of the algorithm.
        :param kwargs:
        """
        clus = MiniBatchKMeans(n_clusters=2, init='k-means++', n_init='auto')

        # Good for KC1 disc = (12, 6) gen = (32, 32)
        disc = (256, 256)
        gen = (256, 256)
        emb_dim = 32
        knn = 5
        rad = 200
        pac = 1
        epochs = 300
        batch_size = 32
        act = 'tanh'

        self.over_samplers_ = (
            BaseSampler("None", "None", None),
            BaseSampler("Random Oversampling", "ROS",
                        RandomOverSampler(sampling_strategy=sampling_strategy, random_state=random_state)),
            BaseSampler("SMOTE", "SMOTE",
                        SMOTE(sampling_strategy=sampling_strategy, random_state=random_state)),
            BaseSampler("Borderline SMOTE", "B-SMOTE",
                        BorderlineSMOTE(sampling_strategy=sampling_strategy, random_state=random_state)),
            BaseSampler("SMOTE SVM", "SVM-SMOTE",
                        SVMSMOTE(sampling_strategy=sampling_strategy, random_state=random_state)),
            BaseSampler("KMeans SMOTE", "KMN-SMOTE",
                        KMeansSMOTE(sampling_strategy=sampling_strategy, kmeans_estimator=clus,
                                    cluster_balance_threshold=0.05, random_state=random_state)),
            BaseSampler("ADASYN", "ADASYN",
                        ADASYN(sampling_strategy=sampling_strategy, random_state=random_state)),

            BaseSampler("Conditional pac GAN", "cGAN",
                        cGAN(embedding_dim=emb_dim, discriminator=disc, generator=gen, pac=pac, adaptive=False,
                             g_activation=act, epochs=epochs, batch_size=batch_size, random_state=random_state)),

            BaseSampler("Safe-Borderline GAN (KNN)", "SBGAN-KNN",
                        sbGAN(embedding_dim=emb_dim, discriminator=disc, generator=gen, pac=pac, adaptive=False,
                              g_activation=act, epochs=epochs, batch_size=batch_size, method='knn', k=knn, r=rad,
                              random_state=random_state)),

            BaseSampler("Safe-Borderline GAN (RAD)", "SBGAN-RAD",
                        sbGAN(embedding_dim=emb_dim, discriminator=disc, generator=gen, pac=pac, adaptive=False,
                              g_activation=act, epochs=epochs, batch_size=batch_size, method='rad', k=knn, r=rad,
                              random_state=random_state)),

            BaseSampler("ctGAN", "ctGAN",
                        ctGAN(embedding_dim=emb_dim, discriminator=disc, generator=gen, pac=pac, adaptive=False,
                              g_activation=None, epochs=epochs, batch_size=batch_size, discriminator_steps=1,
                              log_frequency=True, verbose=False, random_state=random_state)),

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
