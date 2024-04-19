import pathlib
import gzip
import json

import numpy as np
import pandas as pd

import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.preprocessing import LabelEncoder
from sklearn.datasets import make_classification
from sklearn.model_selection import cross_validate, StratifiedKFold
from sklearn.metrics import make_scorer
from sklearn.metrics import f1_score, accuracy_score, balanced_accuracy_score, precision_score, recall_score
from imblearn.metrics import sensitivity_score, specificity_score

import warnings
warnings.filterwarnings("ignore")

pd.set_option('display.max_rows', 100)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', 400)


# #################################################################################################################
# DATASET BASE CLASS
class BaseDataset:
    def __init__(self, name, random_state=0):
        """
        Dataset initializer: BaseDataset is the base class of all dataset subclasses.

        :param random_state: controls random number generation. Set this to a fixed integer to get reproducible results.
        """
        self._name = name
        self._random_state = random_state
        self._dimensionality = 0
        self._num_classes = 0
        self._num_samples = 0

        self._feature_columns = 0
        self._class_column = 0
        self._class_column_name = ''

        self.x_ = None
        self.y_ = None
        self.df_ = None

    # Create synthetic imbalanced datasets with numeric features
    def create_synthetic(self, n_samples=1000, n_classes=2, imb_ratio=(0.5, 0.5)):
        """
        Create a synthetic imbalanced dataset with numeric features.
        """ 
        synthetic_dataset = []
        if n_classes == 2:
            synthetic_dataset = make_classification(
                n_samples=n_samples, n_features=2, n_clusters_per_class=1, flip_y=0, n_classes=2, weights=imb_ratio,
                class_sep=0.5, n_informative=2, n_redundant=0, n_repeated=0, random_state=self._random_state)

        elif n_classes == 4:
            synthetic_dataset = make_classification(
                n_samples=n_samples, n_features=2, n_clusters_per_class=1, flip_y=0, n_classes=4, weights=imb_ratio,
                class_sep=1.0, n_informative=2, n_redundant=0, n_repeated=0, random_state=self._random_state)

        self._feature_columns = 0
        self._class_column = 1

        self.x_ = synthetic_dataset[0]
        self.y_ = synthetic_dataset[1]

        self._num_classes = n_classes
        self._num_samples = n_samples
        self._dimensionality = self.x_.shape[1]

        print("Num Samples:", self._num_samples, "\nClass Distribution:")
        for k in range(self._num_classes):
            print("\tClass", k, ":", len(self.y_[self.y_ == k]), "samples")

    # Load a dataset from an external CSV file
    def load_from_csv(self, path='', feature_cols=range(0, 1), class_col=1):
        """
        Load a dataset from a CSV file
        Args:
            path: The location of the input CSV file.
            feature_cols: A tuple the denotes the indices of the columns with the input variables.
            class_col: The integer index of the column that stores the target variables.
        """
        self._feature_columns = feature_cols
        self._class_column = class_col

        file_extension = pathlib.Path(path).suffix
        if file_extension == '.csv':
            self.df_ = pd.read_csv(path, encoding='utf-8')
            # self.df_ = pd.read_csv(path, encoding='latin-1', header=None)
        else:
            self.df_ = self.get_df(path)

        self._class_column_name = self.df_.columns[len(self.df_.columns) - 1]

        # Shuffle the dataframe
        self.df_.sample(frac=1)

        # Convert x and y to NumPy arrays
        self.x_ = self.df_.iloc[:, feature_cols].to_numpy()
        self.y_ = self.df_.iloc[:, class_col].to_numpy()

        # Label encode the target variables
        class_encoder = LabelEncoder()
        self.y_ = class_encoder.fit_transform(self.y_)

        # Label encode the class column of the Dataframe
        self.df_[self._class_column_name] = self.y_

        # Useful quick reference statistics
        self._num_classes = len(self.df_.iloc[:, class_col].unique())
        self._num_samples = self.x_.shape[0]
        self._dimensionality = self.x_.shape[1]

    def get_dimensionality(self):
        return self._dimensionality

    def get_num_samples(self):
        return self._num_samples

    def get_num_classes(self):
        return self._num_classes

    def get_class_column(self):
        return self._class_column

    # Display the basic dataset parameters
    def display_params(self):
        """
        Display the basic dataset parameters
        :return:
        """
        print("Num Samples:", self._num_samples, "- Dimensions:", self._dimensionality)
        print("Classes:", self._num_classes, "- Class Distribution:")
        for k in range(self._num_classes):
            print("\tClass", k, ":", len(self.y_[self.y_ == k]), "samples")

    def plot(self, dim1=0, dim2=1):
        """
        Uses a 2-dimensional space to plot two features of the dataset and color the data points according to their
        class.
        """
        sns.scatterplot(x=self.x_[:, dim1], y=self.x_[:, dim2], hue=self.y_, legend=False)
        plt.show()

    # Bring balance to the dataset by applying a pipeline with an over-/under-sampling technique. Then, train and test
    # a classifier with cross-validation. A pipeline object is required.
    def cross_val(self, estimator, num_folds, num_threads, classifier_str, sampler_str, order):
        """
        Applies a classification pipeline to the dataset, and then, it evaluates the classification performance by using
        cross validation.

        :param estimator: A multi-stage pipeline object. The pipeline must include a classifier at its final stage.
        :param num_folds: Number of folds of cross validation.
        :param num_threads: The number of parallel threads to deploy for executing cross validation.
        :param classifier_str: A custom string to describe the classifier of the pipeline.
        :param sampler_str: A custom string to describe the other stages of the pipeline.
        :param order: An assistant variable for `results_list`
        :return: object that stores all cross validation results
        """
        scorers = {
            'accuracy': make_scorer(accuracy_score),
            'balanced_accuracy': make_scorer(balanced_accuracy_score),
            'sensitivity': make_scorer(sensitivity_score, average='weighted'),
            'specificity': make_scorer(specificity_score, average='weighted'),
            'f1': make_scorer(f1_score, average='weighted'),
            'precision': make_scorer(precision_score, average='weighted'),
            'recall': make_scorer(recall_score, average='weighted'),
        }

        # cross_validate uses Stratified kFold when cv is int
        cv_results = cross_validate(estimator, self.x_, self.y_, cv=num_folds, scoring=scorers,
                                    return_train_score=False, return_estimator=False, n_jobs=num_threads,
                                    error_score='raise')

        results_list = []
        for key in cv_results.keys():
            for f in range(num_folds):
                lst = [self._name, f+1,  sampler_str, classifier_str, key, cv_results[key][f]]
                results_list.append(lst)

        cv_results['dataset'] = self._name
        cv_results['sampler'] = sampler_str
        cv_results['classifier'] = classifier_str

        mean_results_list = [
            classifier_str,
            sampler_str,
            self._name,
            cv_results['test_accuracy'].mean(),
            cv_results['test_accuracy'].std(),
            cv_results['test_balanced_accuracy'].mean(),
            cv_results['test_balanced_accuracy'].std(),
            cv_results['test_sensitivity'].mean(),
            cv_results['test_sensitivity'].std(),
            cv_results['test_specificity'].mean(),
            cv_results['test_specificity'].std(),
            cv_results['test_f1'].mean(),
            cv_results['test_f1'].std(),
            cv_results['test_precision'].mean(),
            cv_results['test_precision'].std(),
            cv_results['test_recall'].mean(),
            cv_results['test_recall'].std(),
            cv_results['fit_time'].mean(),
            order
        ]

        return results_list, mean_results_list

    def balance(self, sampler, train):
        x_in = self.x_[train]
        y_in = self.y_[train]

        gen_samples_ratio = np.unique(y_in, return_counts=True)[1]

        majority_class = np.array(gen_samples_ratio).argmax()
        num_majority_samples = np.max(np.array(gen_samples_ratio))

        x_balanced = np.copy(x_in)
        y_balanced = np.copy(y_in)

        # Perform oversampling
        for cls in range(self._num_classes):
            if cls != majority_class:
                samples_to_generate = num_majority_samples - gen_samples_ratio[cls]

                generated_samples = None

                # Generate the appropriate number of samples to equalize cls with the majority class.
                # print("\tSampling Class y:", cls, " Gen Samples ratio:", gen_samples_ratio[cls])
                if sampler.short_name_ == "GCOP":
                    reference_data = pd.DataFrame(data={self._class_column_name: [cls] * samples_to_generate})
                    generated_samples = sampler.sampler_.sample_remaining_columns(
                        max_tries_per_batch=500, known_columns=reference_data).iloc[:, 0:self._dimensionality]

                elif sampler.short_name_ == "CTGAN":
                    reference_data = pd.DataFrame(data={self._class_column_name: [cls] * samples_to_generate})
                    generated_samples = sampler.sampler_.sample_remaining_columns(
                        max_tries_per_batch=500, known_columns=reference_data).iloc[:, 0:self._dimensionality]

                elif sampler.short_name_ == "TVAE":
                    reference_data = pd.DataFrame(data={self._class_column_name: [cls] * samples_to_generate})
                    generated_samples = sampler.sampler_.sample_remaining_columns(
                        max_tries_per_batch=500, known_columns=reference_data).iloc[:, 0:self._dimensionality]

                elif sampler.short_name_ == "GAAN":
                    generated_samples = sampler.sampler_.sample(samples_to_generate, cls)

                if generated_samples is not None:
                    # print(generated_samples.shape)
                    generated_classes = np.full(samples_to_generate, cls)

                    x_balanced = np.vstack((x_balanced, generated_samples))
                    y_balanced = np.hstack((y_balanced, generated_classes))

        return x_balanced, y_balanced

    # Return the dataframe
    def get_data(self):
        """
        Return the Dataframe of the dataset
        :return: A Pandas Dataframe with the stored data
        """
        return self.df_

    def get_df(self, path):
        i = 0
        df = {}
        for d in self.parse(path):
            df[i] = d
            i += 1
        return pd.DataFrame.from_dict(df, orient='index')

    def parse(self, path):
        g = gzip.open(path, 'rb')
        for lt in g:
            yield json.loads(lt)


class FakeDataset(BaseDataset):
    def __init__(self, source_dataset, name, random_state=0):
        super().__init__(name, random_state)

        self._source_dataset = BaseDataset(source_dataset['name'], random_state=random_state)
        self._source_dataset.load_from_csv(path=source_dataset['path'], feature_cols=source_dataset['features_cols'],
                                           class_col=source_dataset['class_col'])
        self._feature_columns = source_dataset['features_cols']
        self._class_column = source_dataset['class_col']

    def synthesize(self, generator):
        """
        This function synthesizes a new dataset with the same class distribution as the source dataset.
        """
        generator.fit(self._source_dataset.x_, self._source_dataset.y_)
        self.x_, self.y_ = generator.synthesize_dataset()

    def synthesize_balance(self, generator):
        """
        This function uses the generator to balance the source dataset.
        """
        self.x_, self.y_ = generator.fit_resample(self._source_dataset.x_, self._source_dataset.y_)

    def synthesize_merge(self, generator):
        """
        This function synthesizes a new dataset with the same class distribution as the source dataset. Then, it
        merges the source dataset with the synthetic one.
        The classes are not preserved - the source (real) samples get a flag=0 and the synthetic ones get a flag=1.
        """
        original_x = self._source_dataset.x_
        original_y = self._source_dataset.y_

        generator.fit(original_x, original_y)
        synthetic_x, synthetic_y = generator.synthesize_dataset()

        original_flags = np.zeros((self._source_dataset.get_num_samples(),))
        synthetic_flags = np.ones((self._source_dataset.get_num_samples(),))

        self.x_ = np.vstack((original_x, synthetic_x))
        self.y_ = np.hstack((original_flags, synthetic_flags))

    def return_description(self):
        d = {
            'name': self._name,
            'path': 'No File',
            'features_cols': self._feature_columns,
            'class_col': self._class_column
        }

        return d
