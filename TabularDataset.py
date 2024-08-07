import pathlib
import gzip
import json

import numpy as np
import pandas as pd

import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.preprocessing import OrdinalEncoder, LabelEncoder
from sklearn.datasets import make_classification
from sklearn.model_selection import cross_validate
from sklearn.metrics import make_scorer
from sklearn.metrics import f1_score, accuracy_score, balanced_accuracy_score, precision_score, recall_score
from imblearn.metrics import sensitivity_score, specificity_score

from Dataset import Dataset

import warnings

warnings.filterwarnings("ignore")

pd.set_option('display.max_rows', 1000)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', 400)


# #################################################################################################################
# DATASET BASE CLASS
class TabularDataset(Dataset):
    def __init__(self, name, categorical_columns=None, class_column=None, random_state=0):
        """
        Tabular Dataset initializer

        Args:
            name (string): the name of the tabular dataset
            categorical_columns (List or Tuple): A list or tuple with the indices of the categorical columns (if any).
            class_column (int): The index of the column that stores the class of the sample (if any).
            random_state: Controls random number generation. Set this to a fixed integer to get reproducible results.
        """
        super().__init__(name, class_column, random_state)

        # Raw dataset dimensions - (rows, columns)
        self.num_rows = 0
        self.num_columns = 0

        # Dimensionality is NOT equal to the number of columns.
        # e.g. a single categorical column may have multiple dimensions if one-hot-encoded
        self.dimensionality = 0

        # The raw Dataframe
        self._raw_df = None

        # The Dataframe after some processing (will be used for training)
        self.df_ = None

        # Create one LabelEncoder object for each categorical column
        if categorical_columns is None:
            self.categorical_columns = None
        else:
            self.categorical_columns = list(categorical_columns)
            self._label_encoders = [OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
                                    for _ in range(len(self.categorical_columns))]

        self._class_encoder = LabelEncoder()
        self.x_ = None
        self.y_ = None

    # Create synthetic imbalanced datasets with numeric features
    def create_synthetic(self, num_samples=1000, num_classes=2, imb_ratio=(0.5, 0.5)):
        """
        Create a synthetic imbalanced dataset with numeric features.

        Args:
            num_samples: how many data instances to create.
            num_classes: the number of classes of the synthetic data instances.
            imb_ratio: the imbalance ratio of the data (it must be a tuple comprising n_classes elements with sum=1).
        """
        synthetic_dataset = []
        if num_classes == 2:
            synthetic_dataset = make_classification(
                n_samples=num_samples, n_features=2, n_clusters_per_class=1, flip_y=0, n_classes=2, weights=imb_ratio,
                class_sep=0.5, n_informative=2, n_redundant=0, n_repeated=0, random_state=self._random_state)

        elif num_classes == 4:
            synthetic_dataset = make_classification(
                n_samples=num_samples, n_features=2, n_clusters_per_class=1, flip_y=0, n_classes=4, weights=imb_ratio,
                class_sep=1.0, n_informative=2, n_redundant=0, n_repeated=0, random_state=self._random_state)

        self.class_column = 1
        self.num_rows = num_samples
        self.num_classes = num_classes

        x = synthetic_dataset[0]
        y = synthetic_dataset[1]

        self.num_rows = num_samples
        self.num_columns = self.dimensionality = x.shape[1]

        print("Num Samples:", self.num_rows, "\nClass Distribution:")
        for k in range(self.num_classes):
            print("\tClass", k, ":", len(y[y == k]), "samples")

    # Load a dataset from an external CSV file
    def load_from_csv(self, path=''):
        """
        Load a Dataset from a CSV file and compute several basic statistics (number of samples, number of classes,
        input dimensionality). The target variables are Label Encoded.

        Args:
            path: The location of the input CSV file.
        """
        file_extension = pathlib.Path(path).suffix
        if file_extension == '.csv':
            self._raw_df = pd.read_csv(path, encoding='utf-8', keep_default_na=True, na_values=['<null>'])
        else:
            self._raw_df = self.get_df(path)

        # Preprocessing steps:
        # Step 1: Remove rows with missing values.
        self.df_ = self._raw_df.dropna(inplace=False)
        self.df_.reset_index(drop=True, inplace=True)

        # Step 2: Shuffle the dataframe
        self.df_.sample(frac=1)

        self.num_rows = self.df_.shape[0]
        self.num_columns = self.df_.shape[1]

        # Step 3: If a class column exists, put it in the end
        if self.class_column is not None:
            if self.class_column < self.num_columns - 1:
                df_class = self.df_.iloc[:, self.class_column]
                self.df_.drop(self.df_.columns[self.class_column], axis=1, inplace=True)

                self.df_ = pd.concat([self.df_, df_class], axis=1)

                # Adjust the indices of the categorical columns
                if self.categorical_columns is not None:
                    for col in range(len(self.categorical_columns)):
                        if self.categorical_columns[col] > self.class_column:
                            self.categorical_columns[col] -= 1

                self.class_column = self.num_columns - 1

        # Step 4: Change the name of the Dataframe columns
        self.df_.columns = [str(c) for c in range(self.num_columns)]

        # Step 5: Fit_transform the label encoders
        if self.categorical_columns is not None:
            for c in range(len(self.categorical_columns)):
                col = str(self.categorical_columns[c])
                self.df_[col] = self._label_encoders[c].fit_transform(self.df_[col])

        # Step 6: Label Encode the class labels
        self.df_[str(self.class_column)] = self._class_encoder.fit_transform(self.df_[str(self.class_column)])

        self.x_ = self.df_.iloc[:, 0:self.class_column].to_numpy()
        self.y_ = self.df_.iloc[:, self.class_column].to_numpy()

        # Useful quick reference statistics
        self.num_classes = len(self.df_.iloc[:, self.class_column].unique())
        self.dimensionality = self.x_.shape[1]

    # Display the basic dataset parameters
    def display_params(self):
        """
        Display the basic dataset parameters.
        """
        print("Num Rows:", self.num_rows, ", Num Columns:", self.num_columns)
        print("\nEncoded data Dimensions:", self.dimensionality)
        print("Classes:", self.num_classes, "- Class Distribution:")
        for k in range(self.num_classes):
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

        Args:
            estimator: A multi-stage Pipeline object. The Pipeline must include a classifier at its final stage.
            num_folds: Number of folds for cross validation.
            num_threads: The number of parallel threads for cross validation.
            classifier_str: A custom string that describes the Classifier in the Pipeline.
            sampler_str: A custom string that describes the other stages of the Pipeline.
            order: An assistant variable for enumerating the returned results.

        Returns:
            Two objects that store all cross validation results.
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
                lst = [self._name, f + 1, sampler_str, classifier_str, key, cv_results[key][f]]
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
        """
        Generate artificial data samples to mitigate class imbalance in a dataset. This function is used by the
        re-sampling methods that do not support the `fit_resample` method (like the models of the Synthetic Data
        Vault - SDV).

        Args:
            sampler: A BaseSampler object.
            train: The training set - for the SDV models, train is a concatenation of the feature vectors with classes.

        Returns:
            The balanced dataset.
        """
        x_in = self.x_[train]
        y_in = self.y_[train]

        gen_samples_ratio = np.unique(y_in, return_counts=True)[1]

        majority_class = np.array(gen_samples_ratio).argmax()
        num_majority_samples = np.max(np.array(gen_samples_ratio))

        x_balanced = np.copy(x_in)
        y_balanced = np.copy(y_in)

        # Perform oversampling
        for cls in range(self.num_classes):
            if cls != majority_class:
                samples_to_generate = num_majority_samples - gen_samples_ratio[cls]

                generated_samples = None

                # Generate the appropriate number of samples to equalize cls with the majority class.
                # print("\tSampling Class y:", cls, " Gen Samples ratio:", gen_samples_ratio[cls])
                if sampler.short_name_ == "GCOP" or sampler.short_name_ == "CTGAN" or sampler.short_name_ == "TVAE":
                    reference_data = pd.DataFrame(data={str(self.class_column): [cls] * samples_to_generate})
                    generated_samples = sampler.sampler_.sample_remaining_columns(
                        max_tries_per_batch=500, known_columns=reference_data).iloc[:, 0:self.dimensionality]

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

    @staticmethod
    def parse(path):
        g = gzip.open(path, 'rb')
        for lt in g:
            yield json.loads(lt)
