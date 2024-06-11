import pathlib
import gzip
import json

import numpy as np
import pandas as pd

import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.preprocessing import LabelEncoder
from sklearn.datasets import make_classification
from sklearn.model_selection import cross_validate
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

        Args:
            random_state: Controls random number generation. Set this to a fixed integer to get reproducible results.
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
        self.tags_ = None
        self.df_ = None

    # Create synthetic imbalanced datasets with numeric features
    def create_synthetic(self, n_samples=1000, n_classes=2, imb_ratio=(0.5, 0.5)):
        """
        Create a synthetic imbalanced dataset with numeric features.

        Args:
            n_samples: how many data instances to create.
            n_classes: the number of classes of the synthetic data instances
            imb_ratio: the imbalance ratio of the data (it must be a tuple comprising n_classes elements with sum=1)
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
        Load a Dataset from a CSV file and compute several basic statistics (number of samples, number of classes,
        input dimensionality). The target variables are Label Encoded.

        Args:
            path: The location of the input CSV file.
            feature_cols: A tuple the denotes the indices of the columns with the input variables.
            class_col: The integer index of the column that stores the target variables.
        """
        self._feature_columns = feature_cols
        self._class_column = class_col

        file_extension = pathlib.Path(path).suffix
        if file_extension == '.csv':
            input_df = pd.read_csv(path, encoding='utf-8', keep_default_na=True, na_values=['<null>'])
            # self.df_ = pd.read_csv(path, encoding='latin-1', header=None)
        else:
            input_df = self.get_df(path)

        # The class column must be the last one.
        self._class_column_name = input_df.columns[len(input_df.columns) - 1]

        # Remove rows with missing values
        input_df.dropna(inplace=True)
        input_df.reset_index(drop=True, inplace=True)

        # Shuffle the dataframe
        input_df.sample(frac=1)

        input_x = input_df.iloc[:, feature_cols]
        input_y = input_df.iloc[:, class_col]

        # Process the input vectors and place them to self.x_
        self.x_ = input_x.to_numpy()
        self.df_ = input_x

        # Label encode the target variables
        class_encoder = LabelEncoder()
        self.y_ = class_encoder.fit_transform(input_y.to_numpy())

        # Copy the label encoded class column to the Dataframe
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
                if sampler.short_name_ == "GCOP" or sampler.short_name_ == "CTGAN" or sampler.short_name_ == "TVAE":
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
