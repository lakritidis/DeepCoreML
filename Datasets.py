import pathlib
import gzip
import json

import pandas as pd

import seaborn as sns
import matplotlib.pyplot as plt
from imblearn.metrics import sensitivity_score, specificity_score

from sklearn.preprocessing import LabelEncoder
from sklearn.datasets import make_classification
from sklearn.model_selection import cross_validate
from sklearn.metrics import make_scorer, f1_score, accuracy_score, balanced_accuracy_score

import warnings
warnings.filterwarnings("ignore")

pd.set_option('display.max_rows', 100)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', -1)


# #################################################################################################################
# DATASET BASE CLASS
class BaseDataset:
    def __init__(self, random_state=0):
        """
        Dataset initializer: BaseDataset is the base class of all dataset subclasses.

        :param random_state: controls random number generation. Set this to a fixed integer to get reproducible results.
        """
        self.seed_ = random_state
        self.dimensionality_ = 0
        self.num_classes_ = 0
        self.num_samples_ = 0

        self.feature_columns_ = 0
        self.class_column_ = 0

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
                class_sep=0.5, n_informative=2, n_redundant=0, n_repeated=0, random_state=self.seed_)

        elif n_classes == 4:
            synthetic_dataset = make_classification(
                n_samples=n_samples, n_features=2, n_clusters_per_class=1, flip_y=0, n_classes=4, weights=imb_ratio,
                class_sep=1.0, n_informative=2, n_redundant=0, n_repeated=0, random_state=self.seed_)

        self.feature_columns_ = 0
        self.class_column_ = 1

        self.x_ = synthetic_dataset[0]
        self.y_ = synthetic_dataset[1]

        self.num_classes_ = n_classes
        self.num_samples_ = n_samples
        self.dimensionality_ = self.x_.shape[1]

        print("Num Samples:", self.num_samples_, "\nClass Distribution:")
        for k in range(self.num_classes_):
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
        self.feature_columns_ = feature_cols
        self.class_column_ = class_col

        file_extension = pathlib.Path(path).suffix
        if file_extension == '.csv':
            self.df_ = pd.read_csv(path, encoding='utf-8')
            # self.df_ = pd.read_csv(path, encoding='latin-1', header=None)
        else:
            self.df_ = self.get_df(path)

        # Shuffle the dataframe
        self.df_.sample(frac=1)

        # Convert x and y to NumPy arrays
        self.x_ = self.df_.iloc[:, feature_cols].to_numpy()
        self.y_ = self.df_.iloc[:, class_col].to_numpy()

        # Label encode the target variables
        class_encoder = LabelEncoder()
        self.y_ = class_encoder.fit_transform(self.y_)

        # Useful quick reference statistics
        self.num_classes_ = len(self.df_.iloc[:, class_col].unique())
        self.num_samples_ = self.x_.shape[0]
        self.dimensionality_ = self.x_.shape[1]

    # Display the basic dataset parameters
    def display_params(self):
        """
        Display the basic dataset parameters
        :return:
        """
        print("Num Samples:", self.num_samples_, "- Dimensions:", self.dimensionality_)
        print("Classes:", self.num_classes_, "- Class Distribution:")
        for k in range(self.num_classes_):
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
    def cv_pipeline(self, pipeline, num_folds, num_threads, results_list, classifier_str, sampler_str, order):
        """
        Applies a classification pipeline to the dataset, and then, it evaluates the classification performance by using
        cross validation.

        :param pipeline: A multi-stage pipeline object. The pipeline must include a classifier at its final stage.
        :param num_folds: Number of folds of cross validation.
        :param num_threads: The number of parallel threads to deploy for executing cross validation.
        :param results_list: A list to store the results of performance evaluation.
        :param classifier_str: A custom string to describe the classifier of the pipeline.
        :param sampler_str: A custom string to describe the other stages of the pipeline.
        :param order: An assistant variable for `results_list`
        :return: object that stores all cross validation results
        """
        scorers = {
            'accuracy_score': make_scorer(accuracy_score),
            'balanced_accuracy_score': make_scorer(balanced_accuracy_score),
            'sensitivity_score': make_scorer(sensitivity_score, average='weighted'),
            'specificity_score': make_scorer(specificity_score, average='weighted'),
            'f1_score': make_scorer(f1_score, average='weighted')
        }

        # cross_validate uses Stratified kFold when cv is int
        cv_results = cross_validate(pipeline, self.x_, self.y_, cv=num_folds, scoring=scorers, return_train_score=True,
                                    return_estimator=True, n_jobs=num_threads, error_score='raise')

        results_list.append([
            classifier_str,
            sampler_str,
            cv_results['test_accuracy_score'].mean(),
            cv_results['test_accuracy_score'].std(),
            cv_results['test_balanced_accuracy_score'].mean(),
            cv_results['test_balanced_accuracy_score'].std(),
            cv_results['test_sensitivity_score'].mean(),
            cv_results['test_sensitivity_score'].std(),
            cv_results['test_specificity_score'].mean(),
            cv_results['test_specificity_score'].std(),
            cv_results['test_f1_score'].mean(),
            cv_results['test_f1_score'].std(),
            order
        ])

        return cv_results

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
