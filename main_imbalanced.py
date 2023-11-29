import DataTools

from Datasets import BaseDataset

from DataTransformers import FeatureTransformer

from imblearn.pipeline import make_pipeline

from sklearn.preprocessing import MinMaxScaler

from DataSamplers import DataSamplers
from Classifiers import Classifiers

from ResultHandler import ResultHandler

seed = 1
DataTools.set_random_states(seed)

dataset_path = 'C:/Users/Leo/PycharmProjects/datasets/soft_defect/csv/'

datasets = {
            'Synthetic': None,
            'AR1': {'path': dataset_path + 'ar1.csv', 'features_cols': range(0, 29), 'class_col': 29},
            'AR3': {'path': dataset_path + 'ar3.csv', 'features_cols': range(0, 29), 'class_col': 29},
            'AR4': {'path': dataset_path + 'ar4.csv', 'features_cols': range(0, 29), 'class_col': 29},
            'CM1': {'path': dataset_path + 'cm1.csv', 'features_cols': range(0, 21), 'class_col': 21},
            'JM1': {'path': dataset_path + 'jm1.csv', 'features_cols': range(0, 21), 'class_col': 21},
            'KC1': {'path': dataset_path + 'kc1.csv', 'features_cols': range(0, 21), 'class_col': 21},
            'KC2': {'path': dataset_path + 'kc2.csv', 'features_cols': range(0, 21), 'class_col': 21},
            'KC3': {'path': dataset_path + 'kc3.csv', 'features_cols': range(0, 39), 'class_col': 39},
            'MC1': {'path': dataset_path + 'mc1.csv', 'features_cols': range(0, 38), 'class_col': 38},
            'MC2': {'path': dataset_path + 'mc2.csv', 'features_cols': range(0, 39), 'class_col': 39},
            'MW1': {'path': dataset_path + 'mw1.csv', 'features_cols': range(0, 37), 'class_col': 37},
            'PC1': {'path': dataset_path + 'pc1.csv', 'features_cols': range(0, 21), 'class_col': 24},
            'PC2': {'path': dataset_path + 'pc2.csv', 'features_cols': range(0, 36), 'class_col': 36},
            'PC3': {'path': dataset_path + 'pc3.csv', 'features_cols': range(0, 37), 'class_col': 37},
            'PC4': {'path': dataset_path + 'pc4.csv', 'features_cols': range(0, 37), 'class_col': 37}
            }

dataset = datasets['AR4']

dataset_1 = BaseDataset(random_state=seed)
dataset_1.load_from_csv(path=dataset['path'], feature_cols=dataset['features_cols'], class_col=dataset['class_col'])
dataset_1.display_params()

x = dataset_1.x_
y = dataset_1.y_

classifiers = Classifiers(random_state=seed)
results_list = []

for clf in classifiers.models_:
    samplers = DataSamplers(sampling_strategy='auto', random_state=seed)

    order = 0
    for s in samplers.over_samplers_:
        order += 1
        print("Testing", clf.name_, "with", s.name_)

        trans = FeatureTransformer(feature_encoder=MinMaxScaler())
        pipe_line = make_pipeline(trans, s.sampler_, clf.model_)

        dataset_1.cv_pipeline(pipe_line, num_folds=5, num_threads=1, results_list=results_list,
                              classifier_str=clf.name_, sampler_str=s.name_, order=order)

presenter = ResultHandler(results_list)
print(presenter.to_latex(columns=['Classifier', 'Sampler', 'Accuracy_Mean', 'Balanced_Accuracy_Mean', 'F1_Mean']))
