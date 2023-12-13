import os
from imblearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

# DeepCore ML imports
import DataTools
from Datasets import BaseDataset
from DataSamplers import DataSamplers
from Classifiers import Classifiers
from ResultHandler import ResultHandler

from generators.sb_gan import sbGAN
from generators.c_gan import cGAN
from generators.ct_gan import ctGAN


num_threads = 1
os.environ['OMP_NUM_THREADS'] = str(num_threads)

seed = 1
DataTools.set_random_states(seed)
np_random_state, torch_random_state, cuda_random_state = DataTools.get_random_states()

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

dataset = datasets['CM1']

dataset_1 = BaseDataset(random_state=seed)
dataset_1.load_from_csv(path=dataset['path'], feature_cols=dataset['features_cols'], class_col=dataset['class_col'])
dataset_1.display_params()

x = dataset_1.x_
y = dataset_1.y_

classifiers = Classifiers(random_state=seed)
results_list = []

# Create and train generators
# Use a Random Forest classifier to test the generated data quality. High accuracy reveals high quality data.
# clf = classifiers.models_[3]

# standardizer = StandardScaler()
# x_std = standardizer.fit_transform(x)
# gan = sbGAN(discriminator=(128, 128), generator=(128, 256, 128), method='knn', k=5, r=200, random_state=seed)
# gan = cGAN(discriminator=(128, 128), generator=(128, 256, 128), random_state=seed)
# gan = ctGAN(discriminator=(256, 256), generator=(128, 256, 128), pac=1)
# balanced_data = gan.fit_resample(x, y)
# print(balanced_data[0].shape)
# print(balanced_data[0])
# Simple training
# c_gan.train(x, y)

# latent_x = torch.randn((20, 3))
# print(latent_x)

# zeros = torch.zeros(20, 3)
# std = zeros + 1
# fakez = torch.normal(mean=zeros, std=std)
# print(fakez)

# Smart training
# synthetic_data, mean_costs = c_gan.smart_train(x, y, clf, epochs=500, batch_size=32)
# print(synthetic_data[499])

for clf in classifiers.models_:
    DataTools.reset_random_states(np_random_state, torch_random_state, cuda_random_state)
    samplers = DataSamplers(sampling_strategy='auto', random_state=seed)

    order = 0
    print("")
    for s in samplers.over_samplers_:
        order += 1
        print("Testing", clf.name_, "with", s.name_)

        pipe_line = make_pipeline(s.sampler_, StandardScaler(), clf.model_)

        dataset_1.cv_pipeline(pipe_line, num_folds=5, num_threads=num_threads, results_list=results_list,
                              classifier_str=clf.name_, sampler_str=s.name_, order=order)

        DataTools.reset_random_states(np_random_state, torch_random_state, cuda_random_state)

print(results_list)

presenter = ResultHandler(results_list)
print(presenter.to_df(columns=['Classifier', 'Sampler', 'Accuracy_Mean', 'Balanced_Accuracy_Mean', 'F1_Mean']))

