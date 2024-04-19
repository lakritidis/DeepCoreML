import time

from generators.sb_gan import sbGAN
from generators.c_gan import cGAN
from generators.ct_gan import ctGAN

from generators.gaan_v1 import GAANv1
from generators.gaan_v2 import GAANv2
from generators.gaan_v3 import GAANv3
from generators.gaan_v4 import GAANv4

from Datasets import BaseDataset, FakeDataset
from DataSamplers import DataSamplers
from DataTools import set_random_states, get_random_states, reset_random_states
from ResultHandler import ResultHandler
from Classifiers import Classifiers

from imblearn.pipeline import make_pipeline
from imblearn.metrics import sensitivity_score, specificity_score

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score, accuracy_score, balanced_accuracy_score, precision_score, recall_score
from sklearn.preprocessing import StandardScaler

from sdv.metadata import SingleTableMetadata


def test_model(model, dataset, seed):
    set_random_states(seed)

    """Create, train and test individual generators.
    """
    dset = BaseDataset(dataset['name'], random_state=seed)
    dset.load_from_csv(path=dataset['path'], feature_cols=dataset['features_cols'], class_col=dataset['class_col'])
    dset.display_params()

    x = dset.x_
    y = dset.y_

    if model == "SB-GAN":
        gan = sbGAN(discriminator=(128, 128), generator=(128, 256, 128), method='knn', pac=1, k=5, random_state=seed)
    elif model == "GAANv1":
        gan = GAANv1(discriminator=(128, 128), generator=(128, 256, 128), pac=1, random_state=seed)
    elif model == "GAANv2":
        gan = GAANv2(discriminator=(128, 128), generator=(128, 256, 128), pac=1, max_clusters=10, random_state=seed)
    elif model == "GAANv3":
        gan = GAANv3(discriminator=(128, 128), generator=(128, 256, 128), pac=1, max_clusters=10, random_state=seed)
    elif model == "CONDITIONAL-GAN":
        gan = cGAN(discriminator=(128, 128), generator=(128, 256, 128), pac=1, random_state=seed)
    elif model == "CTGAN":
        gan = ctGAN(discriminator=(256, 256), generator=(128, 256, 128), pac=1)
    else:
        gan = GAANv4(discriminator=(128, 128), generator=(128, 256, 128), pac=1, max_clusters=20, random_state=seed)

    balanced_data = gan.fit_resample(x, y)
    print(balanced_data[0].shape)
    print(balanced_data[0])


# Manually determine what takes place in each cross validation fold
def eval_resampling(datasets, num_folds, random_state):
    set_random_states(random_state)
    np_random_state, torch_random_state, cuda_random_state = get_random_states()

    test_samplers = DataSamplers(SingleTableMetadata(), sampling_strategy='auto', random_state=random_state)
    num_samplers = len(test_samplers.over_samplers_)
    print(num_samplers)

    scorers = {
        'accuracy': accuracy_score,
        'balanced_accuracy': balanced_accuracy_score,
        'sensitivity': sensitivity_score,
        'specificity': specificity_score,
        'f1': f1_score,
        'precision': precision_score,
        'recall': recall_score,
    }

    n_dataset, n_fold = 0, 0
    performance_list = []

    # For each dataset
    for key in datasets.keys():
        reset_random_states(np_random_state, torch_random_state, cuda_random_state)
        n_dataset += 1
        # if num_dataset > 1:
        #     break

        ds = datasets[key]
        original_dataset = BaseDataset(ds['name'], random_state=random_state)
        original_dataset.load_from_csv(path=ds['path'], feature_cols=ds['features_cols'], class_col=ds['class_col'])

        print("\n=================================\n Evaluating dataset", key, " - shape:", original_dataset.x_.shape)

        # Convert all columns to numerical
        metadata = SingleTableMetadata()
        metadata.detect_from_dataframe(original_dataset.df_)
        for k in metadata.columns.keys():
            metadata.columns[k] = {'sdtype': 'numerical'}
        metadata.columns[k] = {'sdtype': 'categorical'}

        n_fold = 0
        skf = StratifiedKFold(n_splits=num_folds, shuffle=True, random_state=random_state)

        # For each fold
        for train, test in skf.split(original_dataset.x_, original_dataset.y_):
            reset_random_states(np_random_state, torch_random_state, cuda_random_state)
            n_fold += 1
            print("\tFold: ", n_fold)

            # Initialize a new set of data samplers
            samplers = DataSamplers(metadata, sampling_strategy='auto', random_state=random_state)

            n_sampler = 0
            # For each sampler, fit and over/under-sample
            for sampler in samplers.over_samplers_:
                n_sampler += 1
                t_s = time.time()

                reset_random_states(np_random_state, torch_random_state, cuda_random_state)
                print("\t\tSampler: ", sampler.short_name_)

                # Fit & balance step
                if (sampler.short_name_ == 'ROS' or sampler.short_name_ == 'SMOTE' or sampler.short_name_ == 'B-SMOTE'
                        or sampler.short_name_ == 'SVM-SMOTE' or sampler.short_name_ == 'KMN-SMOTE'
                        or sampler.short_name_ == 'ADASYN' or sampler.short_name_ == 'CBR'
                        or sampler.short_name_ == 'CGAN' or sampler.short_name_ == 'SBGAN' or sampler.short_name_ == 'ctGAN'):

                    x_balanced, y_balanced = (sampler.sampler_.fit_resample
                                              (original_dataset.x_[train], original_dataset.y_[train]))

                elif sampler.short_name_ == 'GCOP' or sampler.short_name_ == 'CTGAN' or sampler.short_name_ == 'TVAE':

                    sampler.sampler_.fit(original_dataset.df_.iloc[train, :])
                    x_balanced, y_balanced = original_dataset.balance(sampler, train)

                elif sampler.short_name_ == 'GAAN':

                    sampler.sampler_.fit(original_dataset.x_[train], original_dataset.y_[train])
                    x_balanced, y_balanced = original_dataset.balance(sampler, train)

                else:
                    x_balanced = original_dataset.x_[train]
                    y_balanced = original_dataset.y_[train]

                oversampling_duration = time.time() - t_s

                # Initialize a new set of classifiers
                classifiers = Classifiers(random_state=random_state)

                # For each classifier
                for classifier in classifiers.models_:
                    reset_random_states(np_random_state, torch_random_state, cuda_random_state)

                    classifier.fit(x_balanced, y_balanced)
                    y_predict = classifier.predict(original_dataset.x_[test])

                    for scorer in scorers:
                        performance = scorers[scorer](original_dataset.y_[test], y_predict)

                        lst = [key, n_fold, sampler.short_name_, classifier.name_, scorer, performance]
                        performance_list.append(lst)

                    lst = [key, n_fold, sampler.short_name_, classifier.name_, "Fit Time", oversampling_duration]
                    performance_list.append(lst)

    drh = ResultHandler("Resampling", performance_list)
    drh.record_results("resampling_mean")

    print("\n=================================\n")


# This function uses an ImbLearn Pipeline. Each Oversampling/Under-sampling method MUST support the fit_resample method
# To use plug-and-play implementations that do not implement fit_resample, please use eval_resampling
def eval_oversampling_efficacy(datasets, num_threads, random_state):
    """Test the ability of a Generator to improve the performance of a classifier by balancing an imbalanced dataset.
    The Generator performs over-sampling on the minority classes and equalizes the number of samples per class.
    Algorithm:

      1. For each dataset d, for each classifier c, for each sampler s
      2. Fit s
      3. d_balanced <--- over-sample(d with s)
      4. Test classification performance of c on d_balanced
      5. Steps 2-4 are embedded in a pipe-line; the pipe-line is cross validated with 5 folds.
    """
    set_random_states(random_state)
    np_random_state, torch_random_state, cuda_random_state = get_random_states()

    classifiers = Classifiers(random_state=random_state)

    results_list = []

    order = 0
    # For each input dataset
    x = 0
    for key in datasets.keys():
        x += 1
        # if x > 1:
        #     break
        print("\n=================================\n Evaluating dataset", key, "\n=================================\n")
        reset_random_states(np_random_state, torch_random_state, cuda_random_state)
        ds = datasets[key]
        original_dataset = BaseDataset(ds['name'], random_state=random_state)
        original_dataset.load_from_csv(path=ds['path'], feature_cols=ds['features_cols'], class_col=ds['class_col'])

        dataset_results_list = []

        # Convert all columns to numerical
        metadata = SingleTableMetadata()
        metadata.detect_from_dataframe(original_dataset.df_)
        for k in metadata.columns.keys():
            metadata.columns[k] = {'sdtype': 'numerical'}
        metadata.columns[original_dataset.get_class_column() - 1] = {'sdtype': 'categorical'}

        # For each classifier
        for clf in classifiers.models_:
            samplers = DataSamplers(metadata, sampling_strategy='auto', random_state=random_state)

            # For each over-sampler, balance the input dataset. The fit_resample method of each sampler is called
            # internally by the `imblearn` pipeline and the cross validator.
            for s in samplers.over_samplers_:
                reset_random_states(np_random_state, torch_random_state, cuda_random_state)
                order += 1

                print("Testing", clf.name_, "with", s.name_)

                pipe_line = make_pipeline(s.sampler_, StandardScaler(), clf.model_)
                r, _ = original_dataset.cross_val(estimator=pipe_line, num_folds=5, num_threads=num_threads,
                                                  classifier_str=clf.name_, sampler_str=s.name_, order=order)

                for e in range(len(r)):
                    results_list.append(r[e])
                    dataset_results_list.append(r[e])

        # Record the results for this dataset
        drh = ResultHandler("Oversampling", dataset_results_list)
        drh.record_results(key + "_oversampling")

    # Record the results for all datasets
    rh = ResultHandler("Oversampling", results_list)
    rh.record_results("oversampling")


def eval_ml_efficacy(datasets, num_threads, random_state):
    set_random_states(random_state)
    np_random_state, torch_random_state, cuda_random_state = get_random_states()

    results_list = []
    cv_results = []

    order = 0
    # For each input dataset
    x = 0
    for key in datasets.keys():
        x += 1
        # if x > 2:
        #     break
        print("\n=================================\n Evaluating dataset", key, "\n=================================\n")
        order += 1
        ds = datasets[key]
        original_dataset = BaseDataset(ds['name'], random_state=random_state)
        original_dataset.load_from_csv(path=ds['path'], feature_cols=ds['features_cols'], class_col=ds['class_col'])

        dataset_results_list = []
        dataset_cv_results = []

        classifiers = Classifiers(random_state=random_state)
        samplers = DataSamplers(sampling_strategy='auto', random_state=random_state)

        # For each classifier, test classification performance on the original dataset
        for clf in classifiers.models_:
            reset_random_states(np_random_state, torch_random_state, cuda_random_state)
            r, cvr = original_dataset.cross_val(estimator=clf.model_, num_folds=5, num_threads=num_threads,
                                                classifier_str=clf.name_, sampler_str="", order=order)

            results_list.append(r)
            dataset_results_list.append(r)
            cv_results.append(cvr)
            dataset_cv_results.append(cvr)

        # For each over-sampler, create a synthetic dataset with the same class distribution as the original one
        for s in samplers.over_samplers_:
            reset_random_states(np_random_state, torch_random_state, cuda_random_state)

            synthetic_dataset = FakeDataset(ds, key, random_state=random_state)
            synthetic_dataset.synthesize(s.sampler_)

            # For each classifier, test the classification performance on the synthetic dataset
            classifiers = Classifiers(random_state=random_state)
            for clf in classifiers.models_:
                print("Testing", clf.name_, "with", s.name_)
                order += 1
                r, cvr = synthetic_dataset.cross_val(estimator=clf.model_, num_folds=5, num_threads=num_threads,
                                                     classifier_str=clf.name_, sampler_str=s.name_, order=order)

                results_list.append(r)
                dataset_results_list.append(r)
                cv_results.append(cvr)
                dataset_cv_results.append(cvr)

        # Record the results for this dataset
        drh = ResultHandler("Synthetic", dataset_results_list, dataset_cv_results)
        drh.record_results(key + "_synthetic")

    # Record the results for all datasets
    rh = ResultHandler("Synthetic", results_list, cv_results)
    rh.record_results("synthetic")


def eval_detectability(datasets, num_threads, random_state):
    """To evaluate how hard it is to distinguish between real and synthetic instances, we:
     1. Create a synthetic dataset with the same number of samples and class distribution as the original one. We
        mark the synthetic samples with flag 0.
     2. We mark the original samples with flag 1.
     3. Merge and shuffle the datasets -> create a new dataset
     4. Train a classifier on the new dataset and try to predict the flag. The easier it is to predict the flag, the
        more distinguishable between real and synthetic data.
    """
    set_random_states(random_state)
    np_random_state, torch_random_state, cuda_random_state = get_random_states()

    results_list = []
    cv_results = []

    x = 0
    order = 0
    for key in datasets.keys():
        x += 1
        # if x > 2:
        #     break
        print("\n=================================\n Evaluating dataset", key, "\n=================================\n")
        order += 1
        ds = datasets[key]
        reset_random_states(np_random_state, torch_random_state, cuda_random_state)

        dataset_results_list = []
        dataset_cv_results = []

        # Steps 1 & 2: Create a synthetic dataset with the same class distribution as the original. Merge the
        # original and the synthetic datasets and shuffle.
        samplers = DataSamplers(sampling_strategy='auto', random_state=random_state)
        for s in samplers.over_samplers_:
            reset_random_states(np_random_state, torch_random_state, cuda_random_state)

            synthetic_dataset = FakeDataset(ds, key, random_state=random_state)
            synthetic_dataset.synthesize_merge(s.sampler_)

            classifiers = Classifiers(random_state=random_state)
            for clf in classifiers.models_:
                print("Testing", clf.name_, "with", s.name_)
                order += 1
                r, cvr = synthetic_dataset.cross_val(estimator=clf.model_, num_folds=5, num_threads=num_threads,
                                                     classifier_str=clf.name_, sampler_str=s.name_, order=order)

                results_list.append(r)
                dataset_results_list.append(r)
                cv_results.append(cvr)
                dataset_cv_results.append(cvr)

            # Record the results for this dataset
        drh = ResultHandler("Detectability", dataset_results_list, dataset_cv_results)
        drh.record_results(key + "_detectability")

    # Record the results for all datasets
    rh = ResultHandler("Detectability", results_list, cv_results)
    rh.record_results("detectability")
