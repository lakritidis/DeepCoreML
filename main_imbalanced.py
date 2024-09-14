import os
import sys

import numpy as np

import eval as eval_methods
import paths as paths

num_threads = 1
os.environ['OMP_NUM_THREADS'] = str(num_threads)
np.set_printoptions(linewidth=400, threshold=sys.maxsize)

seed = 1

# Datasets with i) continuous columns and ii) two classes
datasets_bin_continuous = {
    # Software Defect Detection


    'ecoli1': {'path': paths.bin_cont + 'ecoli1.csv', 'categorical_cols': (), 'class_col': 7},
    'ecoli2': {'path': paths.bin_cont + 'ecoli2.csv', 'categorical_cols': (), 'class_col': 7},
    'ecoli3': {'path': paths.bin_cont + 'ecoli3.csv', 'categorical_cols': (), 'class_col': 7},
    'ecoli4': {'path': paths.bin_cont + 'ecoli4.csv', 'categorical_cols': (), 'class_col': 7},

    'glass1': {'path': paths.bin_cont + 'glass1.csv', 'categorical_cols': (), 'class_col': 9},
    'glass2': {'path': paths.bin_cont + 'glass2.csv', 'categorical_cols': (), 'class_col': 9},
    'glass4': {'path': paths.bin_cont + 'glass4.csv', 'categorical_cols': (), 'class_col': 9},
    'glass5': {'path': paths.bin_cont + 'glass5.csv', 'categorical_cols': (), 'class_col': 9},
    'glass6': {'path': paths.bin_cont + 'glass6.csv', 'categorical_cols': (), 'class_col': 9},

    # 'led7digit': {'path': paths.bin_cont + 'led7digit-all_vs_1.csv', 'categorical_cols': (), 'class_col': 7}, # <-- COPGAN FAILS

    'pima': {'path': paths.bin_cont + 'pima.csv', 'categorical_cols': (), 'class_col': 8},

    'vehicle0': {'path': paths.bin_cont + 'vehicle0.csv', 'categorical_cols': (), 'class_col': 18},
    'vehicle1': {'path': paths.bin_cont + 'vehicle1.csv', 'categorical_cols': (), 'class_col': 18},
    'vehicle2': {'path': paths.bin_cont + 'vehicle2.csv', 'categorical_cols': (), 'class_col': 18},
    'vehicle3': {'path': paths.bin_cont + 'vehicle3.csv', 'categorical_cols': (), 'class_col': 18},

    'yeast1': {'path': paths.bin_cont + 'yeast1.csv', 'categorical_cols': (), 'class_col': 8},
    'yeast3': {'path': paths.bin_cont + 'yeast3.csv', 'categorical_cols': (), 'class_col': 8},
    'yeast4': {'path': paths.bin_cont + 'yeast4.csv', 'categorical_cols': (), 'class_col': 8},
    'yeast5': {'path': paths.bin_cont + 'yeast5.csv', 'categorical_cols': (), 'class_col': 8},
    'yeast6': {'path': paths.bin_cont + 'yeast6.csv', 'categorical_cols': (), 'class_col': 8},
}


# Datasets with i) discrete columns and ii) two classes
datasets_bin_discrete = {
    'flare': {'path': paths.bin_disc + 'flare-F.csv',
              'categorical_cols': (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10), 'class_col': 11}, # <-- COPGAN FAILS
}

# Datasets with i) both continuous and discrete columns and ii) two classes
datasets_bin_mixed = {

}

# Datasets with i) continuous columns and ii) multiple classes
datasets_multiclass_continuous = {
    'balance': {'path': paths.multi_cont + 'balance.csv', 'categorical_cols': (), 'class_col': 4}, # Errors!
    'dermatology': {'path': paths.multi_cont + 'dermatology.csv', 'categorical_cols': (), 'class_col': 34}, # Missing values
    'ecoli': {'path': paths.multi_cont + 'ecoli.csv', 'categorical_cols': (), 'class_col': 7}, # Errors!
    'glass': {'path': paths.multi_cont + 'glass.csv', 'categorical_cols': (), 'class_col': 9},
    'hayes-roth': {'path': paths.multi_cont + 'hayes-roth.csv', 'categorical_cols': (), 'class_col': 4},  # <--Errors
    'new-thyroid': {'path': paths.multi_cont + 'new-thyroid.csv', 'categorical_cols': (), 'class_col': 5},
    'pageblocks': {'path': paths.multi_cont + 'pageblocks.csv', 'categorical_cols': (), 'class_col': 10},  # <--Errors
    'penbased': {'path': paths.multi_cont + 'penbased.csv', 'categorical_cols': (), 'class_col': 16},  # <--Errors
    'shuttle': {'path': paths.multi_cont + 'shuttle-rev.csv', 'categorical_cols': (), 'class_col': 9},
    'thyroid': {'path': paths.multi_cont + 'thyroid.csv', 'categorical_cols': (), 'class_col': 21},  # <-- TVAE FAILS
    'yeast': {'path': paths.multi_cont + 'yeast.csv', 'categorical_cols': (), 'class_col': 8},
}

# Datasets with i) discrete columns and ii) multiple classes
datasets_multiclass_discrete = {

}

# Datasets with i) both continuous and discrete columns and ii) multiple classes
datasets_multiclass_mixed = {
    'autos': {'path': paths.multi_mix + 'autos.csv',
              'categorical_cols': (1, 2, 3, 4, 5, 6, 7, 13, 14, 16), 'class_col': 25},  #  <--Errors
    'contraceptive': {'path': paths.multi_mix + 'contraceptive.csv', 'categorical_cols': (4, 5, 8), 'class_col': 9},
}


####################################################################

datasets_multiclass = {
    # 'MagicGammaTelescope': {'path': st_path + 'magic_gamma_telescope.csv', 'categorical_cols': (), 'class_col': 10},
    # 'DryBean': {'path': st_path + 'Dry_Bean_Dataset.csv', 'categorical_cols': (), 'class_col': 16},
    'adult': {'path': paths.bin_disc + 'adult.csv', 'categorical_cols': (1, 3, 5, 6, 7, 8, 9, 13), 'class_col': 14},
}

# eval_methods.test_model('CTDGAN-R', datasets_bin_continuous['ecoli1'], seed)
# eval_methods.test_model('CTGAN', datasets_bin_continuous['ecoli1'], seed)
# eval_methods.test_model('CTDGAN-R', datasets_bin_discrete['flare'], seed)
# eval_methods.test_model('CTDGAN-R', datasets_multiclass_mixed['contraceptive'], seed)

eval_methods.eval_resampling(datasets=datasets_bin_continuous, transformer='standardizer', num_folds=5, random_state=seed)
# eval_methods.eval_detectability(datasets=datasets_imb, transformer='standardizer', num_folds=5, random_state=seed)

# eval_methods.eval_ml_efficacy(datasets, num_threads, seed)

# Experiments performed in Information Sciences 2024 paper
# eval_methods.eval_oversampling_efficacy(datasets_imb, num_threads, seed)
