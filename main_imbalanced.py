import os
import sys

import numpy as np

import eval as eval_methods
import paths as paths

num_threads = 1
os.environ['OMP_NUM_THREADS'] = str(num_threads)
np.set_printoptions(linewidth=400, threshold=sys.maxsize)

seed = 1

datasets = {
    # Part 1
    # 'ecoli1': {'path': paths.bin_cont + 'ecoli1.csv', 'categorical_cols': (), 'class_col': 7},
    # 'ecoli2': {'path': paths.bin_cont + 'ecoli2.csv', 'categorical_cols': (), 'class_col': 7},
    # 'ecoli3': {'path': paths.bin_cont + 'ecoli3.csv', 'categorical_cols': (), 'class_col': 7},
    # 'ecoli4': {'path': paths.bin_cont + 'ecoli4.csv', 'categorical_cols': (), 'class_col': 7},

    # 'glass1': {'path': paths.bin_cont + 'glass1.csv', 'categorical_cols': (), 'class_col': 9},
    # 'glass2': {'path': paths.bin_cont + 'glass2.csv', 'categorical_cols': (), 'class_col': 9},
    # 'glass4': {'path': paths.bin_cont + 'glass4.csv', 'categorical_cols': (), 'class_col': 9},
    # 'glass5': {'path': paths.bin_cont + 'glass5.csv', 'categorical_cols': (), 'class_col': 9},
    # 'glass6': {'path': paths.bin_cont + 'glass6.csv', 'categorical_cols': (), 'class_col': 9},

    # 'pima': {'path': paths.bin_cont + 'pima.csv', 'categorical_cols': (), 'class_col': 8},

    # 'vehicle0': {'path': paths.bin_cont + 'vehicle0.csv', 'categorical_cols': (), 'class_col': 18},
    # 'vehicle1': {'path': paths.bin_cont + 'vehicle1.csv', 'categorical_cols': (), 'class_col': 18},
    # 'vehicle2': {'path': paths.bin_cont + 'vehicle2.csv', 'categorical_cols': (), 'class_col': 18},
    # 'vehicle3': {'path': paths.bin_cont + 'vehicle3.csv', 'categorical_cols': (), 'class_col': 18},

    # 'yeast1': {'path': paths.bin_cont + 'yeast1.csv', 'categorical_cols': (), 'class_col': 8},
    # 'yeast3': {'path': paths.bin_cont + 'yeast3.csv', 'categorical_cols': (), 'class_col': 8},
    # 'yeast4': {'path': paths.bin_cont + 'yeast4.csv', 'categorical_cols': (), 'class_col': 8},
    # 'yeast5': {'path': paths.bin_cont + 'yeast5.csv', 'categorical_cols': (), 'class_col': 8},
    # 'yeast6': {'path': paths.bin_cont + 'yeast6.csv', 'categorical_cols': (), 'class_col': 8},

    # Part 2
    # 'DryBean': {'path': paths.multi_cont + 'DryBean.csv', 'categorical_cols': (), 'class_col': 16}, # <-- TVAE FAILS
    # 'flare': {'path': paths.bin_disc + 'flareF.csv', 'categorical_cols': (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10),
    #            'class_col': 11},  # <-- COPGAN FAILS
    'car-vgood': {'path': paths.bin_disc + 'car-vgood.csv', 'categorical_cols': (0, 1, 2, 3, 4, 5), 'class_col': 6},   # <-- ADASYN FAILS
    'glass': {'path': paths.multi_cont + 'glass.csv', 'categorical_cols': (), 'class_col': 9},   # <-- ADASYN FAILS
    'contraceptive': {'path': paths.multi_mix + 'contraceptive.csv', 'categorical_cols': (4, 5, 8), 'class_col': 9},
    # 'yeast': {'path': paths.multi_cont + 'yeast.csv', 'categorical_cols': (), 'class_col': 8}, # <-- TVAE & ALL SMOTE FAIL

    # 'adult': {'path': paths.bin_mix + 'adult.csv', 'categorical_cols': (1, 3, 5, 6, 7, 8, 9, 13), 'class_col': 14},
    # 'vowel': {'path': paths.bin_mix + 'vowel0.csv', 'categorical_cols': (0, 1, 2), 'class_col': 13},
    # 'CreditCard': {'path': paths.bin_mix + 'creditcarddefault.csv', 'categorical_cols': (1, 2, 3, 5, 6, 7, 8, 9, 10),
    #               'class_col': 23},

    # 'balance': {'path': paths.multi_cont + 'balance.csv', 'categorical_cols': (), 'class_col': 4},  # Errors!
    # 'ecoli': {'path': paths.multi_cont + 'ecoli.csv', 'categorical_cols': (), 'class_col': 7},  # Errors!

    # 'hayes-roth': {'path': paths.multi_cont + 'hayes-roth.csv', 'categorical_cols': (), 'class_col': 4},  # <--Errors
    # 'new-thyroid': {'path': paths.multi_cont + 'new-thyroid.csv', 'categorical_cols': (), 'class_col': 5},
    # 'pageblocks': {'path': paths.multi_cont + 'pageblocks.csv', 'categorical_cols': (), 'class_col': 10},  # <--Errors
    # 'penbased': {'path': paths.multi_cont + 'penbased.csv', 'categorical_cols': (), 'class_col': 16},  # <--Errors
    # 'shuttle': {'path': paths.multi_cont + 'shuttle-rev.csv', 'categorical_cols': (), 'class_col': 9},
    # 'thyroid': {'path': paths.multi_cont + 'thyroid.csv', 'categorical_cols': (), 'class_col': 21},  # <-- TVAE FAILS

    # 'MobilePrice': {'path': paths.multi_cont + 'MobilePrice.csv', 'categorical_cols': (1, 3, 5, 17, 18, 19),
    #                'class_col': 20},
    # 'autos': {'path': paths.multi_mix + 'autos.csv', 'categorical_cols': (1, 2, 3, 4, 5, 6, 7, 13, 14, 16),
    #          'class_col': 25},  # <--Errors
}

####################################################################

datasets_multiclass = {
    # 'MagicGammaTelescope': {'path': st_path + 'magic_gamma_telescope.csv', 'categorical_cols': (), 'class_col': 10},
    # 'DryBean': {'path': st_path + 'Dry_Bean_Dataset.csv', 'categorical_cols': (), 'class_col': 16},
    'adult': {'path': paths.bin_disc + 'adult.csv', 'categorical_cols': (1, 3, 5, 6, 7, 8, 9, 13), 'class_col': 14},
}

# eval_methods.test_model('SBGAN', datasets['ecoli1'], seed)
# eval_methods.test_model('CTDGAN-R', datasets['flare'], seed)
# eval_methods.test_model('CTDGAN-R', datasets['contraceptive'], seed)

eval_methods.eval_resampling(datasets=datasets, transformer='standardizer', num_folds=5, random_state=seed)
# eval_methods.eval_detectability(datasets=datasets, transformer='standardizer', num_folds=5, random_state=seed)

# eval_methods.eval_ml_efficacy(datasets, num_threads, seed)

# Experiments performed in Information Sciences 2024 paper
# eval_methods.eval_oversampling_efficacy(datasets_imb, num_threads, seed)
