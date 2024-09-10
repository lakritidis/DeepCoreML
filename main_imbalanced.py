import os
import sys

import numpy as np

import eval as eval_methods
import paths as paths

num_threads = 1
os.environ['OMP_NUM_THREADS'] = str(num_threads)
np.set_printoptions(linewidth=400, threshold=sys.maxsize)

seed = 5

# Datasets with i) continuous columns and ii) two classes
datasets_bin_continuous = {
    # Software Defect Detection
    'AR1': {'path': paths.bin_con_path + 'ar1.csv', 'categorical_cols': (), 'class_col': 29},
    'AR3': {'path': paths.bin_con_path + 'ar3.csv', 'categorical_cols': (), 'class_col': 29},
    'AR4': {'path': paths.bin_con_path + 'ar4.csv', 'categorical_cols': (), 'class_col': 29},
    'CM1': {'path': paths.bin_con_path + 'cm1.csv', 'categorical_cols': (), 'class_col': 21},
    # 'JM1': {'path': paths.bin_con_path + 'jm1.csv', 'categorical_cols': (), 'class_col': 21}, <-- very high imbalance
    'KC1': {'path': paths.bin_con_path + 'kc1.csv', 'categorical_cols': (), 'class_col': 21},
    'KC2': {'path': paths.bin_con_path + 'kc2.csv', 'categorical_cols': (), 'class_col': 21},
    'KC3': {'path': paths.bin_con_path + 'kc3.csv', 'categorical_cols': (), 'class_col': 39},
    # 'MC1': {'path': paths.bin_con_path + 'mc1.csv', 'categorical_cols': (), 'class_col': 38}, <-- TVAE FAILS
    'MC2': {'path': paths.bin_con_path + 'mc2.csv', 'categorical_cols': (), 'class_col': 39},
    'MW1': {'path': paths.bin_con_path + 'mw1.csv', 'categorical_cols': (), 'class_col': 37},
    'PC1': {'path': paths.bin_con_path + 'pc1.csv', 'categorical_cols': (), 'class_col': 21},
    # 'PC2': {'path': paths.bin_con_path + 'pc2.csv', 'categorical_cols': (), 'class_col': 36},  <-- TVAE FAILS
    'PC3': {'path': paths.bin_con_path + 'pc3.csv', 'categorical_cols': (), 'class_col': 37},
    'PC4': {'path': paths.bin_con_path + 'pc4.csv', 'categorical_cols': (), 'class_col': 37},
    'ANT-1.3': {'path': paths.bin_con_path + 'ant-1.3.csv', 'categorical_cols': (), 'class_col': 20},
    'ANT-1.4': {'path': paths.bin_con_path + 'ant-1.4.csv', 'categorical_cols': (), 'class_col': 20},
    'ANT-1.5': {'path': paths.bin_con_path + 'ant-1.5.csv', 'categorical_cols': (), 'class_col': 20},
    'ANT-1.6': {'path': paths.bin_con_path + 'ant-1.6.csv', 'categorical_cols': (), 'class_col': 20},
    'ANT-1.7': {'path': paths.bin_con_path + 'ant-1.7.csv', 'categorical_cols': (), 'class_col': 20},
    # 'CAMEL-1.0': {'path': paths.bin_con_path + 'camel-1.0.csv', 'categorical_cols': (), 'class_col': 20}, <-- SMOTE FAILS
    # 'CAMEL-1.2': {'path': paths.bin_con_path + 'camel-1.2.csv', 'categorical_cols': (), 'class_col': 20}, <-- COPGAN FAILS
    'CAMEL-1.4': {'path': paths.bin_con_path + 'camel-1.4.csv', 'categorical_cols': (), 'class_col': 20},
    'CAMEL-1.6': {'path': paths.bin_con_path + 'camel-1.6.csv', 'categorical_cols': (), 'class_col': 20},
    'IVY-1.1': {'path': paths.bin_con_path + 'ivy-1.1.csv', 'categorical_cols': (), 'class_col': 20},
    'IVY-1.4': {'path': paths.bin_con_path + 'ivy-1.4.csv', 'categorical_cols': (), 'class_col': 20},
    'IVY-2.0': {'path': paths.bin_con_path + 'ivy-2.0.csv', 'categorical_cols': (), 'class_col': 20},
    # 'JEDIT-3.2': {'path': paths.bin_con_path + 'jedit-3.2.csv', 'categorical_cols': (), 'class_col': 20},  # <-- COPGAN FAILS
    # 'JEDIT-4.0': {'path': paths.bin_con_path + 'jedit-4.0.csv', 'categorical_cols': (), 'class_col': 20},  # <-- COPGAN FAILS
    # 'JEDIT-4.1': {'path': paths.bin_con_path + 'jedit-4.1.csv', 'categorical_cols': (), 'class_col': 20},  # <-- COPGAN FAILS
    # 'JEDIT-4.2': {'path': paths.bin_con_path + 'jedit-4.2.csv', 'categorical_cols': (), 'class_col': 20},    # <-- COPGAN FAILS
    # 'JEDIT-4.3': {'path': paths.bin_con_path + 'jedit-4.3.csv', 'categorical_cols': (), 'class_col': 20},    # <-- TVAE FAILS
    'LOG4J-1.0': {'path': paths.bin_con_path + 'log4j-1.0.csv', 'categorical_cols': (), 'class_col': 20},
    'LOG4J-1.1': {'path': paths.bin_con_path + 'log4j-1.1.csv', 'categorical_cols': (), 'class_col': 20},
    'LOG4J-1.2': {'path': paths.bin_con_path + 'log4j-1.2.csv', 'categorical_cols': (), 'class_col': 20},
    # 'LUCE-2.0': {'path': paths.bin_con_path + 'lucene-2.0.csv', 'categorical_cols': (), 'class_col': 20},    # <-- ADASYN FAILS
    'LUCE-2.2': {'path': paths.bin_con_path + 'lucene-2.2.csv', 'categorical_cols': (), 'class_col': 20},
    'LUCE-2.4': {'path': paths.bin_con_path + 'lucene-2.4.csv', 'categorical_cols': (), 'class_col': 20},
    # 'POI-1.5': {'path': paths.bin_con_path + 'poi-1.5.csv', 'categorical_cols': (), 'class_col': 20},        # <-- COPGAN FAILS
    'POI-2.0': {'path': paths.bin_con_path + 'poi-2.0.csv', 'categorical_cols': (), 'class_col': 20},
    'POI-2.5': {'path': paths.bin_con_path + 'poi-2.5.csv', 'categorical_cols': (), 'class_col': 20},
    'POI-3.0': {'path': paths.bin_con_path + 'poi-3.0.csv', 'categorical_cols': (), 'class_col': 20},
    'SYN1.0': {'path': paths.bin_con_path + 'synapse-1.0.csv', 'categorical_cols': (), 'class_col': 20},
    'SYN1.1': {'path': paths.bin_con_path + 'synapse-1.1.csv', 'categorical_cols': (), 'class_col': 20},
    'SYN1.2': {'path': paths.bin_con_path + 'synapse-1.2.csv', 'categorical_cols': (), 'class_col': 20},
    # 'VL14': {'path': paths.bin_con_path + 'velocity-1.4.csv', 'categorical_cols': (), 'class_col': 20},      # <-- COPGAN FAILS
    # 'VL15': {'path': paths.bin_con_path + 'velocity-1.5.csv', 'categorical_cols': (), 'class_col': 20},      # <-- COPGAN FAILS
    'VL16': {'path': paths.bin_con_path + 'velocity-1.6.csv', 'categorical_cols': (), 'class_col': 20},
    'XER-1.2': {'path': paths.bin_con_path + 'xerces-1.2.csv', 'categorical_cols': (), 'class_col': 20},     # <-- COPGAN FAILS
    # 'XER-1.3': {'path': paths.bin_con_path + 'xerces-1.3.csv', 'categorical_cols': (), 'class_col': 20},
    'XER-1.4': {'path': paths.bin_con_path + 'xerces-1.4.csv', 'categorical_cols': (), 'class_col': 20},

    'ecoli1': {'path': paths.bin_con_path + 'ecoli1.csv', 'categorical_cols': (), 'class_col': 7},
    'ecoli2': {'path': paths.bin_con_path + 'ecoli2.csv', 'categorical_cols': (), 'class_col': 7},
    'ecoli3': {'path': paths.bin_con_path + 'ecoli3.csv', 'categorical_cols': (), 'class_col': 7},
    'ecoli4': {'path': paths.bin_con_path + 'ecoli4.csv', 'categorical_cols': (), 'class_col': 7},

    'glass1': {'path': paths.bin_con_path + 'glass1.csv', 'categorical_cols': (), 'class_col': 9},
    'glass2': {'path': paths.bin_con_path + 'glass2.csv', 'categorical_cols': (), 'class_col': 9},
    'glass4': {'path': paths.bin_con_path + 'glass4.csv', 'categorical_cols': (), 'class_col': 9},
    'glass5': {'path': paths.bin_con_path + 'glass5.csv', 'categorical_cols': (), 'class_col': 9},
    'glass6': {'path': paths.bin_con_path + 'glass6.csv', 'categorical_cols': (), 'class_col': 9},

    # 'led7digit': {'path': paths.bin_con_path + 'led7digit-all_vs_1.csv', 'categorical_cols': (), 'class_col': 7}, # <-- COPGAN FAILS

    'pima': {'path': paths.bin_con_path + 'pima.csv', 'categorical_cols': (), 'class_col': 8},

    'vehicle0': {'path': paths.bin_con_path + 'vehicle0.csv', 'categorical_cols': (), 'class_col': 18},
    'vehicle1': {'path': paths.bin_con_path + 'vehicle1.csv', 'categorical_cols': (), 'class_col': 18},
    'vehicle2': {'path': paths.bin_con_path + 'vehicle2.csv', 'categorical_cols': (), 'class_col': 18},
    'vehicle3': {'path': paths.bin_con_path + 'vehicle3.csv', 'categorical_cols': (), 'class_col': 18},

    'yeast1': {'path': paths.bin_con_path + 'yeast1.csv', 'categorical_cols': (), 'class_col': 8},
    'yeast3': {'path': paths.bin_con_path + 'yeast3.csv', 'categorical_cols': (), 'class_col': 8},
    'yeast4': {'path': paths.bin_con_path + 'yeast4.csv', 'categorical_cols': (), 'class_col': 8},
    'yeast5': {'path': paths.bin_con_path + 'yeast5.csv', 'categorical_cols': (), 'class_col': 8},
    'yeast6': {'path': paths.bin_con_path + 'yeast6.csv', 'categorical_cols': (), 'class_col': 8},
}


# Datasets with i) discrete columns and ii) two classes
datasets_bin_discrete = {
    'flare': {'path': paths.bin_dis_path + 'flare-F.csv', 'categorical_cols': (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10), 'class_col': 11}, # <-- COPGAN FAILS
}

# Datasets with i) both continuous and discrete columns and ii) two classes
datasets_bin_mixed = {

}

# Datasets with i) continuous columns and ii) multiple classes
datasets_multiclass_continuous = {
    'balance': {'path': paths.multi_con_path + 'balance.csv', 'categorical_cols': (), 'class_col': 4}, # Errors!
    'dermatology': {'path': paths.multi_con_path + 'dermatology.csv', 'categorical_cols': (), 'class_col': 34}, # Missing values
    'ecoli': {'path': paths.multi_con_path + 'ecoli.csv', 'categorical_cols': (), 'class_col': 7}, # Errors!
    'glass': {'path': paths.multi_con_path + 'glass.csv', 'categorical_cols': (), 'class_col': 9},
    'hayes-roth': {'path': paths.multi_con_path + 'hayes-roth.csv', 'categorical_cols': (), 'class_col': 4}, # Errors
    'new-thyroid': {'path': paths.multi_con_path + 'new-thyroid.csv', 'categorical_cols': (), 'class_col': 5},
    'pageblocks': {'path': paths.multi_con_path + 'pageblocks.csv', 'categorical_cols': (), 'class_col': 10}, # Errors
    'penbased': {'path': paths.multi_con_path + 'penbased.csv', 'categorical_cols': (), 'class_col': 16}, # ! Errors
    'shuttle': {'path': paths.multi_con_path + 'shuttle-rev.csv', 'categorical_cols': (), 'class_col': 9},
    'thyroid': {'path': paths.multi_con_path + 'thyroid.csv', 'categorical_cols': (), 'class_col': 21}, # ! Error TVAE
    'yeast': {'path': paths.multi_con_path + 'yeast.csv', 'categorical_cols': (), 'class_col': 8},
}

# Datasets with i) discrete columns and ii) multiple classes
datasets_multiclass_discrete = {

}

# Datasets with i) both continuous and discrete columns and ii) multiple classes
datasets_multiclass_mixed = {
    'autos': {'path': paths.multi_mix_path + 'autos.csv', 'categorical_cols': (1, 2, 3, 4, 5, 6, 7, 13, 14, 16), 'class_col': 25}, # Errors!
    'contraceptive': {'path': paths.multi_mix_path + 'contraceptive.csv', 'categorical_cols': (4, 5, 8), 'class_col': 9},
}


####################################################################

datasets_multiclass = {
    # 'MagicGammaTelescope': {'path': st_path + 'magic_gamma_telescope.csv', 'categorical_cols': (), 'class_col': 10},
    # 'DryBean': {'path': st_path + 'Dry_Bean_Dataset.csv', 'categorical_cols': (), 'class_col': 16},
    # 'adult': {'path': st_path + 'adult.csv', 'categorical_cols': (1, 3, 5, 6, 7, 8, 9, 13), 'class_col': 14},
}

# eval_methods.test_model('CTDGAN', datasets_bin_continuous['ecoli1'], seed)
# eval_methods.test_model('CTDGAN', datasets_bin_discrete['flare'], seed)
# eval_methods.test_model('CTDGAN', datasets_multiclass_mixed['contraceptive'], seed)

eval_methods.eval_resampling(datasets=datasets_bin_continuous, transformer='standardizer', num_folds=5, random_state=seed)
# eval_methods.eval_detectability(datasets=datasets_imb, transformer='standardizer', num_folds=5, random_state=seed)

# eval_methods.eval_ml_efficacy(datasets, num_threads, seed)

# Experiments performed in Information Sciences 2024 paper
# eval_methods.eval_oversampling_efficacy(datasets_imb, num_threads, seed)