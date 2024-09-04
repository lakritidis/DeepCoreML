import os
import sys

import numpy as np

import eval as eval_methods
import paths as paths


num_threads = 1
os.environ['OMP_NUM_THREADS'] = str(num_threads)
np.set_printoptions(linewidth=400, threshold=sys.maxsize)

seed = 0
sd_path = paths.software_defect_dataset_path
id_path = paths.imbalanced_dataset_path


datasets = {
    'ecoli1': {'path': id_path + 'ecoli1.csv', 'categorical_cols': (), 'class_col': 7},
    'ecoli2': {'path': id_path + 'ecoli2.csv', 'categorical_cols': (), 'class_col': 7},
    'ecoli3': {'path': id_path + 'ecoli3.csv', 'categorical_cols': (), 'class_col': 7},
    'ecoli4': {'path': id_path + 'ecoli4.csv', 'categorical_cols': (), 'class_col': 7},

    # 'flare': {'path': id_path + 'flare-F.csv', 'categorical_cols': (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10), 'class_col': 11},

    'glass1': {'path': id_path + 'glass1.csv', 'categorical_cols': (), 'class_col': 9},
    'glass2': {'path': id_path + 'glass2.csv', 'categorical_cols': (), 'class_col': 9},
    'glass4': {'path': id_path + 'glass4.csv', 'categorical_cols': (), 'class_col': 9},
    'glass5': {'path': id_path + 'glass5.csv', 'categorical_cols': (), 'class_col': 9},
    'glass6': {'path': id_path + 'glass6.csv', 'categorical_cols': (), 'class_col': 9},

    'pima': {'path': id_path + 'pima.csv', 'categorical_cols': (), 'class_col': 8},

    # 'vowel0': {'path': id_path + 'vowel0.csv', 'categorical_cols': (0, 1, 2), 'class_col': 13}, # ADASYN fails @ fold 3

    'yeast3': {'path': id_path + 'yeast3.csv', 'categorical_cols': (), 'class_col': 8},
    'yeast4': {'path': id_path + 'yeast4.csv', 'categorical_cols': (), 'class_col': 8},
    'yeast5': {'path': id_path + 'yeast5.csv', 'categorical_cols': (), 'class_col': 8},
    'yeast6': {'path': id_path + 'yeast6.csv', 'categorical_cols': (), 'class_col': 8},
}

datasets_keel = {
    # 'vowel0': {'path': id_path + 'vowel0.csv', 'categorical_cols': (0, 1, 2), 'class_col': 13}, # ADASYN fails @ fold3
    'cleveland-0_vs_4': {'path': id_path + 'cleveland-0_vs_4.csv', 'categorical_cols': (), 'class_col': 13}, # TVAE
    # 'dermatology-6': {'path': id_path + 'dermatology-6.csv', 'class_col': 34}, GCOP achieves bacc=1

    'ecoli-0-1_vs_2-3-5': {'path': id_path + 'ecoli-0-1_vs_2-3-5.csv', 'categorical_cols': (), 'class_col': 7},
    'ecoli-0-1_vs_5': {'path': id_path + 'ecoli-0-1_vs_5.csv', 'categorical_cols': (), 'class_col': 6},
    'ecoli-0-1-4-6_vs_5': {'path': id_path + 'ecoli-0-1-4-6_vs_5.csv', 'categorical_cols': (), 'class_col': 6},
    'ecoli-0-1-4-7_vs_2-3-5-6': {'path': id_path + 'ecoli-0-1-4-7_vs_2-3-5-6.csv', 'categorical_cols': (),
                                 'class_col': 7},
    'ecoli-0-1-4-7_vs_5-6': {'path': id_path + 'ecoli-0-1-4-7_vs_5-6.csv', 'categorical_cols': (), 'class_col': 6},
    'ecoli-0-2-3-4_vs_5': {'path': id_path + 'ecoli-0-2-3-4_vs_5.csv', 'categorical_cols': (), 'class_col': 7},
    'ecoli-0-2-6-7_vs_3-5': {'path': id_path + 'ecoli-0-2-6-7_vs_3-5.csv', 'categorical_cols': (), 'class_col': 7},
    'ecoli-0-3-4_vs_5': {'path': id_path + 'ecoli-0-3-4_vs_5.csv', 'categorical_cols': (), 'class_col': 7},
    'ecoli-0-3-4-6_vs_5': {'path': id_path + 'ecoli-0-3-4-6_vs_5.csv', 'categorical_cols': (), 'class_col': 7},
    'ecoli-0-3-4-7_vs_5-6': {'path': id_path + 'ecoli-0-3-4-7_vs_5-6.csv', 'categorical_cols': (), 'class_col': 7},
    'ecoli-0-4-6_vs_5': {'path': id_path + 'ecoli-0-4-6_vs_5.csv', 'categorical_cols': (), 'class_col': 6},
    'ecoli-0-6-7_vs_3-5': {'path': id_path + 'ecoli-0-6-7_vs_3-5.csv', 'categorical_cols': (), 'class_col': 7},
    'ecoli-0-6-7_vs_5': {'path': id_path + 'ecoli-0-6-7_vs_5.csv', 'categorical_cols': (), 'class_col': 6},
    # 'ecoli1': {'path': id_path + 'ecoli1.csv', 'categorical_cols': (), 'class_col': 7}, # Multi-class ?
    # 'ecoli2': {'path': id_path + 'ecoli2.csv', 'categorical_cols': (), 'class_col': 7}, # Multi-class ?
    'ecoli3': {'path': id_path + 'ecoli3.csv', 'categorical_cols': (), 'class_col': 7},
    # 'ecoli4': {'path': id_path + 'ecoli4.csv', 'categorical_cols': (), 'class_col': 7}, # Multi-class ?

    'glass-0-1-2-3_vs_4-5-6': {'path': id_path + 'glass-0-1-2-3_vs_4-5-6.csv', 'categorical_cols': (), 'class_col': 9},
    'glass-0-1-4-6_vs_2': {'path': id_path + 'glass-0-1-4-6_vs_2.csv', 'categorical_cols': (), 'class_col': 9},
    'glass-0-1-5_vs_2': {'path': id_path + 'glass-0-1-5_vs_2.csv', 'categorical_cols': (), 'class_col': 9},
    'glass-0-1-6_vs_2': {'path': id_path + 'glass-0-1-6_vs_2.csv', 'categorical_cols': (), 'class_col': 9},
    'glass-0-4_vs_5': {'path': id_path + 'glass-0-4_vs_5.csv', 'categorical_cols': (), 'class_col': 9},
    'glass-0-6_vs_5': {'path': id_path + 'glass-0-6_vs_5.csv', 'categorical_cols': (), 'class_col': 9},
    'glass1': {'path': id_path + 'glass1.csv', 'categorical_cols': (), 'class_col': 9},
    'glass2': {'path': id_path + 'glass2.csv', 'categorical_cols': (), 'class_col': 9},
    'glass4': {'path': id_path + 'glass4.csv', 'categorical_cols': (), 'class_col': 9},
    'glass5': {'path': id_path + 'glass5.csv', 'categorical_cols': (), 'class_col': 9},
    'glass6': {'path': id_path + 'glass6.csv', 'categorical_cols': (), 'class_col': 9},

    'led7digit-0-2-4-5-6-7-8-9_vs_1': {'path': id_path + 'led7digit-0-2-4-5-6-7-8-9_vs_1.csv',
                                       'categorical_cols': (), 'class_col': 7},
    # 'new-thyroid1': {'path': id_path + 'new-thyroid1.csv', 'categorical_cols': (), 'class_col': 5}, # Bal.acc.=1

    'pima': {'path': id_path + 'pima.csv', 'categorical_cols': (), 'class_col': 8},
    # 'segment0': {'path': id_path + 'segment0.csv', 'categorical_cols': (), 'class_col': 19},  # Bal.acc.=1

    # 'flare': {'path': id_path + 'flare-F.csv', 'categorical_cols': (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10), 'class_col': 11},

    'winequality-red-4': {'path': id_path + 'winequality-red-4.csv', 'categorical_cols': (), 'class_col': 11},
    'winequality-red-8_vs_6': {'path': id_path + 'winequality-red-8_vs_6.csv', 'categorical_cols': (), 'class_col': 11},
    'winequality-red-3_vs_5': {'path': id_path + 'winequality-red-3_vs_5.csv', 'categorical_cols': (), 'class_col': 11},
    'winequality-red-8_vs_6-7': {'path': id_path + 'winequality-red-8_vs_6-7.csv', 'categorical_cols': (),
                                 'class_col': 11},
    'winequality-white-3_vs_7': {'path': id_path + 'winequality-white-3_vs_7.csv', 'categorical_cols': (),
                                 'class_col': 11},
    'winequality-white-3-9_vs_5': {'path': id_path + 'winequality-white-3-9_vs_5.csv', 'categorical_cols': (),
                                   'class_col': 11},
    # 'winequality-white-9_vs_4': {'path': id_path + 'winequality-white-9_vs_4.csv', 'categorical_cols': (),
    #                             'class_col': 11}, # smote breaks

    'yeast-0-2-5-7-9_vs_3-6-8': {'path': id_path + 'yeast-0-2-5-7-9_vs_3-6-8.csv', 'categorical_cols': (),
                                 'class_col': 8},
    'yeast-0-2-5-6_vs_3-7-8-9': {'path': id_path + 'yeast-0-2-5-6_vs_3-7-8-9.csv', 'categorical_cols': (),
                                 'class_col': 8},
    'yeast-0-3-5-9_vs_7-8': {'path': id_path + 'yeast-0-3-5-9_vs_7-8.csv', 'categorical_cols': (), 'class_col': 8},
    'yeast-0-5-6-7-9_vs_4': {'path': id_path + 'yeast-0-5-6-7-9_vs_4.csv', 'categorical_cols': (), 'class_col': 8},
    'yeast1': {'path': id_path + 'yeast1.csv', 'categorical_cols': (), 'class_col': 8},
    'yeast-1_vs_7': {'path': id_path + 'yeast-1_vs_7.csv', 'categorical_cols': (), 'class_col': 7},
    'yeast-1-2-8-9_vs_7': {'path': id_path + 'yeast-1-2-8-9_vs_7.csv', 'categorical_cols': (), 'class_col': 8},
    'yeast-1-4-5-8_vs_7': {'path': id_path + 'yeast-1-4-5-8_vs_7.csv', 'categorical_cols': (), 'class_col': 8},
    'yeast-2_vs_8': {'path': id_path + 'yeast-2_vs_8.csv', 'categorical_cols': (), 'class_col': 8},
    'yeast3': {'path': id_path + 'yeast3.csv', 'categorical_cols': (), 'class_col': 8},
    'yeast4': {'path': id_path + 'yeast4.csv', 'categorical_cols': (), 'class_col': 8},
    'yeast5': {'path': id_path + 'yeast5.csv', 'categorical_cols': (), 'class_col': 8},
    'yeast6': {'path': id_path + 'yeast6.csv', 'categorical_cols': (), 'class_col': 8},

    # 'abalone9-18': {'path': id_path + 'abalone9-18.csv', 'categorical_cols': (), 'class_col': 7},
    # 'CAMEL-1.2': {'path': id_path + 'camel-1.2.csv', 'categorical_cols': (), 'class_col': 20},
    # 'CC': {'path': id_path + 'creditcarddefault.csv', 'categorical_cols': (), 'class_col': 24},
}

# Datasets on software defect detection
sdd_datasets = {
    # 'Synthetic': None,
    'AR1': {'path': sd_path + 'ar1.csv', 'categorical_cols': (), 'class_col': 29},
    # 'AR3': {'path': sd_path + 'ar3.csv', 'categorical_cols': (), 'class_col': 29},
    # 'AR4': {'path': sd_path + 'ar4.csv', 'categorical_cols': (), 'class_col': 29},
    'CM1': {'path': sd_path + 'cm1.csv', 'categorical_cols': (), 'class_col': 21},
    # 'JM1': {'path': sd_path + 'jm1.csv', 'categorical_cols': (), 'class_col': 21},

    'KC1': {'path': sd_path + 'kc1.csv', 'categorical_cols': (), 'class_col': 21},
    'KC2': {'path': sd_path + 'kc2.csv', 'categorical_cols': (), 'class_col': 21},
    'KC3': {'path': sd_path + 'kc3.csv', 'categorical_cols': (), 'class_col': 39},
    # 'MC1': {'path': sd_path + 'mc1.csv', 'categorical_cols': (), 'class_col': 38},
    # 'MC2': {'path': sd_path + 'mc2.csv', 'categorical_cols': (), 'class_col': 39},
    # 'MW1': {'path': sd_path + 'mw1.csv', 'categorical_cols': (), 'class_col': 37},
    'PC1': {'path': sd_path + 'pc1.csv', 'categorical_cols': (), 'class_col': 21},
    # 'PC2': {'path': sd_path + 'pc2.csv', 'categorical_cols': (), 'class_col': 36},
    'PC3': {'path': sd_path + 'pc3.csv', 'categorical_cols': (), 'class_col': 37},
    'PC4': {'path': sd_path + 'pc4.csv', 'categorical_cols': (), 'class_col': 37},

    # 'ANT-1.3': {'path': sd_path + 'ant-1.3.csv', 'categorical_cols': (), 'class_col': 20},
    # 'ANT-1.4': {'path': sd_path + 'ant-1.4.csv', 'categorical_cols': (), 'class_col': 20},
    # 'ANT-1.5': {'path': sd_path + 'ant-1.5.csv', 'categorical_cols': (), 'class_col': 20},
    # 'ANT-1.6': {'path': sd_path + 'ant-1.6.csv', 'categorical_cols': (), 'class_col': 20},
    # 'ANT-1.7': {'path': sd_path + 'ant-1.7.csv', 'categorical_cols': (), 'class_col': 20},
    # 'CAMEL-1.0': {'path': sd_path + 'camel-1.0.csv', 'categorical_cols': (), 'class_col': 20},
    'CAMEL-1.2': {'path': sd_path + 'camel-1.2.csv', 'categorical_cols': (), 'class_col': 20},
    'CAMEL-1.4': {'path': sd_path + 'camel-1.4.csv', 'categorical_cols': (), 'class_col': 20},
    'CAMEL-1.6': {'path': sd_path + 'camel-1.6.csv', 'categorical_cols': (), 'class_col': 20},
    'IVY-1.1': {'path': sd_path + 'ivy-1.1.csv', 'categorical_cols': (), 'class_col': 20},
    # 'IVY-1.4': {'path': sd_path + 'ivy-1.4.csv', 'categorical_cols': (), 'class_col': 20},
    'IVY-2.0': {'path': sd_path + 'ivy-2.0.csv', 'categorical_cols': (), 'class_col': 20},
    # 'JEDIT-3.2': {'path': sd_path + 'jedit-3.2.csv', 'categorical_cols': (), 'class_col': 20},
    'JEDIT-4.0': {'path': sd_path + 'jedit-4.0.csv', 'categorical_cols': (), 'class_col': 20},
    'JEDIT-4.1': {'path': sd_path + 'jedit-4.1.csv', 'categorical_cols': (), 'class_col': 20},
    'JEDIT-4.2': {'path': sd_path + 'jedit-4.2.csv', 'categorical_cols': (), 'class_col': 20},
    # 'JEDIT-4.3': {'path': sd_path + 'jedit-4.3.csv', 'categorical_cols': (), 'class_col': 20},
    'LOG4J-1.0': {'path': sd_path + 'log4j-1.0.csv', 'categorical_cols': (), 'class_col': 20},
    'LOG4J-1.1': {'path': sd_path + 'log4j-1.1.csv', 'categorical_cols': (), 'class_col': 20},
    'LOG4J-1.2': {'path': sd_path + 'log4j-1.2.csv', 'categorical_cols': (), 'class_col': 20},
    # 'LUCENE-2.0': {'path': sd_path + 'lucene-2.0.csv', 'categorical_cols': (), 'class_col': 20},
    'LUCE-2.2': {'path': sd_path + 'lucene-2.2.csv', 'categorical_cols': (), 'class_col': 20},
    'LUCE-2.4': {'path': sd_path + 'lucene-2.4.csv', 'categorical_cols': (), 'class_col': 20},
    'POI-1.5': {'path': sd_path + 'poi-1.5.csv', 'categorical_cols': (), 'class_col': 20},
    'POI-2.0': {'path': sd_path + 'poi-2.0.csv', 'categorical_cols': (), 'class_col': 20},
    # 'POI-2.5': {'path': sd_path + 'poi-2.5.csv', 'categorical_cols': (), 'class_col': 20},
    # 'POI-3.0': 'path': sd_path + 'poi-3.0.csv', 'categorical_cols': (), 'class_col': 20},
    # 'SYN1.0': {'path': sd_path + 'synapse-1.0.csv', 'categorical_cols': (), 'class_col': 20},
    # 'SYN1.1': {'path': sd_path + 'synapse-1.1.csv', 'categorical_cols': (), 'class_col': 20},
    # 'SYN1.2': {'path': sd_path + 'synapse-1.2.csv', 'categorical_cols': (), 'class_col': 20},
    'VL14': {'path': sd_path + 'velocity-1.4.csv', 'categorical_cols': (), 'class_col': 20},
    # 'VL15': {'path': sd_path + 'velocity-1.5.csv', 'categorical_cols': (), 'class_col': 20},
    'VL16': {'path': sd_path + 'velocity-1.6.csv', 'categorical_cols': (), 'class_col': 20},
    'XER-1.2': {'path': sd_path + 'xerces-1.2.csv', 'categorical_cols': (), 'class_col': 20},
    'XER-1.3': {'path': sd_path + 'xerces-1.3.csv', 'categorical_cols': (), 'class_col': 20},
    'XER-1.4': {'path': sd_path + 'xerces-1.4.csv', 'categorical_cols': (), 'class_col': 20}
}

# datasets_cat = {'vowel': {'path': id_path + 'vowel0_test.csv', 'categorical_cols': (0, 1, 2), 'class_col': 13} }
# eval_methods.test_model('CTDGAN', datasets_cat['vowel'], seed)

eval_methods.eval_resampling(datasets=datasets, transformer='standardizer', num_folds=5, random_state=seed)
# eval_methods.eval_detectability(datasets=datasets_imb, transformer='standardizer', num_folds=5, random_state=seed)

# eval_methods.eval_ml_efficacy(datasets, num_threads, seed)

# Experiments performed in Information Sciences 2024 paper
# eval_methods.eval_oversampling_efficacy(datasets_imb, num_threads, seed)
