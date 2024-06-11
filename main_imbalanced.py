import os
import sys

import numpy as np

import DeepCoreML.eval as evaluation_methods

num_threads = 1
os.environ['OMP_NUM_THREADS'] = str(num_threads)
np.set_printoptions(linewidth=400, threshold=sys.maxsize)

seed = 13

# path = '../../../../datasets/Imbalanced/keel/'
# path = '../../../../datasets/soft_defect/'
path = '/media/leo/7CE54B377BB9B18B/datasets/Imbalanced/keel/'
# path = '/media/leo/7CE54B377BB9B18B/datasets/soft_defect/'

datasets_imb = {
    'cleveland-0_vs_4': {'path': path + 'cleveland-0_vs_4.csv', 'features_cols': range(0, 13), 'class_col': 13},
    'dermatology-6': {'path': path + 'dermatology-6.csv', 'features_cols': range(0, 34), 'class_col': 34},

    'ecoli-0-1_vs_2-3-5': {'path': path + 'ecoli-0-1_vs_2-3-5.csv', 'features_cols': range(0, 7), 'class_col': 7},
    'ecoli-0-1_vs_5': {'path': path + 'ecoli-0-1_vs_5.csv', 'features_cols': range(0, 6), 'class_col': 6},
    ### 'ecoli-0-1-3-7_vs_2-6': {'path': path + 'ecoli-0-1-3-7_vs_2-6.csv', 'features_cols': range(0, 7), 'class_col': 7}, # Breaks
    'ecoli-0-1-4-6_vs_5': {'path': path + 'ecoli-0-1-4-6_vs_5.csv', 'features_cols': range(0, 6), 'class_col': 6},
    'ecoli-0-1-4-7_vs_2-3-5-6': {'path': path + 'ecoli-0-1-4-7_vs_2-3-5-6.csv', 'features_cols': range(0, 7),
                                 'class_col': 7},
    'ecoli-0-1-4-7_vs_5-6': {'path': path + 'ecoli-0-1-4-7_vs_5-6.csv', 'features_cols': range(0, 6), 'class_col': 6},
    'ecoli-0-2-3-4_vs_5': {'path': path + 'ecoli-0-2-3-4_vs_5.csv', 'features_cols': range(0, 7), 'class_col': 7},
    'ecoli-0-2-6-7_vs_3-5': {'path': path + 'ecoli-0-2-6-7_vs_3-5.csv', 'features_cols': range(0, 7), 'class_col': 7},
    'ecoli-0-3-4_vs_5': {'path': path + 'ecoli-0-3-4_vs_5.csv', 'features_cols': range(0, 7), 'class_col': 7},
    'ecoli-0-3-4-6_vs_5': {'path': path + 'ecoli-0-3-4-6_vs_5.csv', 'features_cols': range(0, 7), 'class_col': 7},
    'ecoli-0-3-4-7_vs_5-6': {'path': path + 'ecoli-0-3-4-7_vs_5-6.csv', 'features_cols': range(0, 7), 'class_col': 7},
    'ecoli-0-4-6_vs_5': {'path': path + 'ecoli-0-4-6_vs_5.csv', 'features_cols': range(0, 6), 'class_col': 6},
    'ecoli-0-6-7_vs_3-5': {'path': path + 'ecoli-0-6-7_vs_3-5.csv', 'features_cols': range(0, 7), 'class_col': 7},
    'ecoli-0-6-7_vs_5': {'path': path + 'ecoli-0-6-7_vs_5.csv', 'features_cols': range(0, 6), 'class_col': 6},
    ### 'ecoli1': {'path': path + 'ecoli1.csv', 'features_cols': range(0, 7), 'class_col': 7}, # Multi-class ?
    ### 'ecoli2': {'path': path + 'ecoli2.csv', 'features_cols': range(0, 7), 'class_col': 7}, # Multi-class ?
    'ecoli3': {'path': path + 'ecoli3.csv', 'features_cols': range(0, 7), 'class_col': 7},
    ###'ecoli4': {'path': path + 'ecoli4.csv', 'features_cols': range(0, 7), 'class_col': 7},# Multi-class ? # Multi-class?

    'glass-0-1-2-3_vs_4-5-6': {'path': path + 'glass-0-1-2-3_vs_4-5-6.csv', 'features_cols': range(0, 9),
                               'class_col': 9},
    'glass-0-1-4-6_vs_2': {'path': path + 'glass-0-1-4-6_vs_2.csv', 'features_cols': range(0, 9), 'class_col': 9},
    'glass-0-1-5_vs_2': {'path': path + 'glass-0-1-5_vs_2.csv', 'features_cols': range(0, 9), 'class_col': 9},
    'glass-0-1-6_vs_2': {'path': path + 'glass-0-1-6_vs_2.csv', 'features_cols': range(0, 9), 'class_col': 9},
    'glass-0-4_vs_5': {'path': path + 'glass-0-4_vs_5.csv', 'features_cols': range(0, 9), 'class_col': 9},
    'glass-0-6_vs_5': {'path': path + 'glass-0-6_vs_5.csv', 'features_cols': range(0, 9), 'class_col': 9},
    'glass1': {'path': path + 'glass1.csv', 'features_cols': range(0, 9), 'class_col': 9},
    'glass2': {'path': path + 'glass2.csv', 'features_cols': range(0, 9), 'class_col': 9},
    'glass4': {'path': path + 'glass4.csv', 'features_cols': range(0, 9), 'class_col': 9},
    'glass5': {'path': path + 'glass5.csv', 'features_cols': range(0, 9), 'class_col': 9},
    'glass6': {'path': path + 'glass6.csv', 'features_cols': range(0, 9), 'class_col': 9},

    'led7digit-0-2-4-5-6-7-8-9_vs_1': {'path': path + 'led7digit-0-2-4-5-6-7-8-9_vs_1.csv',
                                       'features_cols': range(0, 7), 'class_col': 7},
    'new-thyroid1': {'path': path + 'new-thyroid1.csv', 'features_cols': range(0, 5), 'class_col': 5},
    ###'new-thyroid2': {'path': path + 'new-thyroid2.csv', 'features_cols': range(0, 5), 'class_col': 5}, # File Does not exist!
    'pima': {'path': path + 'pima.csv', 'features_cols': range(0, 8), 'class_col': 8},
    'segment0': {'path': path + 'segment0.csv', 'features_cols': range(0, 19), 'class_col': 19},
    ### 'vowel0': {'path': path + 'vowel0.csv', 'features_cols': range(0, 13), 'class_col': 13}, # ADASYN FAILS AT FOLD 3

    'winequality-red-4': {'path': path + 'winequality-red-4.csv', 'features_cols': range(0, 11), 'class_col': 11},
    'winequality-red-8_vs_6': {'path': path + 'winequality-red-8_vs_6.csv', 'features_cols': range(0, 11),
                               'class_col': 11},
    'winequality-red-3_vs_5': {'path': path + 'winequality-red-3_vs_5.csv', 'features_cols': range(0, 11),
                               'class_col': 11},
    'winequality-red-8_vs_6-7': {'path': path + 'winequality-red-8_vs_6-7.csv', 'features_cols': range(0, 11),
                                 'class_col': 11},
    'winequality-white-3_vs_7': {'path': path + 'winequality-white-3_vs_7.csv', 'features_cols': range(0, 11),
                                 'class_col': 11},
    'winequality-white-3-9_vs_5': {'path': path + 'winequality-white-3-9_vs_5.csv', 'features_cols': range(0, 11),
                                   'class_col': 11},
    ### 'winequality-white-9_vs_4': {'path': path + 'winequality-white-9_vs_4.csv', 'features_cols': range(0, 11),
    ###                             'class_col': 11}, # smote breaks

    'yeast-0-2-5-7-9_vs_3-6-8': {'path': path + 'yeast-0-2-5-7-9_vs_3-6-8.csv', 'features_cols': range(0, 8),
                                 'class_col': 8},
    'yeast-0-2-5-6_vs_3-7-8-9': {'path': path + 'yeast-0-2-5-6_vs_3-7-8-9.csv', 'features_cols': range(0, 8),
                                 'class_col': 8},
    'yeast-0-3-5-9_vs_7-8': {'path': path + 'yeast-0-3-5-9_vs_7-8.csv', 'features_cols': range(0, 8), 'class_col': 8},
    'yeast-0-5-6-7-9_vs_4': {'path': path + 'yeast-0-5-6-7-9_vs_4.csv', 'features_cols': range(0, 8), 'class_col': 8},
    'yeast1': {'path': path + 'yeast1.csv', 'features_cols': range(0, 8), 'class_col': 8},
    'yeast-1_vs_7': {'path': path + 'yeast-1_vs_7.csv', 'features_cols': range(0, 7), 'class_col': 7},
    'yeast-1-2-8-9_vs_7': {'path': path + 'yeast-1-2-8-9_vs_7.csv', 'features_cols': range(0, 8), 'class_col': 8},
    'yeast-1-4-5-8_vs_7': {'path': path + 'yeast-1-4-5-8_vs_7.csv', 'features_cols': range(0, 8), 'class_col': 8},
    'yeast-2_vs_8': {'path': path + 'yeast-2_vs_8.csv', 'features_cols': range(0, 8), 'class_col': 8},
    'yeast3': {'path': path + 'yeast3.csv', 'features_cols': range(0, 8), 'class_col': 8},
    'yeast4': {'path': path + 'yeast3.csv', 'features_cols': range(0, 8), 'class_col': 8},
    'yeast5': {'path': path + 'yeast5.csv', 'features_cols': range(0, 8), 'class_col': 8},
    'yeast6': {'path': path + 'yeast6.csv', 'features_cols': range(0, 8), 'class_col': 8},

    # 'abalone9-18': {'path': path + 'abalone9-18.csv', 'features_cols': range(0, 6), 'class_col': 7},
    # 'CAMEL-1.2': {'path': path + 'camel-1.2.csv', 'features_cols': range(0, 20), 'class_col': 20},
    # 'CC': {'path': path + 'creditcarddefault.csv', 'features_cols': range(1, 24), 'class_col': 24},
}

# Datasets on software defect detection
sdd_datasets = {
    # 'Synthetic': None,
    # 'AR1': {'path': path + 'ar1.csv', 'features_cols': range(0, 29), 'class_col': 29},
    # 'AR3': {'path': path + 'ar3.csv', 'features_cols': range(0, 29), 'class_col': 29},
    # 'AR4': {'path': path + 'ar4.csv', 'features_cols': range(0, 29), 'class_col': 29},
    'CM1': {'path': path + 'cm1.csv', 'features_cols': range(0, 21), 'class_col': 21},
    # 'JM1': {'path': path + 'jm1.csv', 'features_cols': range(0, 21), 'class_col': 21},

    'KC1': {'path': path + 'kc1.csv', 'features_cols': range(0, 21), 'class_col': 21},
    'KC2': {'path': path + 'kc2.csv', 'features_cols': range(0, 21), 'class_col': 21},
    'KC3': {'path': path + 'kc3.csv', 'features_cols': range(0, 39), 'class_col': 39},
    # 'MC1': {'path': path + 'mc1.csv', 'features_cols': range(0, 38), 'class_col': 38},
    # 'MC2': {'path': path + 'mc2.csv', 'features_cols': range(0, 39), 'class_col': 39},
    # 'MW1': {'path': path + 'mw1.csv', 'features_cols': range(0, 37), 'class_col': 37},
    'PC1': {'path': path + 'pc1.csv', 'features_cols': range(0, 21), 'class_col': 21},
    # 'PC2': {'path': path + 'pc2.csv', 'features_cols': range(0, 36), 'class_col': 36},
    'PC3': {'path': path + 'pc3.csv', 'features_cols': range(0, 37), 'class_col': 37},
    'PC4': {'path': path + 'pc4.csv', 'features_cols': range(0, 37), 'class_col': 37},

    # 'ANT-1.3': {'path': path + 'ant-1.3.csv', 'features_cols': range(0, 20), 'class_col': 20},
    # 'ANT-1.4': {'path': path + 'ant-1.4.csv', 'features_cols': range(0, 20), 'class_col': 20},
    # 'ANT-1.5': {'path': path + 'ant-1.5.csv', 'features_cols': range(0, 20), 'class_col': 20},
    # 'ANT-1.6': {'path': path + 'ant-1.6.csv', 'features_cols': range(0, 20), 'class_col': 20},
    # 'ANT-1.7': {'path': path + 'ant-1.7.csv', 'features_cols': range(0, 20), 'class_col': 20},
    # 'CAMEL-1.0': {'path': path + 'camel-1.0.csv', 'features_cols': range(0, 20), 'class_col': 20},
    'CAMEL-1.2': {'path': path + 'camel-1.2.csv', 'features_cols': range(0, 20), 'class_col': 20},
    'CAMEL-1.4': {'path': path + 'camel-1.4.csv', 'features_cols': range(0, 20), 'class_col': 20},
    'CAMEL-1.6': {'path': path + 'camel-1.6.csv', 'features_cols': range(0, 20), 'class_col': 20},
    'IVY-1.1': {'path': path + 'ivy-1.1.csv', 'features_cols': range(0, 20), 'class_col': 20},
    # 'IVY-1.4': {'path': path + 'ivy-1.4.csv', 'features_cols': range(0, 20), 'class_col': 20},
    'IVY-2.0': {'path': path + 'ivy-2.0.csv', 'features_cols': range(0, 20), 'class_col': 20},
    # 'JEDIT-3.2': {'path': path + 'jedit-3.2.csv', 'features_cols': range(0, 20), 'class_col': 20},
    'JEDIT-4.0': {'path': path + 'jedit-4.0.csv', 'features_cols': range(0, 20), 'class_col': 20},
    'JEDIT-4.1': {'path': path + 'jedit-4.1.csv', 'features_cols': range(0, 20), 'class_col': 20},
    'JEDIT-4.2': {'path': path + 'jedit-4.2.csv', 'features_cols': range(0, 20), 'class_col': 20},
    # 'JEDIT-4.3': {'path': path + 'jedit-4.3.csv', 'features_cols': range(0, 20), 'class_col': 20},
    'LOG4J-1.0': {'path': path + 'log4j-1.0.csv', 'features_cols': range(0, 20), 'class_col': 20},
    'LOG4J-1.1': {'path': path + 'log4j-1.1.csv', 'features_cols': range(0, 20), 'class_col': 20},
    'LOG4J-1.2': {'path': path + 'log4j-1.2.csv', 'features_cols': range(0, 20), 'class_col': 20},
    # 'LUCENE-2.0': {'path': path + 'lucene-2.0.csv', 'features_cols': range(0, 20), 'class_col': 20},
    'LUCE-2.2': {'path': path + 'lucene-2.2.csv', 'features_cols': range(0, 20), 'class_col': 20},
    'LUCE-2.4': {'path': path + 'lucene-2.4.csv', 'features_cols': range(0, 20), 'class_col': 20},
    'POI-1.5': {'path': path + 'poi-1.5.csv', 'features_cols': range(0, 20), 'class_col': 20},
    'POI-2.0': {'path': path + 'poi-2.0.csv', 'features_cols': range(0, 20), 'class_col': 20},
    # 'POI-2.5': {'path': path + 'poi-2.5.csv', 'features_cols': range(0, 20), 'class_col': 20},
    # 'POI-3.0': 'path': path + 'poi-3.0.csv', 'features_cols': range(0, 20), 'class_col': 20},
    # 'SYN1.0': {'path': path + 'synapse-1.0.csv', 'features_cols': range(0, 20), 'class_col': 20},
    # 'SYN1.1': {'path': path + 'synapse-1.1.csv', 'features_cols': range(0, 20), 'class_col': 20},
    # 'SYN1.2': {'path': path + 'synapse-1.2.csv', 'features_cols': range(0, 20), 'class_col': 20},
    'VL14': {'path': path + 'velocity-1.4.csv', 'features_cols': range(0, 20), 'class_col': 20},
    # 'VL15': {'path': path + 'velocity-1.5.csv', 'features_cols': range(0, 20), 'class_col': 20},
    'VL16': {'path': path + 'velocity-1.6.csv', 'features_cols': range(0, 20), 'class_col': 20},
    'XER-1.2': {'path': path + 'xerces-1.2.csv', 'features_cols': range(0, 20), 'class_col': 20},
    'XER-1.3': {'path': path + 'xerces-1.3.csv', 'features_cols': range(0, 20), 'class_col': 20},
    'XER-1.4': {'path': path + 'xerces-1.4.csv', 'features_cols': range(0, 20), 'class_col': 20}
}


# evaluation_methods.test_model('GAAN', datasets['AR1'], seed)

evaluation_methods.eval_resampling(datasets=datasets_imb, transformer='standardizer', num_folds=5, random_state=seed)
# evaluation_methods.eval_detectability(datasets=datasets_imb, transformer='standardizer', num_folds=5, random_state=seed)

# evaluation_methods.eval_ml_efficacy(datasets, num_threads, seed)

# Experiments performed in Information Sciences 2024 paper
# evaluation_methods.eval_oversampling_efficacy(datasets_imb, num_threads, seed)
