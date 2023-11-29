import DataTools

from TextDatasets import TextDataset
import TextVectorizers
import DimensionalityReducers

from imblearn.pipeline import make_pipeline

from sklearn.ensemble import RandomForestClassifier

seed = 1
DataTools.set_random_states(seed)

dataset_path = 'C:/Users/Leo/PycharmProjects/datasets/text_sentiment/'

datasets = {
            'Twitter_US_Air': (dataset_path + 'Twitter_US_Airline_Sentiment.csv', range(10, 11), 1),
            'Twitter_Fin_Sent': (dataset_path + 'Twitter_Financial_Sentiment.csv', range(0, 1), 1)
            }

dataset = datasets['Twitter_US_Air']

# Example of Vectorization/Dimensionality Reduction
text_ds = TextDataset(random_state=seed)
text_ds.load_from_csv(path=dataset[0], feature_cols=dataset[1], class_col=dataset[2])
text_ds.preprocess()

max_vector_length = 10
# vectorization_method = TextVectorizers.bertVectorizer()
vectorization_method = TextVectorizers.tfidfVectorizer(latent_dimensionality=max_vector_length)
# vectorization_method = TextVectorizers.word2vecVectorizer(latent_dimensionality=max_vector_length)

# reduction_method = DimensionalityReducers.PCAReducer(n_components=64)
# reduction_method = DimensionalityReducers.TSVDReducer(n_components=200, random_state=seed, algorithm='arpack')
reduction_method = DimensionalityReducers.AutoencoderReducer(encoder=(512,), decoder=(512,), n_components=200,
                                                            input_dimensionality=max_vector_length, training_epochs=10)
# reduction_method = DimensionalityReducers.FeatureSelector(n_components=64, random_state=seed)
# reduction_method = None

clf = RandomForestClassifier(n_estimators=100, criterion='gini', max_depth=None, max_features='sqrt',
                             random_state=seed, n_jobs=8)

pipe_line = make_pipeline(vectorization_method, reduction_method, clf)

results_list = []
cv_results = text_ds.cv_pipeline(pipeline=pipe_line, num_folds=5, num_threads=1, results_list=results_list,
                                 classifier_str="Random Forest", sampler_str="tf-idf/pca", order=1)

print(cv_results)
