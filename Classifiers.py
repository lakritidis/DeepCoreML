from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb


class BaseClassifier:
    def __init__(self, name, short_name, model, **kwargs):
        self.name_ = name
        self.short_name_ = short_name
        self.model_ = model
        super().__init__(**kwargs)

    def fit(self, x, y):
        self.model_.fit(x, y)

    def predict(self, y):
        return self.model_.predict(y)


class Classifiers:
    def __init__(self, random_state=0):
        self.num_classifiers_ = 0

        self.models_ = (
            BaseClassifier("Logistic Regression", "LR",
                            LogisticRegression(max_iter=300, random_state=random_state)),
            BaseClassifier("Support Vector Machine", "SVM",
                           SVC(kernel='rbf', C=1, random_state=random_state)),
            BaseClassifier("Decision Tree", "DT",
                           DecisionTreeClassifier(criterion='gini', max_depth=None,
                                                  max_features=None, random_state=random_state)),
            BaseClassifier("XGBoost", "XGBoost", xgb.XGBClassifier()),
            BaseClassifier("Random Forest", "RF",
                           RandomForestClassifier(n_estimators=50, criterion='gini', max_depth=None,
                                                  max_features='sqrt', n_jobs=1, random_state=random_state)),
            BaseClassifier("Multilayer Perceptron", "MLP",
                           MLPClassifier(activation='relu', hidden_layer_sizes=(128, 128), solver='adam', max_iter=300,
                                         random_state=random_state))
        )
        self.num_classifiers_ = len(self.models_)
