
class DataTransformer:
    def __init__(self, feature_encoder=None, class_encoder=None, random_state=0, **kwargs):
        self.seed_ = random_state
        self.feature_encoder_ = feature_encoder
        self.class_encoder_ = class_encoder
        self.transform_time_ = 0
        super().__init__(**kwargs)


class FeatureTransformer:
    def __init__(self, feature_encoder):
        self.feature_encoder_ = feature_encoder

    def fit(self, x, y=None):
        self.feature_encoder_.fit(x, y)

    def transform(self, x):
        return self.feature_encoder_.transform(x)

    def fit_transform(self, x, y=None):
        return self.feature_encoder_.fit_transform(x, y)


class ClassTransformer:
    def __init__(self, class_encoder, random_state=0):
        self.class_encoder_ = class_encoder

    def fit(self, x):
        self.class_encoder_.fit(x)

    def transform(self, x):
        return self.class_encoder_.transform(x)

    def fit_transform(self, x):
        return self.class_encoder_.fit_transform(x)
