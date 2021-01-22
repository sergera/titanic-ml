import pandas as pd
import numpy as np

from presci.analyzer import Analyzer
from presci.transformer import Transformer
from model.model import Model

PATH_TO_TRAINING_DATASET = "train.csv"
PATH_TO_TEST_DATASET = "test.csv"
TARGET = 'Survived'

RARE_THRESHOLD = 0.01
OUTLIER_THRESHOLD = 3

class Pipeline():
    def __init__(self, select_features):
        self.select_features = select_features

        self.transformer = Transformer()
        self.model = Model(max_iter=10000)
    
    def fit(self, data):
        # select features
        data = self.select_features(data)

        # get meta data
        analyzer = Analyzer(        
            data, 
            TARGET,
            unique_threshold=0.7,
            rare_threshold=RARE_THRESHOLD,
            skewness_threshold=0.3,
        )
        self.meta_data = analyzer.get_meta_data()

        # replace rare labels
        cat_vars = data.loc[:,self.meta_data["categorical_features"]]
        self.transformer.set_frequent_labels(cat_vars, RARE_THRESHOLD)
        rare_replaced = self.transformer.replace_rare_labels(cat_vars)
        data.update(rare_replaced)

        # encode categorical and discrete variables
        self.transformer.set_custom_encoder(
            {
                "Cabin": {"Rare":0,"A":1,"G":1,"C":2,"F":2,"B":3,"E":3,"D":3},
                "Embarked": {"S":0, "Q":0, "C":1},
                "Pclass": {3:1, 2:2 ,1:3},
                "SibSp": {8:0, 5:0, 4:1, 3:2, 0:3, 2:4, 1:4},
                "Parch": {6:0, 4:0, 5:1, 0:2, 2:3, 1:4, 3:4},
                "Sex": {"male":0, "female":1}
            }
        )
        encoded = self.transformer.custom_encode(data)
        data.update(encoded)

        # replace missing values
        features_only = data.drop(TARGET, axis=1)
        self.transformer.fit_mice(features_only)
        missing_replaced = self.transformer.replace_missing(features_only)
        data.update(missing_replaced)

        # attempt to normalize distribution of continuous features
        to_log = data.loc[:,self.meta_data["skewed_features"]]
        logged = self.transformer.log(to_log)
        data.update(logged)

        # scale features
        to_scale = data.drop([TARGET, "Age"], axis=1)
        self.transformer.fit_minmax_scaler(to_scale)
        scaled_features = self.transformer.scale(to_scale)
        data.update(scaled_features)

        X_train, X_test, y_train, y_test = self.transformer.split(data, TARGET)

        self.model.fit(X_train, y_train, X_test, y_test) 
        self.model.evaluate_model()

    def predict(self, data):
        # select features
        transformed_data = self.select_features(data)

        # replace rare
        rare_replaced = self.transformer.replace_rare_labels(transformed_data)
        transformed_data.update(rare_replaced)

        # encode
        transformed_data = self.transformer.custom_encode(transformed_data)

        # replace missing
        missing_replaced = self.transformer.replace_missing(transformed_data)
        transformed_data.update(missing_replaced)

        # normalize distribution
        to_log = transformed_data.loc[:,self.meta_data["skewed_features"]]
        logged = self.transformer.log(to_log)
        transformed_data.update(logged)

        # scale
        to_scale = transformed_data.drop("Age", axis=1)
        scaled = self.transformer.scale(to_scale)
        transformed_data.update(scaled)

        # predict
        predictions = self.model.predict(transformed_data).astype(int)

        predictions_df = pd.DataFrame({'PassengerId': data.PassengerId.copy(), 'Survived': predictions.copy()})
        predictions_df.to_csv("./predictions.csv",  index=False)

def select_features(data):
    copy = data.copy()
    copy.drop(["PassengerId", "Name", "Ticket"], inplace=True, axis=1)
    copy.loc[:,"Cabin"] = copy.loc[:,"Cabin"].apply(lambda x: x[0] if type(x) == str else x)
    return copy

pipeline = Pipeline(select_features)

# train
train = pd.read_csv(PATH_TO_TRAINING_DATASET)
pipeline.fit(train)

# test
test = pd.read_csv(PATH_TO_TEST_DATASET)
pipeline.predict(test)
