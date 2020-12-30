import pandas as pd
import numpy as np

from presci.app.presci import PreSci
from model.model import Model

PATH_TO_TRAINING_DATASET = "train.csv"
PATH_TO_TEST_DATASET = "test.csv"
TARGET = 'Survived'

if __name__ == '__main__':
    # load dataset
    train = pd.read_csv(PATH_TO_TRAINING_DATASET)
    test = pd.read_csv(PATH_TO_TEST_DATASET)

    train.drop(["PassengerId", "Name", "Ticket"], inplace=True, axis=1)
    # select features and apply custom transformation
    def custom_transform(data):
        data.loc[:,"Cabin"] = data.loc[:,"Cabin"].apply(lambda x: x[0] if type(x) == str else x)
        data.loc[:,"Cabin"] = data.loc[:,"Cabin"].copy().replace({"T":0,"A":1,"G":2,"C":3,"F":4,"B":5,"E":6,"D":7})
        data.loc[:,"Embarked"] = data.loc[:,"Embarked"].copy().replace({"S":0, "Q":1, "C":2})
        data.loc[:,"Pclass"] = data.loc[:,"Pclass"].copy().replace({3:1, 1:3})
        data.loc[:,"SibSp"] = data.loc[:,"SibSp"].copy().replace({8:0, 5:0, 4:1, 3:2, 0:3, 2:4, 1:5})
        data.loc[:,"Parch"] = data.loc[:,"Parch"].copy().replace({6:0, 4:0, 5:1, 0:2, 2:3, 1:4, 3:5})
        data.loc[:,"Sex"] = data.loc[:,"Sex"].copy().replace({"male":0, "female":1})
        return data

    preprocessor = PreSci(
        train, 
        TARGET,
        unique_threshold=0.7, 
        rare_threshold=0.01, 
        skewness_threshold=0.3,
        dont_scale=["Age"],
        to_onehot_encode=["Embarked"],
        remove_outliers=False,
        callback=custom_transform
    )

    X_train, X_test, y_train, y_test = preprocessor.transform_fit()
    model = Model(X_train, y_train, X_test, y_test, solver='saga', max_iter=10000)

    model.evaluate_model()

    transformed_test = preprocessor.transform(test)
    predictions = model.predict(transformed_test).astype(int)

    predictions_df = pd.DataFrame({'PassengerId': test.PassengerId.copy(), 'Survived': predictions.copy()})
    predictions_df.to_csv("./predictions.csv",  index=False)


