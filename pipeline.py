import pandas as pd
import numpy as np

from presci.presci import PreSci
from model.model import Model

PATH_TO_TRAINING_DATASET = "train.csv"
PATH_TO_TEST_DATASET = "test.csv"
TARGET = 'Survived'

if __name__ == '__main__':
    # load dataset
    train = pd.read_csv(PATH_TO_TRAINING_DATASET)
    test = pd.read_csv(PATH_TO_TEST_DATASET)

    def custom_transform(data):
        data.drop(["PassengerId", "Name", "Ticket"], inplace=True, axis=1)
        data.loc[:,"Cabin"] = data.loc[:,"Cabin"].apply(lambda x: x[0] if type(x) == str else x)
        return data

    preprocessor = PreSci(
        train, 
        TARGET,
        unique_threshold=0.7,
        rare_threshold=0.01, 
        skewness_threshold=0.3,
        dont_scale=["Age"],
        remove_outliers=False,
        callback=custom_transform,
        custom_encoders={
            "Cabin": {"Rare":0,"A":1,"G":1,"C":2,"F":2,"B":3,"E":3,"D":3},
            "Embarked": {"S":0, "Q":0, "C":1},
            "Pclass": {3:1, 2:2 ,1:3},
            "SibSp": {8:0, 5:0, 4:1, 3:2, 0:3, 2:4, 1:4},
            "Parch": {6:0, 4:0, 5:1, 0:2, 2:3, 1:4, 3:4},
            "Sex": {"male":0, "female":1}
        }
    )

    X_train, X_test, y_train, y_test = preprocessor.transform_fit()
    model = Model(X_train, y_train, X_test, y_test, solver='saga', max_iter=10000)

    model.evaluate_model()

    transformed_test = preprocessor.transform(test)
    predictions = model.predict(transformed_test).astype(int)

    predictions_df = pd.DataFrame({'PassengerId': test.PassengerId.copy(), 'Survived': predictions.copy()})
    predictions_df.to_csv("./predictions.csv",  index=False)


