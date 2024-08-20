  import pandas as pd
  import numpy as np
  import matplotlib.pyplot as plt
  from sklearn.pipeline import Pipeline
  from sklearn.base import BaseEstimator, TransformerMixin
  from sklearn.impute import SimpleImputer
  from sklearn.model_selection import StratifiedShuffleSplit
  from sklearn.ensemble import RandomForestClassifier
  from sklearn.model_selection import GridSearchCV

  titanicData = pd.read_csv("/Users/raghavgangatirkar/Downloads/titanic/train.csv")
  print(titanicData)

  #training & testing data 80/20 split, making sure all the parameters have an equal ratio within each DS for accurate training
  split = StratifiedShuffleSplit(n_splits=1, test_size = 0.2)
  #splitting the trainign data and testing data in order to have equal ratios of each parameter
  for train_indices, test_indices in split.split(titanicData, titanicData[["Survived", "Pclass", "Sex"]]):
    trainingSet = titanicData.loc[train_indices]
    testingSet = titanicData.loc[test_indices]

  print(trainingSet)

  #fills in all the missing values in the dataset
  class AgeImputer(BaseEstimator, TransformerMixin):
    def fit (self, X, y=None):
      return self
    def transform(self, X):
      imputer = SimpleImputer(strategy = "mean")
      X['Age'] = imputer.fit_transform(X[['Age']])
      return X

  #essentially going through the categorical columns and making each value numeric, either a one or zero using a OneHot Encoder
  from sklearn.preprocessing import OneHotEncoder
  class FeatureEncoder(BaseEstimator, TransformerMixin):
    def fit(self, X, y= None):
      return self
     #transformer that turns to columns into onehot encoded format
    def transform(self,X):
      encoder = OneHotEncoder()
      matrix = encoder.fit_transform(X[['Embarked']]).toarray()
      column_names = ["C", "S", "Q", "N"]
      for i in range(len(matrix.T)):
        X[column_names[i]] = matrix.T[i]

      matrix = encoder.fit_transform(X[['Sex']]).toarray()
      column_names = ["female", "male"]
      for i in range(len(matrix.T)):
        X[column_names[i]] = matrix.T[i]

      return X

  #gets rid of all the uneeded stuff
  class FeatureDropper(BaseEstimator, TransformerMixin):
    def fit(self, X, y= None):
      return self

    def transform(self,X):
      X = X.drop(["Embarked","Name","Ticket","Cabin","Sex","N"],axis=1, errors = "ignore")
      return X

  #making a pipeline through which I can pass the dataset using the classes I made
  from sklearn.pipeline import Pipeline
  transformationPipeline = Pipeline([("ageimputer", AgeImputer()), ("featureencoder",FeatureEncoder()), ("featuredropper", FeatureDropper())])
  trainingSet = transformationPipeline.fit_transform(trainingSet)
  print(trainingSet)

  #scaling and preparing the training portion of the train dataset
  trainingSet.info()
  X= trainingSet.drop(["Survived"],axis = 1)
  y = trainingSet["Survived"]
  from sklearn.preprocessing import StandardScaler
  scaler = StandardScaler()
  X_data = scaler.fit_transform(X)
  y_data = y.to_numpy()


  #the model
  classifier = RandomForestClassifier()
  parameterGrid = [
    {"n_estimators":[10,100,250,500], "max_depth":[None, 5, 10], "min_samples_split": [2,3,4]}
  ]
  gridSearch = GridSearchCV(classifier,parameterGrid, cv= 3, scoring='accuracy', return_train_score=True)
  gridSearch.fit(X_data, y_data)
  finalClassifier = gridSearch.best_estimator_
  print(finalClassifier)
  testingSet = transformationPipeline.fit_transform(testingSet)
  print(testingSet)

  #test portion of training dataset
  X_test = testingSet.drop(["Survived"],axis = 1)
  y_test = testingSet["Survived"]
  scaler = StandardScaler()
  X_test_data = scaler.fit_transform(X_test)
  y_test_data = y_test.to_numpy()
  print(finalClassifier.score(X_test_data,y_test))

  titanicData = transformationPipeline.fit_transform(titanicData)
  print(titanicData)

  #entire training dataset
  X_titanicData = titanicData.drop(["Survived"],axis = 1)
  y_titanicData = titanicData["Survived"]
  scaler = StandardScaler()
  X_titanicData = scaler.fit_transform(X_titanicData)
  y_titanicData = y_titanicData.to_numpy()

  gridSearch = GridSearchCV(classifier,parameterGrid, cv= 3, scoring='accuracy', return_train_score=True)
  gridSearch.fit(X_titanicData, y_titanicData)
  finalClassifier = gridSearch.best_estimator_
  print(finalClassifier)
  print(finalClassifier.score(X_titanicData, y_titanicData))

  #testing dataset
  titanicRealTestData = pd.read_csv('/Users/raghavgangatirkar/Downloads/titanic/test.csv')
  titanicRealTestData = transformationPipeline.fit_transform(titanicRealTestData)


  #final dataset for testing and submission
  X_finalData = titanicRealTestData.ffill()
  scaler = StandardScaler()
  X_finalData = scaler.fit_transform(X_finalData)
  predictions = finalClassifier.predict(X_finalData)
  final_df = pd.DataFrame(titanicRealTestData['PassengerId'])
  final_df['Survived'] = predictions
  final_df.to_csv('/Users/raghavgangatirkar/Downloads/titanic/predictions.csv', index=False)