import numpy as np
import pandas as pd
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier

def cleanData(dataframe):
    dataframe['Sex'] = dataframe['Sex'].map({'male': 0, 'female': 1})
    dataframe['Embarked'] = dataframe['Embarked'].map({'C': 0, 'Q': 1, 'S': 2})
    return dataframe

def addMissingAges(dataFrame, ageMean, ageSD):
    dataFrame['Age'] = np.where(np.isnan(dataFrame['Age']), np.random.normal(ageMean, ageSD, 1)[0], dataFrame['Age'])
    return dataFrame

def createResultsDF(dataframe, classifier, _features):
    ids = dataframe['PassengerId']
    results = pd.DataFrame(classifier.predict(dataframe[_features]), columns=['Survived'])
    return pd.concat([ids,results], axis=1, join='inner')

trainingDF = pd.read_csv('./train.csv', header = 0)
trainingAgeMean = np.nanmean(list(trainingDF['Age']))
trainingAgeSD = np.nanstd(list(trainingDF['Age']))

cleanData(trainingDF)
addMissingAges(trainingDF, trainingAgeMean, trainingAgeSD)

# TODO also consider Fare and Embarked column
# features = list(trainingDF.columns[[2,4,5,6,7,9,11]])
features = list(trainingDF.columns[[2,4,5,6,7]])
x = trainingDF[features]
y = trainingDF['Survived']

clf = RandomForestClassifier(n_estimators=10)
clf = clf.fit(x, y)

testDF= pd.read_csv('./test.csv', header = 0)
cleanData(testDF)
addMissingAges(testDF, trainingAgeMean, trainingAgeSD)

resultsDF = createResultsDF(testDF,clf,features)

resultsDF.to_csv('./results.csv', index=False)
