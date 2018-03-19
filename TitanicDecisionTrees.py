import numpy as np
import pandas as pd
from sklearn import tree
from sklearn.externals.six import StringIO  
import pydotplus
from sklearn.ensemble import RandomForestClassifier


# ----- FUNCTIONS -----

def cleanData(dataframe):
    dataframe['Sex'] = dataframe['Sex'].map({'male': 0, 'female': 1})
    dataframe['Embarked'] = dataframe['Embarked'].map({'C': 0, 'Q': 1, 'S': 2})
    return dataframe

def createGraph(classifier, filename):
    dot_data = StringIO()  
    tree.export_graphviz(classifier, out_file=dot_data, feature_names=features)  
    graph = pydotplus.graph_from_dot_data(dot_data.getvalue())  
    graph.write_png('./'+ filename +'.png')
    

# ----- TRAINING -----
trainingDF = pd.read_csv('./train.csv', header = 0)
cleanData(trainingDF)

features = list(trainingDF.columns[[2,4,5,6,7,9,11]])
x = trainingDF[features]
y = trainingDF['Survived']

# ----- Draw Decision Tree -----
# clf = tree.DecisionTreeClassifier()
# clf = clf.fit(x,y)
# createGraph(clf,'Titanic')


# ----- Random Forest -----
# This specifies the number of trees to use
clf = RandomForestClassifier(n_estimators=10)
clf = clf.fit(x, y)

# # print (clf.predict([[3, 1, 15, 1, 0, 5,0]]))

# # ----- TESTING -----
testDF= pd.read_csv('./test.csv', header = 0)
cleanData(testDF)

ids = testDF['PassengerId'][:3]
results = pd.DataFrame(clf.predict(testDF[features][:3]), columns=['Survived'])

resultsDF = pd.concat([ids,results], axis=1, join='inner')

resultsDF.to_csv('./results.csv', index=False)

print(resultsDF)


# TODOs

# Sensibly handle age in training set, rather than
# deleting it, generate a random number based on
# the mean and SD.

# Figure out how to pass in an age value for the 
# test data if the field is empty 
