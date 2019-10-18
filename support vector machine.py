import pandas as pd
import numpy as np
from sklearn import model_selection , svm
from sklearn.metrics import confusion_matrix , classification_report



data = pd.read_csv("breast-cancer-wisconsin.data")
print(data.head())

data.replace("?" , -99999 , inplace=True)
data.drop(['id'] , axis=1 , inplace=True)

print(data.head(10))

x = np.array(data.drop(["Class"],axis=1))
y = np.array(data["Class"])

train_x , test_x , train_y,test_y = model_selection.train_test_split(x,y , test_size=0.3 , random_state=0)

classifier = svm.SVC()
classifier.fit(train_x , train_y)

pred = classifier.predict(test_x)

print(confusion_matrix(test_y ,pred))
print(classification_report(test_y , pred))

#prediction
z = np.array([[4,2,1,1,1,2,3,2,1] , [4,2,1,2,2,2,3,2,1]])
z = z.reshape(len(z) , -1)
print(classifier.predict(z))
