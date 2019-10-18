import pandas as pd
import numpy as np

from sklearn import preprocessing , model_selection , svm , metrics

data = pd.read_csv("Social_Network_Ads.csv")
print(data.head(10))


x = data.iloc[:,[2,3]].values
y = data.iloc[:,4].values

print(x)
print(y)

scaler = preprocessing.StandardScaler()
scaler.fit(x)
scaled_feature = scaler.transform(x)
print(scaled_feature)


train_x , test_x , train_y , test_y = model_selection.train_test_split(scaled_feature , y , test_size=0.3 , random_state=0)

classifier = svm.SVC( )
classifier.fit(train_x , train_y)

pred = classifier.predict(test_x)


print(metrics.confusion_matrix(test_y , pred))
print(metrics.classification_report(test_y , pred))



#prediction
z = np.array([19,19000])
z = z.reshape(1,-1)

scaler1 = preprocessing.StandardScaler()
scaler1.fit(z)
scaled = scaler1.transform(z)

print(classifier.predict(scaled))





