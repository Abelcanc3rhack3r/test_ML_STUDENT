import random
import joblib
from sklearn.metrics import accuracy_score
from sklearn import tree
from sklearn import preprocessing
import numpy as np
import pickle
import graphviz
from sklearn.model_selection import train_test_split
import pandas as pd
fake_data= pd.DataFrame(columns=["Sex","Age","Marks"])
# create 100 rows of fake data
for i in range(400):
    fake_data.loc[i]=[random.choice(["M","F"]),random.randint(5,60),random.randint(0,100)]
fake_data.head()
fake_data["pass"]=np.where(((fake_data["Sex"]=="M") & (fake_data["Marks"]>50)) | ((fake_data["Sex"]=="F") & (fake_data["Marks"]>40)), "yes","no")
fake_data["pass"]=np.where(fake_data["Age"]<10, "yes",fake_data["pass"])
fake_data.head(20)

# Encode Sex column into integers
sex = preprocessing.LabelEncoder()
sex.fit(["M", 'F'])
fake_data["Sex"]=sex.transform(fake_data["Sex"])
joblib.dump(sex, "./models/sex.pkl", compress=3)
fake_data.head()

# Encode pass column into integers
ple = preprocessing.LabelEncoder()
ple.fit(["no","yes"])
fake_data["pass"]=ple.transform(fake_data["pass"])
joblib.dump(ple, "./models/ple.pkl", compress=3)
fake_data.head()



# Create a feature vector
feature_vector=fake_data[["Sex","Age","Marks"]]
feature_vector.head()

# Create a target vector
target_vector=fake_data["pass"]
target_vector.head()

# Split the data into train and test
X_train, X_test, y_train, y_test = train_test_split(feature_vector, target_vector, test_size=0.2, random_state=42)



# Create a decision tree classifier
clf = tree.DecisionTreeClassifier()
clf = clf.fit(X_train, y_train)

# Predict the test data using the model
y_pred = clf.predict(X_test)

# Calculate the accuracy of the model

print(accuracy_score(y_test, y_pred))
#test male
joblib.dump(clf, "./models/student_pass_predictor.pkl", compress=3)

clf2 = joblib.load("./models/student_pass_predictor.pkl")

y_pred = clf2.predict(X_test)

# Calculate the accuracy of the model

print(accuracy_score(y_test, y_pred))



y_pred2= clf2.predict([[1,20,0]])
print("pass?", y_pred2)

y_pred3= clf2.predict([[1,20,100]])
print("pass?", y_pred3)