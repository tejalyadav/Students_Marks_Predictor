import pandas as pd
import numpy as np
import sklearn
from sklearn import linear_model
from sklearn.utils import shuffle
import matplotlib.pyplot as plt
from matplotlib import style
import pickle


data = pd.read_csv("student-mat.csv", sep=";")

data = data[["G1", "G2", "G3", "studytime", "failures", "absences"]]
predict = "G3"

x = np.array(data.drop([predict], 1))
y = np.array(data[predict])

x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x, y, test_size=0.2)

linear = linear_model.LinearRegression()

linear.fit(x_train, y_train)
model_fit=linear.fit(x_train, y_train)
accuracy = linear.score(x_test, y_test)
print("The accuracy is {}".format(accuracy))

linear=linear.predict(x_test)
print(linear)

#dumping the model object

pickle.dump(model_fit,open('linear.pkl','wb'))

#Reloading the model object 
linear=pickle.load(open('linear.pkl','rb'))

