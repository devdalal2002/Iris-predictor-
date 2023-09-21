from flask import Flask,render_template,url_for,request
from flask_material import Material
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn import model_selection
# EDA PKg
import pandas as pd 
import numpy as np 

# ML Pkg
import joblib


app = Flask(__name__)
Material(app)

@app.route('/')
def index():
    return render_template("index.html")

@app.route('/preview')
def preview():
    df = pd.read_csv("data/iris.csv")
    return render_template("preview.html",df_view = df)

@app.route('/',methods=["POST"])
def analyze():
	if request.method == 'POST':
		petal_length = request.form['petal_length']
		sepal_length = request.form['sepal_length']
		petal_width = request.form['petal_width']
		sepal_width = request.form['sepal_width']
		model_choice = request.form['model_choice']

		# Clean the data by convert from unicode to float 
		sample_data = [sepal_length,sepal_width,petal_length,petal_width]
		clean_data = [float(i) for i in sample_data]

		# Reshape the Data as a Sample not Individual Features
		ex1 = np.array(clean_data).reshape(1,-1)
		X = np.array[:,0:8]
		Y = np.array[:,8]
		test_size = 0.33
		seed = 7
		X_train, X_test, Y_train, Y_test = model_selection.train_test_split(X, Y, test_size=test_size, random_state=seed)
		knn = KNeighborsClassifier()
		logit = LogisticRegression()
		svm = SVC()
		knn.fit(X_train, Y_train)
		joblib.dump(knn, 'knn_model_iris.pkl') 
		logit.fit(X_train, Y_train)
		joblib.dump(logit, 'logit_model_iris.pkl')
		svm.fit(X_train, Y_train)
		joblib.dump(svm, 'svm_model_iris.pkl') 


		# ex1 = np.array([6.2,3.4,5.4,2.3]).reshape(1,-1)

		# Reloading the Model
		if model_choice == 'logitmodel':
		    logit_model = joblib.load('data/logit_model_iris.pkl')
		    result_prediction = logit_model.predict(ex1)
		elif model_choice == 'knnmodel':
			knn_model = joblib.load('data/knn_model_iris.pkl')
			result_prediction = knn_model.predict(ex1)
		elif model_choice == 'svmmodel':
			svm_model = joblib.load('data/svm_model_iris.pkl')
			result_prediction = svm_model.predict(ex1)

	return render_template('index.html', petal_width=petal_width,
		sepal_width=sepal_width,
		sepal_length=sepal_length,
		petal_length=petal_length,
		clean_data=clean_data,
		result_prediction=result_prediction,
		model_selected=model_choice)


if __name__ == '__main__':
	app.run(debug=True)