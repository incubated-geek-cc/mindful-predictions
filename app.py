# web app packages
import requests
from flask import Flask, render_template, redirect, url_for, request,jsonify
from werkzeug.wrappers import Request, Response

from datetime import datetime,timedelta
import json
import os

# for data loading and transformation
import numpy as np 
import pandas as pd

# for statistics output
from scipy import stats
from scipy.stats import randint

# for data preparation and preprocessing for model
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.datasets import make_classification
from sklearn.preprocessing import binarize, LabelEncoder, MinMaxScaler

# models
# Logistic Regression
from sklearn.linear_model import LogisticRegression
# Tree Classifier
from sklearn.tree import DecisionTreeClassifier
# Random Forest
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.model_selection import RandomizedSearchCV
# Bagging
from sklearn.ensemble import BaggingClassifier, AdaBoostClassifier
# KNN
from sklearn.neighbors import KNeighborsClassifier
# Naive Bayes
from sklearn.naive_bayes import GaussianNB 
# Stacking
from mlxtend.classifier import StackingClassifier

# model evaluation and validation 
from sklearn import metrics
from sklearn.metrics import accuracy_score, mean_squared_error, precision_recall_curve
from sklearn.model_selection import cross_val_score

# for db connection
import sqlite3
db_filename="database.db"
# for saving/loading the ML model
import pickle
model_filename="models/model.pkl"

# to bypass warnings in the jupyter notebook
import warnings
from pandas.core.common import SettingWithCopyWarning
warnings.simplefilter(action="ignore", category=SettingWithCopyWarning)

warnings.filterwarnings("ignore",category=UserWarning)
warnings.filterwarnings("ignore",category=DeprecationWarning)
warnings.filterwarnings("ignore",category=FutureWarning)
warnings.filterwarnings("ignore",category=PendingDeprecationWarning)

def get_current_datetime(format_str): # returns a string
	now = datetime.now()
	timestamp = now.strftime(format_str)
	return timestamp

def datetime_str_parser(datetime_str,format_str): # returns a datetime.datetime obj formatted:
	# "%Y-%m-%d %H:%M:%S.%f"
	# 2021-01-05 21:40:05.956493
	datetime_obj=datetime.strptime(datetime_str, format_str)
	return datetime_obj

def datetime_formatter(datetime_obj,format_str): # return a string in desired format
	datetime_str=datetime_obj.strftime(format_str)
	return datetime_str

def datetime_obj_formatter(datetime_obj,format_str): # return a datetime.datetime obj in desired format
	datetime_str=datetime_formatter(datetime_obj,format_str)
	formatted_datetime_obj=datetime_str_parser(datetime_str,format_str)
	return formatted_datetime_obj

app=Flask(__name__)
app.config["SEND_FILE_MAX_AGE_DEFAULT"] = 0

# instantiate index page
@app.route("/")
def index():
   	return render_template("index.html")

# return model predictions
@app.route("/api/predict", methods=["GET"])
def profile_and_predict():
	msg_data={}
	for k in request.args.keys():
		val=request.args.get(k)
		msg_data[k]=val
	f = open("output/X_test.json","r")
	X_test = json.load(f)
	f.close()
	all_cols=X_test
	input_df=pd.DataFrame(msg_data,columns=all_cols,index=[0])
	model = pickle.load(open(model_filename, "rb"))
	arr_results = model.predict(input_df)
	treatment_likelihood=""
	if arr_results[0]==0:
		treatment_likelihood="No"
	elif arr_results[0]==1:
		treatment_likelihood="Yes"
	return treatment_likelihood

# get from [training logs] most recent timestamp
@app.route("/api/latest_logs", methods=["GET"])
def query_latest_logs():
	sqlite_connection = sqlite3.connect(db_filename)
	cursor = sqlite_connection.execute("SELECT T2.`index`, T2.timestamp, T2.description, T2.classification_accuracy, T2.classification_error, T2.false_positive_rate, T2.precision_score, T2.auc_score, T2.cross_validated_auc_score, T2.time_elapsed/1000000000 FROM (SELECT timestamp AS latest_timestamp FROM training_logs ORDER BY DATETIME(timestamp) DESC LIMIT 0,7) AS T1 LEFT JOIN (SELECT * FROM training_logs) AS T2 ON T1.latest_timestamp=T2.timestamp ORDER BY `index`")
	
	rows=cursor.fetchall()
	all_records=rows.copy()
	df=pd.DataFrame(data=all_records,columns=["sn","timestamp","description", 
		 "classification_accuracy", "classification_error","false_positive_rate",
		 "precision_score", "auc_score","cross_validated_auc_score","time_elapsed"])
	response=df.to_json(orient="records")
	sqlite_connection.close()
	return jsonify(response)

# get from [training logs] most recent timestamp
@app.route("/api/latest_timestamp/train_model", methods=["GET"])
def query_model_last_trained():
	sqlite_connection = sqlite3.connect(db_filename)
	cursor = sqlite_connection.execute("SELECT MAX(DATETIME(timestamp)) AS model_last_trained FROM training_logs")
	response={ "model_last_trained": ""}
	for row in cursor:
	   response["model_last_trained"]=row[0]
	sqlite_connection.close()
	return jsonify(response)

# get from [cleaned data] most recent timestamp
@app.route("/api/latest_timestamp/data_updated", methods=["GET"])
def query_data_last_updated():
	sqlite_connection = sqlite3.connect(db_filename)
	cursor = sqlite_connection.execute("SELECT MAX(DATETIME(datetime_created)) AS data_last_updated FROM cleaned_data")
	response={ "data_last_updated": ""}
	for row in cursor:
	   response["data_last_updated"]=row[0]
	sqlite_connection.close()
	return jsonify(response)

# get total no. of records from all years in [cleaned data]
@app.route("/api/no_of_records/total", methods=["GET"])
def query_total_no_of_records():
	sqlite_connection = sqlite3.connect(db_filename)
	cursor = sqlite_connection.execute("SELECT year, COUNT(*) as no_of_records FROM (SELECT * FROM cleaned_data) AS T1 WHERE EXISTS (SELECT year,latest_datetime_created FROM (SELECT year, MAX(datetime_created) AS latest_datetime_created FROM cleaned_data GROUP BY year) AS T2 WHERE T1.year = T2.year AND T1.datetime_created = T2.latest_datetime_created) GROUP BY year UNION ALL SELECT 'Total', COUNT(*) as no_of_records FROM (SELECT * FROM cleaned_data) AS T1 WHERE EXISTS (SELECT year,latest_datetime_created FROM (SELECT year, MAX(datetime_created) AS latest_datetime_created FROM cleaned_data GROUP BY year) AS T2 WHERE T1.year = T2.year AND T1.datetime_created = T2.latest_datetime_created)")
	response=[]
	for row in cursor:
		year=row[0]
		no_of_records=row[1]
		responseObj={"year":year,"no_of_records":no_of_records}
		response.append(responseObj)

	sqlite_connection.close()
	return jsonify(response)

# get latest dataset from in [cleaned data]
@app.route("/api/latest_dataset", methods=["GET"])
def query_latest_data_records():
	selectedPage = request.args.get("selectedPage")
	recordsPerPage = request.args.get("recordsPerPage")
	selectedPage=int(selectedPage)
	recordsPerPage=int(recordsPerPage)

	lowerLimit=(recordsPerPage*selectedPage)-recordsPerPage
	upperLimit=recordsPerPage #(records*selectedPage)

	limit_clause=" LIMIT "+str(lowerLimit)+","+str(upperLimit)

	sql_query="FROM (SELECT * FROM cleaned_data) AS T1 WHERE EXISTS (SELECT year,latest_datetime_created FROM (SELECT year, MAX(datetime_created) AS latest_datetime_created FROM cleaned_data GROUP BY year) AS T2 WHERE T1.year = T2.year AND T1.datetime_created = T2.latest_datetime_created)"

	# Get column names
	sqlite_connection = sqlite3.connect(db_filename)
	sqlite_connection.row_factory = sqlite3.Row
	cursor = sqlite_connection.cursor()
	cursor.execute("SELECT * "+sql_query+limit_clause)
	r=cursor.fetchone() 
	jsonObjKeys=r.keys().copy()
	sqlite_connection.close()

	# Get data records
	sqlite_connection = sqlite3.connect(db_filename)
	cursor = sqlite_connection.execute("SELECT * "+sql_query+limit_clause)
	rows=cursor.fetchall()
	all_records=rows.copy()
	sqlite_connection.close()
	
	df=pd.DataFrame(data=all_records,columns=jsonObjKeys)
	df.drop(["datetime_created"],axis=1,inplace=True)
	response=df.to_json(orient="records")

	# get total no of records for entire table for latest_data_records
	sqlite_connection = sqlite3.connect(db_filename)
	cursor = sqlite_connection.execute("SELECT COUNT(*) AS total_no_of_records "+sql_query)
	responseObj={"total_no_of_records":""}
	for row in cursor:
		total_no_of_records=row[0]
		responseObj["total_no_of_records"]=total_no_of_records
	sqlite_connection.close()

	final_response={
		"data":response,
		"total_no_of_records": responseObj
	}
	return jsonify(final_response)

# get all logs from in [training_logs]
@app.route("/api/training_logs", methods=["GET"])
def query_training_logs():
	selectedPage = request.args.get("selectedPage")
	recordsPerPage = request.args.get("recordsPerPage")
	selectedPage=int(selectedPage)
	recordsPerPage=int(recordsPerPage)

	lowerLimit=(recordsPerPage*selectedPage)-recordsPerPage
	upperLimit=recordsPerPage #(records*selectedPage)

	limit_clause=" LIMIT "+str(lowerLimit)+","+str(upperLimit)
	sql_query="FROM training_logs"

	# Get column names
	sqlite_connection = sqlite3.connect(db_filename)
	sqlite_connection.row_factory = sqlite3.Row
	cursor = sqlite_connection.cursor()
	cursor.execute("SELECT * "+sql_query+limit_clause)
	r=cursor.fetchone() 
	jsonObjKeys=r.keys().copy()
	sqlite_connection.close()

	# Get data records
	sqlite_connection = sqlite3.connect(db_filename)
	cursor = sqlite_connection.execute("SELECT * "+sql_query+limit_clause)
	rows=cursor.fetchall()
	all_records=rows.copy()
	sqlite_connection.close()
	
	df=pd.DataFrame(data=all_records,columns=jsonObjKeys)
	df.drop(["index"],axis=1,inplace=True)
	response=df.to_json(orient="records")

	# get total no of records for entire table for latest_data_records
	sqlite_connection = sqlite3.connect(db_filename)
	cursor = sqlite_connection.execute("SELECT COUNT(*) AS total_no_of_records "+sql_query)
	responseObj={"total_no_of_records":""}
	for row in cursor:
		total_no_of_records=row[0]
		responseObj["total_no_of_records"]=total_no_of_records
	sqlite_connection.close()

	final_response={
		"data":response,
		"total_no_of_records": responseObj
	}
	return jsonify(final_response)

# get total no. of records from all years in [cleaned data]
@app.route("/api/latest/corr_scores", methods=["GET"])
def query_latest_corr_scores():
	sqlite_connection = sqlite3.connect(db_filename)
	cursor = sqlite_connection.execute("SELECT * FROM (SELECT * FROM corr_scores) AS T1 WHERE EXISTS (SELECT latest_datetime_created FROM (SELECT MAX(timestamp) AS latest_datetime_created FROM corr_scores) AS T2 WHERE T1.timestamp = T2.latest_datetime_created)")
	response=[]
	for row in cursor:
		index=row[0]
		correlation_score=row[1]
		responseObj={"Correlation Score":correlation_score,"Feature":index}
		response.append(responseObj)

	sqlite_connection.close()
	return jsonify(response)

# get latest dataset from in [cleaned data]
@app.route("/api/train_model", methods=["POST"])
def create_model():
	# initialise and start logging model training
	date_format="%Y-%m-%d %H:%M:%S.%f"

	start=get_current_datetime(date_format)
	start=datetime_str_parser(start,date_format)
	end = get_current_datetime(date_format)
	end=datetime_str_parser(end,date_format)

	sqlite_connection = sqlite3.connect(db_filename)
	input_df=pd.read_sql("SELECT * FROM (SELECT * FROM cleaned_data) AS T1 WHERE EXISTS (SELECT year,latest_datetime_created FROM (SELECT year, MAX(datetime_created) AS latest_datetime_created FROM cleaned_data GROUP BY year) AS T2 WHERE T1.year = T2.year AND T1.datetime_created = T2.latest_datetime_created)",sqlite_connection)
	sqlite_connection.close()
	input_df.drop(["year","datetime_created"],axis=1,inplace=True)

	all_labels_df={}
	all_encoded_labels_df={}
	label_dict = {}
	df_sklearn=input_df.copy()

	for feature in df_sklearn:
		le = LabelEncoder()
		le.fit(df_sklearn[feature])
		le_name_mapping = dict(zip(le.classes_, le.transform(le.classes_)))
		label_key = "label_" + feature
		if feature=="no_employees":
			     label_value=["1_to_5", "6_to_25", "26_to_100", "100_to_500", "500_to_1000", "more_than_1000"]
			     le_transformer={"1_to_5":1, "6_to_25":2, "26_to_100":3, "100_to_500":4, "500_to_1000":5, "more_than_1000":6}
			     df_sklearn[feature]=df_sklearn[feature].apply(lambda x: le_transformer[x])
			     df_sklearn[feature].apply(int)
		elif df_sklearn[feature].dtype == "int64":
			     label_value = list(df_sklearn[feature].sort_values(ascending=False).unique().copy())
		else:
			     df_sklearn[feature] = le.transform(df_sklearn[feature])
			     label_value = [*le_name_mapping]
		all_labels_df[feature]={}
		all_encoded_labels_df[feature]={}
		label_dict[label_key]=label_value
	     
	features_df_sklearn=input_df.copy()
	df=pd.DataFrame()
	parsed=None

	for feature in features_df_sklearn:
	    transformed_arr=list(df_sklearn[feature])
	    original_arr=list(features_df_sklearn[feature])
	    df=pd.DataFrame({
	        "transformed": transformed_arr,
	        "original": original_arr
	    })
	    df.drop_duplicates(keep="first",inplace=True)
	    result=df.to_json(orient="records")
	    parsed=json.loads(result)
	    subdict={}
	    encoded_subdict={}
	    for p in parsed:
	        transformed_p=p["transformed"]
	        original_p=p["original"]
	        subdict[transformed_p]=original_p
	        encoded_subdict[original_p]=transformed_p
	    all_labels_df[feature]=subdict
	    all_encoded_labels_df[feature]=encoded_subdict

    
	input_df_original=df_sklearn.copy()
	features=list(input_df_original.columns)
	for feature in features:
		input_df_original[feature]=input_df_original[feature].apply(lambda x: all_labels_df[feature][x])
	# get correlation scores of features to target=treatment	
	input_df=df_sklearn.copy()

	# scale features
	scaler=MinMaxScaler()
	input_df["age"] = scaler.fit_transform(input_df[["age"]])

	corrmat=input_df.corr()
	k=(len(input_df.columns)-1)
	cols=corrmat.nlargest(k, "treatment")["treatment"].index
	cm=np.corrcoef(input_df[cols].values.T)
	corr_scores=corrmat["treatment"].sort_values(ascending=False,key=abs)
	corr_scores=corr_scores.iloc[:9]
	corr_scores_df=pd.DataFrame(data=list(corr_scores),columns=["Correlation_Score"],index=corr_scores.index)
	corr_scores_df.sort_values(["Correlation_Score"],ascending=False,key=abs)
	print("\n########### Top Correlation Scores ###############")
	print(corr_scores_df)
	
	timestamp = get_current_datetime(date_format)
	corr_scores_df["timestamp"]=timestamp
	sqlite_connection = sqlite3.connect(db_filename)
	corr_scores_df.to_sql("corr_scores", sqlite_connection, if_exists="append")
	sqlite_connection.close()

	# feature extraction
	data_cols=list(corr_scores.index.copy())
	input_df=input_df[data_cols]

	feature_cols=data_cols.copy()
	feature_cols.remove("treatment")

	# split into training and test datasets
	X = input_df[feature_cols]
	y = input_df.treatment
	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=0)

	# feature importance
	forest = ExtraTreesClassifier(n_estimators=250, random_state=0)
	forest.fit(X, y)
	importances = forest.feature_importances_
	std = np.std([tree.feature_importances_ for tree in forest.estimators_], axis=0)
	indices = np.argsort(importances)[::-1]
	labels = []
	for f in range(X.shape[1]):
		labels.append(feature_cols[f])

	print("\n########### Top Feature Importance ###############")
	feature_imp_df=pd.DataFrame(data=list(importances),columns=["Feature_Importance"],index=labels)
	feature_imp_df.sort_values(["Feature_Importance"], ascending=False)
	print(feature_imp_df)


	print("\n########### Part 3. MODEL TRAINING ###############")
	def evalClassModel(model, y_test, y_pred_class):
		print("Accuracy:", metrics.accuracy_score(y_test, y_pred_class))
		print("Null accuracy:\n", y_test.value_counts())
		print("Percentage of ones:", y_test.mean())
		print("Percentage of zeros:",1 - y_test.mean())
		print("True:", y_test.values[0:25])
		print("Pred:", y_pred_class[0:25])

		confusion = metrics.confusion_matrix(y_test, y_pred_class)
		TP = confusion[1, 1]
		TN = confusion[0, 0]
		FP = confusion[0, 1]
		FN = confusion[1, 0]

		classification_accuracy=metrics.accuracy_score(y_test, y_pred_class)
		classification_error=1-classification_accuracy
		print("Classification Accuracy:", classification_accuracy)
		print("Classification Error:", classification_error)

		false_positive_rate = FP/float(TN + FP)
		precision=metrics.precision_score(y_test, y_pred_class)
		auc_score=metrics.roc_auc_score(y_test, y_pred_class)
		cross_validated_auc_score=cross_val_score(model, X, y, cv=10, scoring="roc_auc").mean()
		print("False Positive Rate:", false_positive_rate)
		print("Precision:", precision)
		print("AUC Score:", auc_score)
		print("Cross-validated AUC:", cross_validated_auc_score)

		print("First 10 predicted responses:\n", model.predict(X_test)[0:10])
		print("First 10 predicted probabilities of class members:\n", model.predict_proba(X_test)[0:10])
		model.predict_proba(X_test)[0:10, 1]
		y_pred_prob = model.predict_proba(X_test)[:, 1]
		y_pred_prob = y_pred_prob.reshape(-1,1)
		y_pred_class = binarize(y_pred_prob, 0.3)[0]
		print("First 10 predicted probabilities:\n", y_pred_prob[0:10])

		roc_auc = metrics.roc_auc_score(y_test, y_pred_prob)
		fpr, tpr, thresholds = metrics.roc_curve(y_test, y_pred_prob)

		def evaluate_threshold(threshold):
			print("Sensitivity for " + str(threshold) + " :", tpr[thresholds > threshold][-1])
			print("Specificity for " + str(threshold) + " :", 1 - fpr[thresholds > threshold][-1])

		predict_mine=np.where(y_pred_prob > 0.50, 1, 0)
		confusion = metrics.confusion_matrix(y_test, predict_mine)
		confusion_df=pd.DataFrame(confusion,columns=["Predicted_N","Predicted_P"],index=["Actual_N","Actual_P"])
		print(confusion_df)

		return [classification_accuracy,classification_error,false_positive_rate,precision,auc_score,cross_validated_auc_score]

	# initialise dict to save accuracy scores and trained models
	method_dict = {}
	rmse_dict = {}
    
	def log_event():
		end = get_current_datetime(date_format)
		end=datetime_str_parser(end,date_format)
		time_elapsed=(end-start)
		return (time_elapsed)

	def logisticRegression():
		logreg = LogisticRegression()
		logreg.fit(X_train, y_train)
		y_pred_class = logreg.predict(X_test)
		print("\n########### Logistic Regression ###############")
		score_arr=evalClassModel(logreg, y_test, y_pred_class)
		accuracy_score=score_arr[0]
		method_dict["Log. Regres."] = accuracy_score * 100
		rmse_dict["Log. Regres."]=logreg
		time_elapsed=log_event()
		score_arr.append(time_elapsed)
		return score_arr

	# parameter tuning methods
	def tuningCV(knn):
		k_range = list(range(1, 31))
		k_scores = []
		for k in k_range:
			knn = KNeighborsClassifier(n_neighbors=k)
			scores=cross_val_score(knn, X, y, cv=10, scoring="accuracy")
			k_scores.append(scores.mean())
		print(k_scores)

	def tuningGridSerach(knn):
		k_range = list(range(1, 31))
		print(k_range)
		param_grid = dict(n_neighbors=k_range)
		print(param_grid)
		grid = GridSearchCV(knn, param_grid, cv=10, scoring="accuracy")
		grid.fit(X, y)
		grid.grid_scores_
		print(grid.grid_scores_[0].parameters)
		print(grid.grid_scores_[0].cv_validation_scores)
		print(grid.grid_scores_[0].mean_validation_score)
		grid_mean_scores = [result.mean_validation_score for result in grid.grid_scores_]
		print(grid_mean_scores)
		print("\nGridSearch best score", grid.best_score_)
		print("\nGridSearch best params", grid.best_params_)
		print("\nGridSearch best estimator", grid.best_estimator_)

	def tuningRandomizedSearchCV(model, param_dist):
		rand = RandomizedSearchCV(model, param_dist, cv=10, scoring="accuracy", n_iter=10, random_state=5)
		rand.fit(X, y)
		rand.cv_results_
		print("\nRand. Best Score: ", rand.best_score_)
		print("\nRand. Best Params: ", rand.best_params_)
		best_scores=[]
		for _ in range(20):
			rand = RandomizedSearchCV(model, param_dist, cv=10, scoring='accuracy', n_iter=10)
			rand.fit(X, y)
			best_scores.append(round(rand.best_score_, 3))
		print(best_scores)

	def tuningMultParam(knn):
		k_range = list(range(1, 31))
		weight_options = ["uniform", "distance"]
		param_grid = dict(n_neighbors=k_range, weights=weight_options)
		print(param_grid)
		grid = GridSearchCV(knn, param_grid, cv=10, scoring="accuracy")
		grid.fit(X, y)
		print(grid.grid_scores_)
		print("\nMultiparam. Best Score: ", grid.best_score_)
		print("\nMultiparam. Best Params: ", grid.best_params_)


	def Knn():
		knn = KNeighborsClassifier(n_neighbors=5)
		k_range = list(range(1, 31))
		weight_options = ["uniform", "distance"]
		param_dist = dict(n_neighbors=k_range, weights=weight_options)
		tuningRandomizedSearchCV(knn, param_dist)
		knn = KNeighborsClassifier(n_neighbors=27, weights="uniform")
		knn.fit(X_train, y_train)
		y_pred_class = knn.predict(X_test)
		print("\n########### KNeighborsClassifier ###############")
		score_arr = evalClassModel(knn, y_test, y_pred_class)
		accuracy_score=score_arr[0]
		method_dict["KNN"] = accuracy_score * 100
		rmse_dict["KNN"]=knn
		time_elapsed=log_event()
		score_arr.append(time_elapsed)
		return score_arr

	def treeClassifier():
		tree = DecisionTreeClassifier()
		features_size = feature_cols.__len__()
		param_dist = {
			"max_depth": [3, None], 
			"max_features": randint(1, features_size), 
			"min_samples_split": randint(2, 9), 
			"min_samples_leaf": randint(1, 9),
			"criterion": ["gini", "entropy"]
		}
		tuningRandomizedSearchCV(tree, param_dist)
		tree = DecisionTreeClassifier(max_depth=3,min_samples_split=8,max_features=6,criterion="entropy",min_samples_leaf=7)
		tree.fit(X_train, y_train)
		y_pred_class = tree.predict(X_test)
		print("\n########### Tree classifier ###############")
		score_arr=evalClassModel(tree, y_test, y_pred_class)
		accuracy_score=score_arr[0]
		method_dict["Tree clas."] = accuracy_score * 100
		rmse_dict["Tree clas."]=tree
		time_elapsed=log_event()
		score_arr.append(time_elapsed)
		return score_arr

	def randomForest():
		forest = RandomForestClassifier(n_estimators = 20)
		features_size = feature_cols.__len__()
		param_dist = {
			"max_depth": [3, None],
			"max_features": randint(1, features_size),
			"min_samples_split": randint(2, 9),
			"min_samples_leaf": randint(1, 9),
			"criterion": ["gini", "entropy"]
		}
		tuningRandomizedSearchCV(forest, param_dist)
		forest = RandomForestClassifier(max_depth = None,min_samples_leaf=8,min_samples_split=2,n_estimators=20,random_state=1)
		my_forest = forest.fit(X_train, y_train)
		y_pred_class = my_forest.predict(X_test)
		print("\n########### Random Forests ###############")
		score_arr=evalClassModel(my_forest, y_test, y_pred_class)
		accuracy_score=score_arr[0]
		method_dict["R. Forest"] = accuracy_score * 100
		rmse_dict["R. Forest"]=my_forest
		time_elapsed=log_event()
		score_arr.append(time_elapsed)
		return score_arr

	def bagging():
		bag = BaggingClassifier(DecisionTreeClassifier(), max_samples=1.0, max_features=1.0, bootstrap_features=False)
		bag.fit(X_train, y_train)
		y_pred_class = bag.predict(X_test)
		print("\n########### Bagging ###############")
		score_arr=evalClassModel(bag, y_test, y_pred_class)
		accuracy_score=score_arr[0]
		method_dict["Bagging"] = accuracy_score * 100
		rmse_dict["Bagging"]=bag
		time_elapsed=log_event()
		score_arr.append(time_elapsed)
		return score_arr

	def boosting():
		clf = DecisionTreeClassifier(criterion="entropy", max_depth=1)
		boost = AdaBoostClassifier(base_estimator=clf, n_estimators=500)
		boost.fit(X_train, y_train)
		y_pred_class = boost.predict(X_test)
		print("\n########### Boosting ###############")
		score_arr=evalClassModel(boost, y_test, y_pred_class)
		accuracy_score=score_arr[0]
		method_dict["Boosting"] = accuracy_score * 100
		rmse_dict["Boosting"]=boost
		time_elapsed=log_event()
		score_arr.append(time_elapsed)
		return score_arr

	def stacking():
		clf1 = KNeighborsClassifier(n_neighbors=1)
		clf2 = RandomForestClassifier(random_state=1)
		clf3 = GaussianNB()
		lr = LogisticRegression()
		stack = StackingClassifier(classifiers=[clf1, clf2, clf3], meta_classifier=lr)
		stack.fit(X_train, y_train)
		y_pred_class = stack.predict(X_test)
		print("\n########### Stacking ###############")
		score_arr = evalClassModel(stack, y_test, y_pred_class)
		accuracy_score=score_arr[0]
		method_dict["Stacking"] = accuracy_score * 100
		rmse_dict["Stacking"]=stack
		time_elapsed=log_event()
		score_arr.append(time_elapsed)
		return score_arr
    
	data_records=[]
    
	def do_something(description):
		start=get_current_datetime(date_format)
		start=datetime_str_parser(start,date_format)
		timestamp = get_current_datetime(date_format)
		data_records.append([description,timestamp])
    
	def do_something_else(score_arr):
		last_index=(len(data_records)-1)
		last_record=data_records[last_index]
		last_record.extend(score_arr)


	do_something("Train Logistic Regression Model")
	score_arr=logisticRegression()
	do_something_else(score_arr)
	
	do_something("Train KNeighbors Classifier Model")
	score_arr=Knn()
	do_something_else(score_arr)

	do_something("Train Decision Tree Classifier Model")	
	score_arr=treeClassifier()
	do_something_else(score_arr)

	do_something("Train Random Forests Model")
	score_arr=randomForest()
	do_something_else(score_arr)

	do_something("Train Bagging Model")
	score_arr=bagging()
	do_something_else(score_arr)

	do_something("Train Boosting Model")
	score_arr=boosting()
	do_something_else(score_arr)	
	
	do_something("Train Stacking Model")
	score_arr=stacking()
	do_something_else(score_arr)

	data=pd.DataFrame(data_records,columns=[
		"description","timestamp","classification_accuracy","classification_error",
		"false_positive_rate","precision_score","auc_score","cross_validated_auc_score","time_elapsed"])
	
	print("\n########### OUTPUT TRAINING LOGS ###############")
	sqlite_connection = sqlite3.connect(db_filename)
	data.to_sql("training_logs", sqlite_connection, if_exists="append")
	sqlite_connection.close()
	response=data.to_json(orient="records")

	print("\n########### Part 4. MODEL EVALUATION ###############")
	s=pd.Series(method_dict)
	s=s.sort_values(ascending=False)
	model_scores=pd.DataFrame(s,columns=["Accuracy Score %"])
	winner=model_scores.index[0]
	winner_score=method_dict[winner]
	model=rmse_dict[winner]
	pickle.dump(model, open(model_filename, "wb"))
	print(winner+" model of accuracy "+str(winner_score)+"% has been saved to "+model_filename)

	with open("output/all_labels_df.json", "w") as outfile:
		json.dump(all_labels_df, outfile)
	with open("output/all_encoded_labels_df.json", "w") as outfile2:
		json.dump(all_encoded_labels_df, outfile2)
	with open("output/X_test.json", "w") as outfile3:
		json.dump(list(X_test.columns), outfile3)

	return jsonify(response)

if __name__ == "__main_":
	app.debug = False
	from werkzeug.serving import run_simple
	run_simple("localhost", 5000, app)