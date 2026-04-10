import joblib
import numpy as np
import pandas as pd
import csv
import os
import time

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

from tensorflow.keras.callbacks import EarlyStopping # type: ignore
from tensorflow.keras.models import load_model # type: ignore
from tensorflow.keras.models import Sequential # type: ignore
from tensorflow.keras.layers import Dense, Dropout # type: ignore
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report

class Model():
	def __init__(self):
		self.initialized = False

	def create(self, drop_data=0, dropout=0.3):
		if self.initialized:
			raise RuntimeError("Model is already initialized")

		data = self.get_data()
		data = data.drop(data.sample(frac=drop_data).index)
		
		self.mlb = MultiLabelBinarizer()
		y = self.mlb.fit_transform(data["wikiprojects"])

		X_train_text, X_test_text, self.y_train, self.y_test = train_test_split(
			data["lead"], y, test_size=0.2, random_state=12
		)

		self.vectorizer = TfidfVectorizer(
			max_features=10000,
			stop_words="english",
			ngram_range=(1, 2),
			min_df=5
		)

		self.X_train = self.vectorizer.fit_transform(X_train_text).toarray()
		self.X_test = self.vectorizer.transform(X_test_text).toarray()

		self.model = Sequential()

		self.model.add(Dense(512, input_dim=self.X_train.shape[1], activation="relu"))
		self.model.add(Dropout(dropout))

		self.model.add(Dense(256, activation="relu"))
		self.model.add(Dropout(dropout))

		self.model.add(Dense(y.shape[1], activation="sigmoid"))

		self.model.compile(
			optimizer="adam",
			loss="binary_crossentropy",
			metrics=["accuracy"]
		)

		self.initialized = True
		return self

	def train(self, stop_early=True, epochs=10, validation_split=0.1):
		callbacks = [EarlyStopping(
			monitor="val_loss",
			patience=2,
			restore_best_weights=True
		)] if stop_early else []

		start = time.time()
		history = self.model.fit(
			self.X_train,
			self.y_train,
			epochs=epochs,
			batch_size=32,
			validation_split=validation_split,
			callbacks=callbacks
		)
		time_taken = time.time() - start

		return history, time_taken

	def get_data(self):
		data = pd.read_csv("../data/all_data.csv")
		data["wikiprojects"] = data["wikiprojects"].str.split("|")

		data = data.dropna()

		for _, row in data.iterrows():
			assert type(row["articlename"]) == str
			assert type(row["lead"]) == str
			assert type(row["wikiprojects"]) == list

		keep_projects = []
		with open("../data/wikiprojects.csv", "r") as f:
			reader = csv.reader(f)
			for line in reader:
				keep_projects.append(line[0])

		for idx, row in data.iterrows():
			new_projects = [wp for wp in row["wikiprojects"] if wp in keep_projects]
			data.at[idx, "wikiprojects"] = new_projects

		return data

	def get_report(self, threshold=0.5, output_dict=False):
		y_pred_probs = self.model.predict(self.X_test)
		y_pred = (y_pred_probs >= threshold).astype(int)

		return classification_report(
			self.y_test,
			y_pred,
			target_names=self.mlb.classes_,
			zero_division=0,
			output_dict=output_dict
		)

	def save(self, name):
		if not self.initialized:
			raise RuntimeError("Model is not initialized")
		
		os.mkdir(f"../model/{name}")
		self.model.save(f"../model/{name}/wiki_model.keras")
		joblib.dump(self.vectorizer, f"../model/{name}/tfidf_vectorizer.pkl")
		joblib.dump(self.mlb, f"../model/{name}/mlb_binarizer.pkl")
		joblib.dump([self.X_test, self.y_test], f"../model/{name}/test_data.pkl")

	def load(self, name):
		if self.initialized:
			raise RuntimeError("Model is already initialized")

		self.model = load_model(f"../model/{name}/wiki_model.keras")
		self.vectorizer = joblib.load(f"../model/{name}/tfidf_vectorizer.pkl")
		self.mlb = joblib.load(f"../model/{name}/mlb_binarizer.pkl")
		self.X_test, self.y_test = joblib.load(f"../model/{name}/test_data.pkl")
		self.initialized = True

		return self
	
	def get_probabilities(self, text):
		if not self.initialized:
			raise RuntimeError("Model is not initialized")

		vectorized_text = self.vectorizer.transform([text]).toarray()
		probabilities = self.model.predict(vectorized_text)[0]

		return probabilities

	def predict(self, text, threshold=0.5):
		probabilities = self.get_probabilities(text)

		predictions = (probabilities >= threshold).astype(int)
		projects = self.mlb.inverse_transform(np.array([predictions]))[0]

		return projects