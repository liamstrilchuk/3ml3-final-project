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
from sklearn.metrics import classification_report, fbeta_score

class Model():
	def __init__(self):
		self.initialized = False

	def create(self, drop_data=0, dropout=0.3, max_features=10000, layer_size_factor=1):
		if self.initialized:
			raise RuntimeError("Model is already initialized")

		data = self.get_data()
		data = data.drop(data.sample(frac=drop_data).index)
		
		self.mlb = MultiLabelBinarizer()
		y = self.mlb.fit_transform(data["wikiprojects"])

		self.X_train_text, self.X_test_text, self.y_train, self.y_test = train_test_split(
			data["lead"], y, test_size=0.2, random_state=12
		)

		self.vectorizer = TfidfVectorizer(
			max_features=max_features,
			stop_words="english",
			ngram_range=(1, 2),
			min_df=5
		)

		self.X_train = self.vectorizer.fit_transform(self.X_train_text).toarray()
		self.X_test = self.vectorizer.transform(self.X_test_text).toarray()

		self.model = Sequential()

		self.model.add(Dense(int(512 * layer_size_factor), input_dim=self.X_train.shape[1], activation="relu"))
		self.model.add(Dropout(dropout))

		self.model.add(Dense(int(256 * layer_size_factor), activation="relu"))
		self.model.add(Dropout(dropout))

		self.model.add(Dense(y.shape[1], activation="sigmoid"))

		self.model.compile(
			optimizer="adam",
			loss="binary_crossentropy",
			metrics=["accuracy"]
		)

		self.X_test_text = self.X_test_text.tolist()

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
		threshold = getattr(self, "optimized_thresholds", threshold)
		y_pred = (y_pred_probs >= threshold).astype(int)

		return classification_report(
			self.y_test,
			y_pred,
			target_names=self.mlb.classes_,
			zero_division=0,
			output_dict=output_dict
		)

	def wrong_where(self, project, threshold=0.5, num_samples=5):
		if not self.initialized:
			raise RuntimeError("Model is not initialized")
		
		if not project in self.mlb.classes_:
			raise ValueError("Project not found in classes")
		
		idx = list(self.mlb.classes_).index(project)

		y_pred_probs = self.model.predict(self.X_test, verbose=0)
		threshold = getattr(self, "optimized_thresholds", threshold)
		y_pred = (y_pred_probs >= threshold).astype(int)

		wrong_indices = np.where(y_pred[:, idx] != self.y_test[:, idx])[0]
		np.random.shuffle(wrong_indices)
		wrong_indices = wrong_indices[:num_samples]

		results = []
		for sidx in wrong_indices:
			actual_projects = self.mlb.inverse_transform(np.array([self.y_test[sidx]]))[0]
			predicted_projects = self.mlb.inverse_transform(np.array([y_pred[idx]]))[0]

			results.append({
				"text": self.X_test_text[sidx],
				"actual": actual_projects,
				"predicted": predicted_projects
			})

		return results
	
	def optimize_thresholds(self, step=0.05):
		if not self.initialized:
			raise RuntimeError("Model is not initialized")

		y_pred_probs = self.model.predict(self.X_test, verbose=0)
		num_classes = len(self.mlb.classes_)
		
		self.optimized_thresholds = np.full(num_classes, 0.5)
		threshold_list = np.arange(0.05, 0.96, step)

		for c in range(num_classes):
			y_true_class = self.y_test[:, c]
			y_prob_class = y_pred_probs[:, c]

			best_f1 = -1
			best_threshold = 0.5

			for t in threshold_list:
				y_pred_class = (y_prob_class >= t).astype(int)
				score = fbeta_score(y_true_class, y_pred_class, beta=0.35, zero_division=0)

				if score > best_f1:
					best_f1 = score
					best_threshold = t
				
			self.optimized_thresholds[c] = best_threshold

		return self.optimized_thresholds

	def save(self, name):
		if not self.initialized:
			raise RuntimeError("Model is not initialized")
		
		os.mkdir(f"../model/{name}")
		self.model.save(f"../model/{name}/wiki_model.keras")
		joblib.dump(self.vectorizer, f"../model/{name}/tfidf_vectorizer.pkl")
		joblib.dump(self.mlb, f"../model/{name}/mlb_binarizer.pkl")
		joblib.dump([self.X_test_text, self.X_test, self.y_test], f"../model/{name}/test_data.pkl")

	def load(self, name):
		if self.initialized:
			raise RuntimeError("Model is already initialized")

		self.model = load_model(f"../model/{name}/wiki_model.keras")
		self.vectorizer = joblib.load(f"../model/{name}/tfidf_vectorizer.pkl")
		self.mlb = joblib.load(f"../model/{name}/mlb_binarizer.pkl")
		self.X_test_text, self.X_test, self.y_test = joblib.load(f"../model/{name}/test_data.pkl")
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

		threshold = getattr(self, "optimized_thresholds", threshold)
		predictions = (probabilities >= threshold).astype(int)
		projects = self.mlb.inverse_transform(np.array([predictions]))[0]

		return projects