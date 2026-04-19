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
		"""
		Create a model with the given hyperparameters

		:param drop_data: how much of the dataset to ignore
		:param dropout: dropout probability
		:param max_features: maximum number of features for the vectorizer
		:param layer_size_factor: size of the hidden layers, as a fraction of 512/256
		"""

		if self.initialized:
			raise RuntimeError("Model is already initialized")

		# load data, and drop the set fraction of it
		data = self.get_data()
		data = data.drop(data.sample(frac=drop_data).index)
		
		# transform the project data into multi-hot encoded vectors
		self.mlb = MultiLabelBinarizer()
		y = self.mlb.fit_transform(data["wikiprojects"])

		# split the training and test data 80/20
		self.X_train_text, self.X_test_text, self.y_train, self.y_test = train_test_split(
			data["lead"], y, test_size=0.2, random_state=12
		)

		# create the vectorizer
		self.vectorizer = TfidfVectorizer(
			max_features=max_features,
			stop_words="english",
			ngram_range=(1, 2),
			min_df=5
		)

		# train the vectorizer on the training data to avoid leakage
		self.X_train = self.vectorizer.fit_transform(self.X_train_text).toarray()

		# transform the test data using the trained vectorizer
		self.X_test = self.vectorizer.transform(self.X_test_text).toarray()

		self.model = Sequential()

		# add the first hidden layer with dropout
		self.model.add(Dense(int(512 * layer_size_factor), input_dim=self.X_train.shape[1], activation="relu"))
		self.model.add(Dropout(dropout))

		# add the second smaller hidden layer with dropout
		self.model.add(Dense(int(256 * layer_size_factor), activation="relu"))
		self.model.add(Dropout(dropout))

		# create the output layer with one neuron per project
		self.model.add(Dense(y.shape[1], activation="sigmoid"))

		self.model.compile(
			optimizer="adam",
			loss="binary_crossentropy",
			metrics=["accuracy"]
		)

		# save the text for each item in the test set
		self.X_test_text = self.X_test_text.tolist()

		self.initialized = True
		return self

	def train(self, stop_early=True, epochs=10, validation_split=0.1):
		"""
		Train the model with the given parameters.

		:param stop_early: whether to use EarlyStopping callback
		:param epochs: maximum number of epochs to train for
		:param validation_split: how much of the training data to reserve for validation
		"""
		
		# stop early if loss increases
		callbacks = [EarlyStopping(
			monitor="val_loss",
			patience=2,
			restore_best_weights=True
		)] if stop_early else []

		# train the model with given parameters, tracking training time
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
		"""
		Load and return the dataset. 
		"""

		# read the dataset and split projects into list
		data = pd.read_csv("../data/all_data.csv")
		data["wikiprojects"] = data["wikiprojects"].str.split("|")

		# drop any rows with null values
		data = data.dropna()

		# ensure all rows have the expected types
		for _, row in data.iterrows():
			assert type(row["articlename"]) == str
			assert type(row["lead"]) == str
			assert type(row["wikiprojects"]) == list

		# only keep the projects we're classifying; discard the rest
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
		"""
		Run model on the test set and return a report of its performance.

		:param threshold: what prediction confidence is needed to classify
		:param output_dict: whether to output human-readable object or dict
		"""

		y_pred_probs = self.model.predict(self.X_test)
		# use the threshold parameter, or the optimized thresholds if they exist
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
		"""
		Find examples in the test data where there was a misclassification.

		:param project: what project to find misclassifications for
		:param threshold: what confidence is needed to classify
		:param num_samples: how many examples to return
		"""
		
		if not self.initialized:
			raise RuntimeError("Model is not initialized")
		
		if not project in self.mlb.classes_:
			raise ValueError("Project not found in classes")
		
		# find the index of the provided class
		idx = list(self.mlb.classes_).index(project)

		y_pred_probs = self.model.predict(self.X_test, verbose=0)
		# use threshold parameter, or optimized thresholds if they exist
		threshold = getattr(self, "optimized_thresholds", threshold)
		y_pred = (y_pred_probs >= threshold).astype(int)

		# find the places where there are misclassifications, and shuffle them
		wrong_indices = np.where(y_pred[:, idx] != self.y_test[:, idx])[0]
		np.random.shuffle(wrong_indices)

		# take max num_samples examples
		wrong_indices = wrong_indices[:num_samples]

		results = []
		for sidx in wrong_indices:
			# find the true projects, and what the model predicted, from the binarizer
			actual_projects = self.mlb.inverse_transform(np.array([self.y_test[sidx]]))[0]
			predicted_projects = self.mlb.inverse_transform(np.array([y_pred[sidx]]))[0]

			results.append({
				"text": self.X_test_text[sidx],
				"actual": actual_projects,
				"predicted": predicted_projects
			})

		return results
	
	def optimize_thresholds(self, beta=0.6, step=0.05):
		"""
		Find the best per-class thresholds to maximize performance metrics.

		:param beta: value for F_beta score; <1 prioritizes precision
		:param step: how much to step by
		"""
		
		if not self.initialized:
			raise RuntimeError("Model is not initialized")

		# get prediction probabilities
		y_pred_probs = self.model.predict(self.X_test, verbose=0)
		num_classes = len(self.mlb.classes_)
		
		# create list, set to 0.5 for each class by default
		self.optimized_thresholds = np.full(num_classes, 0.5)

		# create range from 0.5-0.95, increasing by step
		threshold_list = np.arange(0.05, 0.96, step)

		# for each class find optimized threshold
		for c in range(num_classes):
			y_true_class = self.y_test[:, c]
			y_prob_class = y_pred_probs[:, c]

			best_f1 = -1
			best_threshold = 0.5

			# calculate fbeta score for each possible threshold, find the best
			for t in threshold_list:
				y_pred_class = (y_prob_class >= t).astype(int)
				score = fbeta_score(y_true_class, y_pred_class, beta=beta, zero_division=0)

				if score > best_f1:
					best_f1 = score
					best_threshold = t
				
			self.optimized_thresholds[c] = best_threshold

		return self.optimized_thresholds

	def save(self, name):
		"""
		Save the model to be loaded later.

		:param name: what folder to save the model to
		"""
		
		if not self.initialized:
			raise RuntimeError("Model is not initialized")
		
		# save all data - model, vectorizer, binarizer, and dataset
		os.mkdir(f"../model/{name}")
		self.model.save(f"../model/{name}/wiki_model.keras")
		joblib.dump(self.vectorizer, f"../model/{name}/tfidf_vectorizer.pkl")
		joblib.dump(self.mlb, f"../model/{name}/mlb_binarizer.pkl")
		joblib.dump([self.X_test_text, self.X_test, self.y_test], f"../model/{name}/test_data.pkl")

	def load(self, name):
		"""
		Load a previously saved model.

		:param name: what folder the model was saved to
		"""
		
		if self.initialized:
			raise RuntimeError("Model is already initialized")

		# load all data - model, vectorizer, binarizer, and dataset
		self.model = load_model(f"../model/{name}/wiki_model.keras")
		self.vectorizer = joblib.load(f"../model/{name}/tfidf_vectorizer.pkl")
		self.mlb = joblib.load(f"../model/{name}/mlb_binarizer.pkl")
		self.X_test_text, self.X_test, self.y_test = joblib.load(f"../model/{name}/test_data.pkl")
		self.initialized = True

		return self
	
	def get_probabilities(self, text):
		"""
		Get the probability for each class for the given text.

		:param text: text to predict
		"""
		
		if not self.initialized:
			raise RuntimeError("Model is not initialized")

		vectorized_text = self.vectorizer.transform([text]).toarray()
		probabilities = self.model.predict(vectorized_text)[0]

		return probabilities

	def predict(self, text, threshold=0.5):
		"""
		Get the predicted classes (probability > threshold) for the given text

		:param text: text to predict
		"""

		probabilities = self.get_probabilities(text)

		# use threshold parameter, or optimized thresholds if they exist
		threshold = getattr(self, "optimized_thresholds", threshold)
		predictions = (probabilities >= threshold).astype(int)
		projects = self.mlb.inverse_transform(np.array([predictions]))[0]

		return projects