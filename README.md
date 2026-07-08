# Wikipedia Article Classification

This repository contains a deep feed-forward neural network designed to automate the multi-label classification of new Wikipedia articles into relevant WikiProjects. This tool acts as a proof-of-concept to assist volunteer reviewers and help reduce the significant backlog of unreviewed articles on Wikipedia. 

## Technologies Used
* **Machine Learning:** TensorFlow/Keras
* **NLP & Data Processing:** scikit-learn (TF-IDF Vectorization, MultiLabelBinarizer)
* **Languages:** Python, Jupyter Notebooks

## File Structure
The codebase is divided into interactive notebooks for experimentation and data collection, alongside a modular Python file for the core model:

* **`data_collection.ipynb`**: Uses Wikipedia's APIs to collect article leads and their associated WikiProjects. It applies random undersampling to balance the dataset and outputs the results to `all_data.csv`.
* **`create_model.ipynb`**: A walkthrough notebook that covers data splitting, text vectorization, multi-label binarization, and initial model training.
* **`model.py`**: A generalized, modular interface containing the `Model` class. This allows for parameterized testing (adjusting dataset size, dropout, etc.) and includes functions to save, load, and generate reports for the models.
* **`evaluation.ipynb`**: Runs automated experiments on a separate thread using the `Model` class to evaluate different hyperparameters and returns performance statistics.
* **`final_model.ipynb`**: Contains the code for the fully optimized final model and its corresponding evaluation metrics.
