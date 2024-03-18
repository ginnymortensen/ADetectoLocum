# %%
# %%
# Author: Genevieve Mortensen, code helped by ChatGPT-4
# Modified: 02/26/2024
# Purpose: Process .wav files into acoustic features, extract Llama 2 embeddings, and classify using SVC, XGBoost, neural network, and logistic regression

# Get OPENSMILE
# !curl --header "Host: objects.githubusercontent.com" --header "User-Agent: Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36" --header "Accept: text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.7" --header "Accept-Language: en-US,en;q=0.9" --header "Referer: https://github.com/audeering/opensmile/releases" "https://objects.githubusercontent.com/github-production-release-asset-2e65be/304339469/66433913-6fa7-4bb3-92fc-7f5ee6ad6f21?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=AKIAVCODYLSA53PQK4ZA%2F20240223%2Fus-east-1%2Fs3%2Faws4_request&X-Amz-Date=20240223T042606Z&X-Amz-Expires=300&X-Amz-Signature=7ebd579c9910733ba465155e12ba68b9bd606963d131e1611546431ce23fdb87&X-Amz-SignedHeaders=host&actor_id=0&key_id=0&repo_id=304339469&response-content-disposition=attachment%3B%20filename%3Dopensmile-3.0.2-linux-x86_64.zip&response-content-type=application%2Foctet-stream" -L -o "opensmile-3.0.2-linux-x86_64.zip"
# !unzip opensmile-3.0.2-linux-x86_64.zip
# ./bin/SMILExtract -h (this will confirm correct installation)

import subprocess
import os
import torch
import pandas as pd
import numpy as np
from multiprocessing import Pool
# from transformers import AutoTokenizer, AutoModel
# from ctransformers import AutoModelForCausalLM
from torch.cuda.amp import autocast
import logging
logging.getLogger("transformers").setLevel(logging.ERROR)
import warnings
warnings.filterwarnings("ignore")
from sklearn.svm import SVC
import xgboost as xgb
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_validate
import matplotlib.pyplot as plt

# %%
# Extract eGeMAPS features using OpenSMILE and save output to a CSV file, but also return the vector for embedding
def acoustic_features(audio_path, output_dir, opensmile_config_path):
    
    output_csv_path = os.path.join(output_dir, os.path.basename(audio_path).replace('.wav', '.csv'))
    command = [
        "../opensmile-3.0.2-linux-x86_64/bin/SMILExtract",
        "-C", opensmile_config_path,
        "-I", audio_path,
        "-O", output_csv_path
    ]
    subprocess.run(command, check=True)
    with open(output_csv_path, 'r') as f:
        lines = f.readlines()
        data_start = lines.index('@data\n') + 1   # Read the CSV and return the data after '@data'
        data_lines = lines[data_start:]

    data_list = [line.strip().split(',') for line in data_lines if line.strip()]
    return data_list

# %%
#Extract acoustic features and return labels
def get_training_features(root_dir, output_dir, config_file):
    features_list = []
    labels = []

    for label_dir in ["cn", "ad"]:
        dir_path = os.path.join(root_dir, label_dir)
        for filename in os.listdir(dir_path):
            if filename.endswith(".wav"):
                audio_path = os.path.join(dir_path, filename)
                features = acoustic_features(audio_path, output_dir, config_file)
                features_list.append(features)
                labels.append(0 if label_dir == "cn" else 1)
                
    print("Training features and labels acquired")
    labels = np.array(labels)
    return features_list, labels

#Same with test
def get_test_features(test_root_dir, test_labels_path, output_dir, config_file):
    features_list = []
    file_ids = []
    labels = []
    test_labels_df = pd.read_csv(test_labels_path, header=None)

    label_mapping = {'CN': 0, 'AD': 1}
    test_labels_df['label'] = test_labels_df.iloc[:, 1].map(label_mapping)
    test_labels_df['label'] = test_labels_df['label'].fillna(-1).astype(int)
    file_label_dict = pd.Series(test_labels_df['label'].values, index=test_labels_df.iloc[:, 0].astype(str).str.strip()).to_dict()

    for filename in os.listdir(test_root_dir):
        if filename.endswith(".wav"):
            file_id = filename.split('.')[0].strip()
            audio_path = os.path.join(test_root_dir, filename)
            features = acoustic_features(audio_path, output_dir, config_file)
            features_list.append(features)
            file_ids.append(file_id)
            # -1 means we don't have a label for the file and our binary classifiers won't work
            label = file_label_dict.get(file_id, -1)
            labels.append(label)
            if label == -1:
                print(f"Label missing for file_id: {file_id}")

    labels = np.array(labels)
    print("Testing features, file identifiers, and labels acquired")
    return features_list, file_ids, labels

# %%
# SVC
def sv_classifier(train_embeddings, labels, test_embeddings, test_labels):
    
    svc = SVC()
    param_grid = {
        'kernel': ['linear', 'rbf', 'sigmoid'],
        'C': [0.01, 0.1, 1, 10, 100],
        'gamma': ['scale', 'auto', 0.01, 0.1, 1, 10, 100]
    }

    grid_search = GridSearchCV(svc, param_grid, cv=5, scoring='f1', verbose=1, n_jobs=-1)
    grid_search.fit(train_embeddings, labels)
    print(f"Best hyperparameters: {grid_search.best_params_}")

    best_model = grid_search.best_estimator_
    cv_results = cross_validate(best_model, train_embeddings, labels, cv=5, scoring=['accuracy', 'precision', 'recall', 'f1'])
    svc_predictions = best_model.predict(test_embeddings)

    cv_metrics = {
        'cv_accuracy': round(np.mean(cv_results['test_accuracy']), 3),
        'cv_precision': round(np.mean(cv_results['test_precision']), 3),
        'cv_recall': round(np.mean(cv_results['test_recall']), 3),
        'cv_f1_score': round(np.mean(cv_results['test_f1']), 3),
        'cv_accuracy_std': round(np.std(cv_results['test_accuracy']), 3),
        'cv_precision_std': round(np.std(cv_results['test_precision']), 3),
        'cv_recall_std': round(np.std(cv_results['test_recall']), 3),
        'cv_f1_score_std': round(np.std(cv_results['test_f1']), 3)
    }
    test_metrics = {
        'accuracy': round(accuracy_score(test_labels, svc_predictions), 3),
        'precision': round(precision_score(test_labels, svc_predictions, average='binary'), 3),
        'recall': round(recall_score(test_labels, svc_predictions, average='binary'), 3),
        'f1_score': round(f1_score(test_labels, svc_predictions, average='binary'), 3)
    }
    metrics = {**cv_metrics, **test_metrics}
    return metrics

#XGBoost Classifier
def xgboost_classifier(train_embeddings, labels, test_embeddings, test_labels):

    xgb_clf = xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss')
    param_grid = {
        'max_depth': [3, 4, 5],
        'n_estimators': [50, 100, 200],
        'learning_rate': [0.01, 0.1, 0.2]
    }

    grid_search = GridSearchCV(xgb_clf, param_grid, cv=5, scoring='f1', verbose=1)
    grid_search.fit(train_embeddings, labels)
    print(f"Best XGB hyperparameters: {grid_search.best_params_}")
    
    best_model = grid_search.best_estimator_
    cv_results = cross_validate(best_model, train_embeddings, labels, cv=5, scoring=['accuracy', 'precision', 'recall', 'f1'])
    xgb_predictions = best_model.predict(test_embeddings)

    cv_metrics = {
        'cv_accuracy': round(np.mean(cv_results['test_accuracy']), 3),
        'cv_precision': round(np.mean(cv_results['test_precision']), 3),
        'cv_recall': round(np.mean(cv_results['test_recall']), 3),
        'cv_f1_score': round(np.mean(cv_results['test_f1']), 3),
        'cv_accuracy_std': round(np.std(cv_results['test_accuracy']), 3),
        'cv_precision_std': round(np.std(cv_results['test_precision']), 3),
        'cv_recall_std': round(np.std(cv_results['test_recall']), 3),
        'cv_f1_score_std': round(np.std(cv_results['test_f1']), 3)
    }
    test_metrics = {
        'accuracy': round(accuracy_score(test_labels, xgb_predictions), 3),
        'precision': round(precision_score(test_labels, xgb_predictions, average='binary'), 3),
        'recall': round(recall_score(test_labels, xgb_predictions, average='binary'), 3),
        'f1_score': round(f1_score(test_labels, xgb_predictions, average='binary'), 3)
    }
    metrics = {**cv_metrics, **test_metrics}
    return metrics

# Logistic Regression classifier
def lr_classifier(train_embeddings, labels, test_embeddings, test_labels):

    log_reg = LogisticRegression(solver='liblinear')
    param_grid = {
        'penalty': ['l1', 'l2'],
        'C': [0.01, 0.1, 1, 10, 100]
    }

    grid_search = GridSearchCV(log_reg, param_grid, cv=5, scoring='f1', verbose=1)
    grid_search.fit(train_embeddings, labels)
    print(f"Best hyperparameters: {grid_search.best_params_}")
    
    best_model = grid_search.best_estimator_
    cv_results = cross_validate(best_model, train_embeddings, labels, cv=5, scoring=['accuracy', 'precision', 'recall', 'f1'])
    log_reg_predictions = best_model.predict(test_embeddings)

    cv_metrics = {
        'cv_accuracy': round(np.mean(cv_results['test_accuracy']), 3),
        'cv_precision': round(np.mean(cv_results['test_precision']), 3),
        'cv_recall': round(np.mean(cv_results['test_recall']), 3),
        'cv_f1_score': round(np.mean(cv_results['test_f1']), 3),
        'cv_accuracy_std': round(np.std(cv_results['test_accuracy']), 3),
        'cv_precision_std': round(np.std(cv_results['test_precision']), 3),
        'cv_recall_std': round(np.std(cv_results['test_recall']), 3),
        'cv_f1_score_std': round(np.std(cv_results['test_f1']), 3)
    }
    test_metrics = {
        'accuracy': round(accuracy_score(test_labels, log_reg_predictions), 3),
        'precision': round(precision_score(test_labels, log_reg_predictions, average='binary'), 3),
        'recall': round(recall_score(test_labels, log_reg_predictions, average='binary'), 3),
        'f1_score': round(f1_score(test_labels, log_reg_predictions, average='binary'), 3)
    }
    metrics = {**cv_metrics, **test_metrics}
    return metrics


# The embedding vector is very sparse and not very suitable for neural network classification. The loss is high.
def nn_classifier(train_embeddings, labels, test_embeddings, test_labels):
    n_splits = 5
    kf = KFold(n_splits=n_splits)
    label_encoder = LabelEncoder()
    all_labels = np.concatenate([labels, test_labels])
    label_encoder.fit(all_labels)
    
    cv_accuracies = []
    cv_precisions = []
    cv_recalls = []
    cv_f1_scores = []
    
    for train_index, val_index in kf.split(train_embeddings):
        # Split data into training and validation for the current fold
        train_data, val_data = train_embeddings[train_index], train_embeddings[val_index]
        train_labels, val_labels = labels[train_index], labels[val_index]
        
        # Encode labels for the current fold using the pre-fitted label_encoder
        train_labels_encoded = to_categorical(label_encoder.transform(train_labels))
        val_labels_encoded = to_categorical(label_encoder.transform(val_labels))
        
        # Define and compile the model architecture
        model = Sequential([
            Dense(128, activation='relu', input_shape=(train_embeddings.shape[1],)), # Coudl use sigmoid but relu is better for hidden layers
            Dense(64, activation='relu'),
            Dense(len(np.unique(all_labels)), activation='softmax')
        ])
        model.compile(optimizer=Adam(), loss='categorical_crossentropy', metrics=['accuracy'])
        
        # Train on the current fold
        model.fit(train_data, train_labels_encoded, epochs=10, batch_size=32, verbose=0)
        
        # Predict on the validation set
        val_predictions = np.argmax(model.predict(val_data), axis=-1)
        val_labels_decoded = np.argmax(val_labels_encoded, axis=-1)
        
        # Calculate and store metrics for the current fold
        cv_accuracies.append(accuracy_score(val_labels_decoded, val_predictions))
        cv_precisions.append(precision_score(val_labels_decoded, val_predictions, average='macro'))
        cv_recalls.append(recall_score(val_labels_decoded, val_predictions, average='macro'))
        cv_f1_scores.append(f1_score(val_labels_decoded, val_predictions, average='macro'))
    
    # Average CV metrics
    cv_metrics = {
        'cv_accuracy': round(np.mean(cv_accuracies), 3),
        'cv_precision': round(np.mean(cv_precisions), 3),
        'cv_recall': round(np.mean(cv_recalls), 3),
        'cv_f1_score': round(np.mean(cv_f1_scores), 3),
        'cv_accuracy_std': round(np.std(cv_accuracies), 3),
        'cv_precision_std': round(np.std(cv_precisions), 3),
        'cv_recall_std': round(np.std(cv_recalls), 3),
        'cv_f1_score_std': round(np.std(cv_f1_scores), 3)
    }

    # Encode the labels for the entire dataset for final training and testing
    labels_encoded = to_categorical(label_encoder.transform(labels))
    test_labels_encoded = to_categorical(label_encoder.transform(test_labels))
    
    # Re-initialize, compile, and fit the model on the entire training set
    model = Sequential([
        Dense(128, activation='relu', input_shape=(train_embeddings.shape[1],)),
        Dense(64, activation='relu'),
        Dense(len(np.unique(all_labels)), activation='softmax')
    ])
    model.compile(optimizer=Adam(), loss='categorical_crossentropy', metrics=['accuracy'])
    model.fit(train_embeddings, labels_encoded, epochs=10, batch_size=32, verbose=0)
    
    # Evaluate on the test set
    predictions = np.argmax(model.predict(test_embeddings), axis=-1)
    test_labels_decoded = np.argmax(test_labels_encoded, axis=-1)
    test_metrics = {
        'accuracy': round(accuracy_score(test_labels_decoded, predictions), 3),
        'precision': round(precision_score(test_labels_decoded, predictions, average='binary'), 3),
        'recall': round(recall_score(test_labels_decoded, predictions, average='binary'), 3),
        'f1_score': round(f1_score(test_labels_decoded, predictions, average='binary'), 3)
    }
    # Combine CV and testing metrics
    metrics = {**cv_metrics, **test_metrics}
    return metrics


# %%
import numpy as np

# def preprocess_features(feature_vectors):
#     processed_vectors = []
#     for vector_wrapper in feature_vectors:
#         vector = vector_wrapper[0]  # Extract the actual feature vector
#         processed_vector = []
#         for feature in vector:
#             if feature in ['unknown', '?']:
#                 numeric_feature = 0.0  # Use 0.0 for 'unknown' or '?' values
#             else:
#                 try:
#                     numeric_feature = float(feature)
#                 except ValueError:
#                     numeric_feature = 0.0  # Also use 0.0 if conversion fails
#             processed_vector.append(numeric_feature)
#         processed_vectors.append(processed_vector)
#     else:
#         # Handle unexpected cases where there might be more than one sub-list
#         print("Warning: Found a vector_wrapper with unexpected structure.")
#         # You could extend this part to handle such cases as needed
#     return np.array(processed_vectors)  # Stack processed vectors into a 2D numpy array

def preprocess_features(feature_vectors):
    processed_vectors = []
    for vector_wrapper in feature_vectors:
        vector = vector_wrapper[0]  # Extract the actual feature vector
        processed_vector = [float(feature) for feature in vector if feature not in ['unknown', '?']]
        processed_vectors.append(processed_vector)
    return np.array(processed_vectors)

# %%
if __name__ == "__main__":
    # Get the goods (acoustic features) - This part takes 10-15 minutes to run.
    output_dir = "/home/gamorten/privacyLLM/Healthcare-Chatbots/openSMILE_outputs"
    config_file = "/home/gamorten/privacyLLM/Healthcare-Chatbots/opensmile-3.0.2-linux-x86_64/config/egemaps/v02/eGeMAPSv02.conf"
        # training data
    root_dir = '/home/gamorten/privacyLLM/Healthcare-Chatbots/ADReSSo21/diagnosis/train/audio'
    training_features, labels = get_training_features(root_dir, output_dir, config_file)
         # testing data
    test_root_dir = '/home/gamorten/privacyLLM/Healthcare-Chatbots/ADReSSo21/diagnosis/test-dist/audio'
    test_labels_path = '/home/gamorten/privacyLLM/Healthcare-Chatbots/groundtruth/task1.csv'
    test_features, test_file_ids, test_labels = get_test_features(test_root_dir, test_labels_path, output_dir, config_file)

    # Need to preprocess features for classifier input
    training_features_num = preprocess_features(training_features)
    test_features_num = preprocess_features(test_features)
    print(training_features_num)

    # Classify me, Captain!
    results = {}
    results['Neural Network'] = nn_classifier(training_features_num, labels, test_features_num, test_labels)
    results['SVC'] = sv_classifier(training_features_num, labels, test_features_num, test_labels)
    results['XGBoost'] = xgboost_classifier(training_features_num, labels, test_features_num, test_labels)
    results['Logistic Regression'] = lr_classifier(training_features_num, labels, test_features_num, test_labels)

    # Convert results to pandas dataframe
    results_df = pd.DataFrame(results).T
    # Create the table visualization
    fig, ax = plt.subplots(figsize=(12, 2))  # Adjust figure size as needed
    ax.axis('tight')
    ax.axis('off')
    the_table = ax.table(cellText=results_df.values, colLabels=results_df.columns, rowLabels=results_df.index, cellLoc='center', loc='center')
    the_table.auto_set_font_size(False)
    the_table.set_fontsize(12)  # Adjust font size as needed
    the_table.scale(2,2)  # Adjust scale as needed to fit your content

    plt.title('Acoustic Model Performance Metrics', fontsize=12, y=1.1)  # Adjust title and positioning
    plt.savefig(os.path.join(os.getcwd(), '../results/acoustic_performance.png'))
    plt.show()


