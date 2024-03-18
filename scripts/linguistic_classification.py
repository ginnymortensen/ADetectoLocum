# %%
# %%
# Author: Genevieve Mortensen, code helped by ChatGPT-4
# Modified: 02/26/2024
# Purpose: Process .wav files into linguistic features, extract Llama 2 text embeddings, and classify using SVC, XGBoost, and Neural Network and LR.

import os
import librosa
import torch
import pandas as pd
import numpy as np
from transformers import Wav2Vec2ForCTC, Wav2Vec2Tokenizer
from transformers import AutoTokenizer, AutoModel
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


# Check to see which device is available
def check_device():
    cuda_flag = torch.cuda.is_available()
    print('torch version = ' + torch.__version__ )
    if cuda_flag:
        device = torch.cuda.current_device()
        device_name = torch.cuda.get_device_name(device)
    else:
        device_name = 'CPU'
    print('CUDA version = ' + torch.version.cuda)
    print('current device = '+ device_name)
    return device


# Use models to transcribe audio to text
def audio_to_text(audio_path, device):
    tokenizer = Wav2Vec2Tokenizer.from_pretrained("facebook/wav2vec2-base-960h")
    model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-base-960h").to(device)  # Move model to GPU
    audio_input, _ = librosa.load(audio_path, sr=16000)
    # Convert audio input to the correct tensor, move it to GPU
    inputs = tokenizer(audio_input, return_tensors="pt", padding="longest")
    input_values = inputs.input_values.to(device)  # Move inputs to GPU
    with torch.no_grad(): # not calculating gradients
        logits = model(input_values).logits # Move logits to GPU
    predicted_ids = torch.argmax(logits, dim=-1)
    transcription = tokenizer.batch_decode(predicted_ids.cpu())[0]  # Move tensors to CPU for decoding
    return transcription

#Transcribe training files, return text and labels
def get_training_transcriptions(root_dir, device):
    transcriptions = []
    labels = []  # 0 for 'cn', 1 for 'ad'
    for label_dir in ["cn", "ad"]:
        dir_path = os.path.join(root_dir, label_dir)
        for filename in os.listdir(dir_path):
            if filename.endswith(".wav"):
                audio_path = os.path.join(dir_path, filename)
                transcription = audio_to_text(audio_path, device)
                transcriptions.append(transcription)
                labels.append(0 if label_dir == "cn" else 1)
    print("Training transcriptions and labels acquired")
    labels = np.array(labels)
    return transcriptions, labels

def get_test_transcriptions(test_root_dir, test_labels_path, device):
    transcriptions = []
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
            transcription = audio_to_text(audio_path, device)
            transcriptions.append(transcription)
            file_ids.append(file_id)
            # -1 means we don't have a label for the file and our binary classifiers won't work
            label = file_label_dict.get(file_id, -1)
            labels.append(label)
            if label == -1:
                print(f"Label missing for file_id: {file_id}")
    labels = np.array(labels)
    print("Testing transcriptions, file identifiers, and labels acquired")
    return transcriptions, file_ids, labels

# Extract model 2 embeddings
def get_model_embeddings(texts, model_name, device, batch_size):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        print("Setting pad_token to eos_token.") # no padding token, so we set it to eos_token
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModel.from_pretrained(model_name).to(device)
    embeddings = []
    embedding_sizes = []
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i+batch_size]
        inputs = tokenizer(batch_texts, return_tensors="pt", padding=True, truncation=True, max_length=512).to(device)
        with torch.no_grad(), autocast(): # not calculating gradients, also autocasting - saves memory and speeds up training
            outputs = model(**inputs)
            batch_embeddings = outputs.last_hidden_state.mean(dim=1) # get embedding from last hidden state
            embeddings.append(batch_embeddings.cpu().numpy())
            embedding_sizes.extend([batch_embeddings.size(1)] * len(batch_texts))
    print("It's working, Spongebob! We got the embeddings!")
    # Return the stacked embeddings and their sizes
    return np.vstack(embeddings), embedding_sizes


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

# %%
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


# %%
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

# %%

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
if __name__ == '__main__':

    # Check device available, offload to GPU if available
    device = check_device()

    # Get linguistic features
        # training data
    root_dir = '/home/gamorten/privacyLLM/Healthcare-Chatbots/ADReSSo21/diagnosis/train/audio'
    training_transcriptions, labels = get_training_transcriptions(root_dir, device)
         # testing data
    test_root_dir = '/home/gamorten/privacyLLM/Healthcare-Chatbots/ADReSSo21/diagnosis/test-dist/audio'
    test_labels_path = '/home/gamorten/privacyLLM/Healthcare-Chatbots/groundtruth/task1.csv'
    test_transcriptions, test_file_ids, test_labels = get_test_transcriptions(test_root_dir, test_labels_path, device)

    # Choose your fighter, Llama 2 or Zephyr
    model_name = "meta-llama/Llama-2-7b-hf"
    # model_name = "HuggingFaceH4/zephyr-7b-beta"

    # Extract embeddings in batches
    train_embeddings, train_size = get_model_embeddings(training_transcriptions, model_name=model_name, device=device, batch_size=1)
    test_embeddings, test_size = get_model_embeddings(test_transcriptions, model_name=model_name, device=device, batch_size=1)
    print(train_size)
    print(test_size)

    # Perform classification and get results
    results = {}
    results['Neural Network'] = nn_classifier(train_embeddings, labels, test_embeddings, test_labels)
    results['SVC'] = sv_classifier(train_embeddings, labels, test_embeddings, test_labels)
    results['XGBoost'] = xgboost_classifier(train_embeddings, labels, test_embeddings, test_labels)
    results['Logistic Regression'] = lr_classifier(train_embeddings, labels, test_embeddings, test_labels)

    # Convert results to pandas dataframe
    results_df = pd.DataFrame(results).T
    # display(results_df)

    # Create the table visualization
    fig, ax = plt.subplots(figsize=(12, 2))  # Adjust figure size as needed
    ax.axis('tight')
    ax.axis('off')
    the_table = ax.table(cellText=results_df.values, colLabels=results_df.columns, rowLabels=results_df.index, cellLoc='center', loc='center')
    the_table.auto_set_font_size(False)
    the_table.set_fontsize(12)  # Adjust font size as needed
    the_table.scale(1.5,1.5)  # Adjust scale as needed to fit your content

    plt.title('Linguistic Model Performance Metrics', fontsize=12, y=1.1)  # Adjust title and positioning
    plt.savefig(os.path.join(os.getcwd(), '../results/linguistic_performance.png'))
    plt.show()



