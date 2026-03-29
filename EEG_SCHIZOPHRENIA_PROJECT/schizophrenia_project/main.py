# =========================
# EEG SCHIZOPHRENIA PROJECT
# =========================

import os
import re
import numpy as np
import pandas as pd
import mne

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping

import matplotlib.pyplot as plt
import seaborn as sns


# =========================
# PATH CONFIGURATION
# =========================

DATA_PATH = "data/"   # <-- keep your dataset here
CLINICAL_FILE = os.path.join(DATA_PATH, "ASZED_SpreadSheet.csv")


# =========================
# LOAD EDF FILES
# =========================

def get_edf_files(path):
    edf_files = []
    for root, _, files in os.walk(path):
        for file in files:
            if file.endswith(".edf"):
                edf_files.append(os.path.join(root, file))
    return edf_files


def map_subjects(edf_files):
    subject_map = []
    for file in edf_files:
        match = re.search(r"subject_(\d+)", file)
        if match:
            subject_id = int(match.group(1))
            subject_map.append((subject_id, file))
    return subject_map


# =========================
# FEATURE EXTRACTION
# =========================

def extract_eeg_features(edf_file):
    raw = mne.io.read_raw_edf(edf_file, preload=True, verbose=False)

    # Bandpass filter
    raw.filter(0.5, 40)

    data = raw.get_data()
    features = []

    for channel in data:
        features.append(np.mean(channel))
        features.append(np.std(channel))
        features.append(np.var(channel))

    return np.array(features)


# =========================
# BUILD DATASET
# =========================

def build_dataset(subject_map):
    all_features = []
    all_subjects = []

    for subject_id, file_path in subject_map:
        try:
            feats = extract_eeg_features(file_path)
            all_features.append(feats)
            all_subjects.append(subject_id)
        except:
            continue

    features_df = pd.DataFrame(all_features)
    features_df["subject_id"] = all_subjects

    return features_df


# =========================
# MERGE CLINICAL DATA
# =========================

def merge_clinical(features_df):
    clinical = pd.read_csv(CLINICAL_FILE)

    clinical.rename(columns={"sn": "subject_id"}, inplace=True)

    clinical["subject_id"] = clinical["subject_id"].str.replace("subject_", "")
    clinical["subject_id"] = clinical["subject_id"].astype(int)

    clinical["gender"] = clinical["gender"].map({"M": 1, "F": 0})
    clinical["label"] = clinical["category"].map({"Patient": 1, "Control": 0})

    dataset = pd.merge(features_df, clinical, on="subject_id", how="left")

    dataset = dataset.drop(columns=["subject_id", "category", "language", "node"])

    return dataset


# =========================
# TRAIN MODELS
# =========================

def train_models(X_train, X_test, y_train, y_test):

    results = {}

    # Random Forest
    rf = RandomForestClassifier(n_estimators=200, random_state=42)
    rf.fit(X_train, y_train)

    rf_pred = rf.predict(X_test)
    results["Random Forest"] = accuracy_score(y_test, rf_pred)

    print("\nRandom Forest")
    print(confusion_matrix(y_test, rf_pred))
    print(classification_report(y_test, rf_pred))

    # SVM
    svm = SVC(kernel="rbf", C=10)
    svm.fit(X_train, y_train)

    svm_pred = svm.predict(X_test)
    results["SVM"] = accuracy_score(y_test, svm_pred)

    print("\nSVM")
    print(confusion_matrix(y_test, svm_pred))
    print(classification_report(y_test, svm_pred))

    # Logistic Regression
    lr = LogisticRegression(max_iter=500)
    lr.fit(X_train, y_train)

    lr_pred = lr.predict(X_test)
    results["Logistic Regression"] = accuracy_score(y_test, lr_pred)

    print("\nLogistic Regression")
    print(confusion_matrix(y_test, lr_pred))
    print(classification_report(y_test, lr_pred))

    return results


# =========================
# CNN MODEL
# =========================

def train_cnn(X_train, X_test, y_train, y_test):

    X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
    X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)

    model = Sequential()

    model.add(Conv1D(64, 3, activation="relu", input_shape=(X_train.shape[1], 1)))
    model.add(BatchNormalization())
    model.add(MaxPooling1D(2))

    model.add(Conv1D(128, 3, activation="relu"))
    model.add(MaxPooling1D(2))

    model.add(Flatten())
    model.add(Dense(128, activation="relu"))
    model.add(Dropout(0.5))
    model.add(Dense(1, activation="sigmoid"))

    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

    early_stop = EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True)

    model.fit(X_train, y_train, epochs=20, batch_size=32,
              validation_split=0.2, callbacks=[early_stop])

    loss, acc = model.evaluate(X_test, y_test)

    print("\nCNN Accuracy:", acc)

    return acc


# =========================
# MAIN FUNCTION
# =========================

def main():

    print("Loading data...")

    edf_files = get_edf_files(DATA_PATH)
    subject_map = map_subjects(edf_files)

    features_df = build_dataset(subject_map)
    dataset = merge_clinical(features_df)

    X = dataset.drop("label", axis=1)
    y = dataset["label"]

    # Handle missing values
    imputer = SimpleImputer(strategy="mean")
    X = imputer.fit_transform(X)

    # Scaling
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Train ML models
    results = train_models(X_train, X_test, y_train, y_test)

    # Train CNN
    cnn_acc = train_cnn(X_train, X_test, y_train, y_test)

    results["CNN"] = cnn_acc

    print("\nFinal Model Comparison:")
    print(results)


if __name__ == "__main__":
    main()
