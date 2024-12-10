import os
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from lbp_utils import detect_and_preprocess_face, extract_combined_features
import cv2

label_mapping = {
    '1-10 anak': 0,
    '11-20 remaja': 1,
    '21-30 transisi': 2,
    '31-40 masa matang': 3,
    '41-50 dewasa': 4,
    '51-60 usia pertengahan': 5,
    '61-70 tua': 6,
    '71-80 lanjut usia': 7,
    '81-90 lanjut usia tua': 8
}

def prepare_data(data_dir):
    features, labels = [], []
    for label_dir in os.listdir(data_dir):
        label_path = os.path.join(data_dir, label_dir)
        if os.path.isdir(label_path):
            label = label_mapping.get(label_dir, -1)
            if label == -1:
                continue

            for img_file in os.listdir(label_path):
                img_path = os.path.join(label_path, img_file)
                image = cv2.imread(img_path)
                if image is not None:
                    face = detect_and_preprocess_face(image)
                    if face is not None:
                        feature = extract_combined_features(face)
                        features.append(feature)
                        labels.append(label)

    return np.array(features), np.array(labels)

def train_knn(X, y, k=5):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    knn = KNeighborsClassifier(n_neighbors=k, weights='distance')
    knn.fit(X_train, y_train)

    y_pred = knn.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {accuracy * 100:.2f}%")
    return knn, X_test, y_test
