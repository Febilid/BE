import cv2
import os
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from lbp_utils import detect_and_preprocess_face, extract_combined_features, reduce_dimensionality


def prepare_data(data_dir):
    """
    Membaca dataset dan menyiapkan fitur dan label dari dataset.
    """
    features, labels = [], []
    for label_dir in os.listdir(data_dir):
        label_path = os.path.join(data_dir, label_dir)
        if os.path.isdir(label_path):
            for img_file in os.listdir(label_path):
                img_path = os.path.join(label_path, img_file)
                image = cv2.imread(img_path)
                if image is not None:
                    face = detect_and_preprocess_face(image)
                    if face is not None:
                        feature = extract_combined_features(face)
                        features.append(feature)
                        labels.append(label_dir)  # Ambil nama folder sebagai label

    features = np.array(features)
    labels = np.array(labels)
    print(f"Jumlah sampel: {len(features)}, Dimensi fitur awal: {features.shape}")

    # Reduksi Dimensi dengan PCA
    X_reduced, pca = reduce_dimensionality(features)

    # Split data untuk KNN
    X_train, X_test, y_train, y_test = train_test_split(X_reduced, labels, test_size=0.2, random_state=42)

    # Latih model KNN
    knn = KNeighborsClassifier(n_neighbors=5, weights='distance')
    knn.fit(X_train, y_train)

    # Hitung akurasi
    y_pred = knn.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Akurasi model KNN: {accuracy:.2f}")

    return knn, X_train, y_train, pca, accuracy
