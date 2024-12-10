import cv2
import matplotlib.pyplot as plt
from knn_model import prepare_data, train_knn, label_mapping
from lbp_utils import detect_and_preprocess_face, extract_combined_features

def test_knn_model(test_image_path, knn, X_train, y_train):
    test_image = cv2.imread(test_image_path)
    face = detect_and_preprocess_face(test_image)
    if face is None:
        print("No face detected in test image.")
        return

    test_feature = extract_combined_features(face)
    distances, indices = knn.kneighbors([test_feature], n_neighbors=3)

    top_matches = []
    for i, idx in enumerate(indices[0]):
        label = y_train[idx]
        label_name = [name for name, id_ in label_mapping.items() if id_ == label][0]
        top_matches.append((label_name, distances[0][i]))

    print("Top-3 Matches:")
    for rank, (label, dist) in enumerate(top_matches, 1):
        print(f"{rank}. Label: {label}, Distance: {dist:.2f}")

    plt.imshow(cv2.cvtColor(test_image, cv2.COLOR_BGR2RGB))
    plt.title(f"Predicted: {top_matches[0][0]}")
    plt.axis("off")
    plt.show()

if __name__ == "__main__":
    data_dir = r"D:\Be\BE\ML\uploads\age"
    test_image_path = r"D:\Be\BE\ML\uploads\images\shamil.jpg"

    X, y = prepare_data(data_dir)
    knn, _, _ = train_knn(X, y)
    test_knn_model(test_image_path, knn, X, y)
