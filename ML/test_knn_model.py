import cv2
import matplotlib.pyplot as plt
from knn_model import prepare_data
from lbp_utils import detect_and_preprocess_face, extract_combined_features


def test_knn_model(test_image_path, knn, pca, accuracy):
    """
    Uji model KNN menggunakan gambar baru tanpa label.
    """
    print("Membaca gambar uji...")
    test_image = cv2.imread(test_image_path)
    face = detect_and_preprocess_face(test_image)

    if face is None:
        print("No face detected in test image.")
        return

    print("Ekstrak fitur...")
    test_feature = extract_combined_features(face)
    test_feature_reduced = pca.transform([test_feature])

    print("Melakukan prediksi dengan KNN...")
    predicted_label = knn.predict(test_feature_reduced)[0]

    print(f"Hasil Prediksi: {predicted_label}")

    # Visualisasi hasil
    plt.figure(figsize=(8, 8))
    plt.axis("off")
    plt.title(f"Hasil Prediksi: {predicted_label}\nAkurasi Model: {accuracy:.2f}")
    plt.imshow(cv2.cvtColor(test_image, cv2.COLOR_BGR2RGB))
    plt.show()

if __name__ == "__main__":
    # Path dataset dan uji gambar
    data_dir = r"D:\Be\BE\ML\uploads\age"
    test_image_path = r"D:\Be\BE\ML\uploads\images\ridho.jpg"

    knn, X_train, y_train, pca, accuracy = prepare_data(data_dir)

    test_knn_model(test_image_path, knn, pca, accuracy)
