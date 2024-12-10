import cv2
import numpy as np
from skimage.feature import hog, local_binary_pattern

def detect_and_preprocess_face(image):
    """
    Deteksi wajah dalam gambar dan lakukan pra-pemrosesan.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(100, 100))

    if len(faces) > 0:
        x, y, w, h = max(faces, key=lambda rect: rect[2] * rect[3])
        face = gray[y:y+h, x:x+w]
        face = cv2.resize(face, (100, 100), interpolation=cv2.INTER_AREA)

        mask = np.zeros_like(face)
        mask = cv2.circle(mask, (50, 50), 40, 255, -1)
        face = cv2.bitwise_and(face, mask)
        return face
    return None

def extract_canny_features(face):
    """
    Ekstraksi fitur Canny edge dari wajah.
    """
    edges = cv2.Canny(face, threshold1=30, threshold2=100)
    edges_hist, _ = np.histogram(edges.ravel(), bins=np.arange(0, 256), density=True)
    return edges_hist

def extract_hog_features(face):
    """
    Ekstraksi fitur HOG dari wajah.
    """
    hog_features = hog(face, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(2, 2), block_norm='L2-Hys', feature_vector=True)
    return hog_features

def extract_lbp_features(face):
    """
    Ekstraksi fitur LBP dari wajah.
    """
    lbp = local_binary_pattern(face, P=8, R=1, method='uniform')
    lbp_hist, _ = np.histogram(lbp.ravel(), bins=np.arange(0, 10), density=True)
    return lbp_hist

def extract_combined_features(face):
    """
    Gabungkan fitur dari Canny, HOG, dan LBP.
    """
    canny_features = extract_canny_features(face)
    hog_features = extract_hog_features(face)
    lbp_features = extract_lbp_features(face)
    return np.hstack((canny_features, hog_features, lbp_features))
