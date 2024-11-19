import os
import numpy as np
from PIL import Image
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score


def extract_features_labels(data_dir):
    """
    Extrait les caractéristiques (images redimensionnées) et labels des données.
    """
    categories = ['bike', 'car']
    images, labels = [], []

    for label, category in enumerate(categories):
        category_dir = os.path.join(data_dir, category)
        for img_name in os.listdir(category_dir):
            img_path = os.path.join(category_dir, img_name)
            try:
                # Ouvrir l'image avec Pillow
                img = Image.open(img_path)

                # Convertir en RGB si l'image a un mode incompatible
                if img.mode not in ('RGB', 'L'):
                    img = img.convert('RGB')

                img_resized = img.resize((64, 64))
                images.append(np.array(img_resized).flatten())  # Aplatir
                labels.append(label)
            except Exception as e:
                print(f"Image ignorée : {img_path} - Erreur : {e}")

    return np.array(images), np.array(labels)


def train_svm(data_dir):
    """
    Entraîne un SVM sur les données extraites.
    """
    # Charger les données
    X_train, y_train = extract_features_labels(os.path.join(data_dir, "train"))
    X_test, y_test = extract_features_labels(os.path.join(data_dir, "test"))

    # Entraîner un SVM
    model = SVC(kernel="linear", random_state=42)
    model.fit(X_train, y_train)

    # Calculer les accuracies
    train_accuracy = accuracy_score(y_train, model.predict(X_train))
    test_accuracy = accuracy_score(y_test, model.predict(X_test))

    return train_accuracy, test_accuracy
