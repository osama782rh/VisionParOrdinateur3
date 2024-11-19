import os
import shutil
import random
from sklearn.model_selection import train_test_split

def create_balanced_dataset(data_dir, test_data_dir, output_dir):
    """
    Crée un dataset équilibré à partir de data1 et test_data1, puis le sépare en train et test.
    """
    bike_train = os.path.join(data_dir, "bike")
    car_train = os.path.join(data_dir, "car")
    bike_test = os.path.join(test_data_dir, "bike")
    car_test = os.path.join(test_data_dir, "car")

    # Collecter toutes les images
    bike_images = [os.path.join(bike_train, img) for img in os.listdir(bike_train)] + \
                  [os.path.join(bike_test, img) for img in os.listdir(bike_test)]
    car_images = [os.path.join(car_train, img) for img in os.listdir(car_train)] + \
                 [os.path.join(car_test, img) for img in os.listdir(car_test)]

    # Équilibrer les données
    min_count = min(len(bike_images), len(car_images))
    bike_images = random.sample(bike_images, min_count)
    car_images = random.sample(car_images, min_count)

    # Mélanger et séparer en train/test
    all_images = bike_images + car_images
    all_labels = [0] * len(bike_images) + [1] * len(car_images)  # 0: bike, 1: car
    data = list(zip(all_images, all_labels))
    random.shuffle(data)

    train_data, test_data = train_test_split(data, test_size=0.3, random_state=42)

    # Sauvegarder les datasets
    for dataset, name in zip([train_data, test_data], ["train", "test"]):
        dataset_dir = os.path.join(output_dir, name)
        os.makedirs(dataset_dir, exist_ok=True)
        for img_path, label in dataset:
            label_dir = os.path.join(dataset_dir, 'bike' if label == 0 else 'car')
            os.makedirs(label_dir, exist_ok=True)
            shutil.copy(img_path, label_dir)

    return len(train_data), len(test_data)
