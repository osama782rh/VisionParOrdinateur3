import os


def analyze_accuracy(data_dir, test_data_dir):
    """
    Analyse des données bike/car pour expliquer l'écart d'accuracy entre train et test.
    """
    bike_train = os.path.join(data_dir, "bike")
    car_train = os.path.join(data_dir, "car")
    bike_test = os.path.join(test_data_dir, "bike")
    car_test = os.path.join(test_data_dir, "car")

    # Compter le nombre d'images
    bike_train_count = len(os.listdir(bike_train))
    car_train_count = len(os.listdir(car_train))
    bike_test_count = len(os.listdir(bike_test))
    car_test_count = len(os.listdir(car_test))

    return (
        f"Dataset d'entraînement :\n"
        f"  - Bike : {bike_train_count}\n"
        f"  - Car : {car_train_count}\n"
        f"Dataset de test :\n"
        f"  - Bike : {bike_test_count}\n"
        f"  - Car : {car_test_count}\n\n"
        f"Explication :\n"
        f"- Si le dataset d'entraînement est déséquilibré ou beaucoup plus grand, le modèle "
        f"peut surapprendre (overfitting).\n"
        f"- Les données de test peuvent avoir une distribution différente, rendant difficile la généralisation."
    )
