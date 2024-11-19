from sklearn.model_selection import KFold
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import CategoricalCrossentropy
from tensorflow.keras.metrics import Accuracy
from src.cnn_model.build_model import build_cnn_model
import numpy as np

def evaluate_model(model, X_test, y_test):
    """
    Évalue le modèle sur les données de test.
    """
    results = model.evaluate(X_test, y_test, verbose=0)
    return results[1]  # Retourne l'accuracy

def fine_tune_model(X_train, y_train, X_val, y_val, param_grid):
    """
    Fine-tuning du modèle en testant différentes combinaisons d'hyperparamètres.
    """
    best_model = None
    best_accuracy = 0
    results = []

    for params in param_grid:
        print(f"Test avec paramètres : {params}")
        model = build_cnn_model()

        # Modifier les paramètres
        optimizer = Adam(learning_rate=params['learning_rate'])
        loss = CategoricalCrossentropy()
        model.compile(optimizer=optimizer, loss=loss, metrics=['accuracy'])

        history = model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=params['epochs'],
            batch_size=params['batch_size'],
            verbose=0
        )

        val_accuracy = history.history['val_accuracy'][-1]
        results.append({**params, 'val_accuracy': val_accuracy})

        if val_accuracy > best_accuracy:
            best_accuracy = val_accuracy
            best_model = model

    return best_model, results

def cross_validate_model(X, y, folds=5, epochs=10, batch_size=32):
    """
    Implémente une cross-validation en 5 folds.
    """
    kf = KFold(n_splits=folds, shuffle=True, random_state=42)
    accuracies = []

    for train_index, test_index in kf.split(X):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        model = build_cnn_model()
        model.compile(optimizer=Adam(), loss=CategoricalCrossentropy(), metrics=['accuracy'])
        model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, verbose=0)

        accuracy = model.evaluate(X_test, y_test, verbose=0)[1]
        accuracies.append(accuracy)

    mean_accuracy = np.mean(accuracies)
    std_accuracy = np.std(accuracies)
    return mean_accuracy, std_accuracy
