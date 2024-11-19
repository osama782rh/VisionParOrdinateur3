from src.introduction import question1, question2, question3
from src.cnn_model import build_model
from src.cnn_training import train_model
from src.cnn_evaluation import evaluate_model
import numpy as np


def main():
    while True:
        print("=== Projet Vision par Ordinateur ===")
        print("Choisissez une partie à exécuter :")
        print("1 - Introduction")
        print("2 - Création du réseau CNN")
        print("3 - Entraînement du CNN")
        print("4 - Évaluation")
        print("q - Quitter")
        choice = input("Entrez votre choix : ").strip().lower()

        if choice == "1":
            introduction_menu()
        elif choice == "2":
            cnn_menu()
        elif choice == "3":
            cnn_training_menu()
        elif choice == "4":
            cnn_evaluation_menu()
        elif choice == "q":
            print("Merci d'avoir utilisé ce programme. À bientôt !")
            break
        else:
            print("Choix invalide. Veuillez réessayer.")


def introduction_menu():
    while True:
        print("\n=== Partie Introduction ===")
        print("1 - Observer et expliquer l'écart des accuracies")
        print("2 - Créer un dataset équilibré")
        print("3 - Ré-entraîner un SVM")
        print("b - Retour au menu principal")
        choice = input("Choisissez une question (1-3) : ").strip().lower()

        if choice == "1":
            result = question1.analyze_accuracy("data/data1", "data/test_data1")
            print(result)
        elif choice == "2":
            train_count, test_count = question2.create_balanced_dataset(
                "data/data1", "data/test_data1", "data/processed"
            )
            print(f"Dataset équilibré créé : {train_count} images pour l'entraînement, {test_count} images pour le test.")
        elif choice == "3":
            train_acc, test_acc = question3.train_svm("data/processed")
            print(f"Accuracy entraînement : {train_acc:.2f}, Accuracy test : {test_acc:.2f}")
        elif choice == "b":
            break
        else:
            print("Choix invalide. Veuillez réessayer.")


def cnn_menu():
    while True:
        print("\n=== Partie Création du réseau CNN ===")
        print("1 - Construire le modèle CNN")
        print("2 - Résumé et visualisation du modèle")
        print("b - Retour au menu principal")
        choice = input("Choisissez une option (1-2) : ").strip().lower()

        if choice == "1":
            print("Construction du modèle CNN...")
            model = build_model.build_cnn_model()
            print("Modèle construit avec succès !")
        elif choice == "2":
            print("Résumé du modèle CNN :")
            model = build_model.build_cnn_model()
            model.summary()
        elif choice == "b":
            break
        else:
            print("Choix invalide. Veuillez réessayer.")


def cnn_training_menu():
    while True:
        print("\n=== Partie Entraînement du CNN ===")
        print("1 - Compiler et entraîner le modèle")
        print("2 - Tracer les courbes d'accuracy et de loss")
        print("b - Retour au menu principal")
        choice = input("Choisissez une option (1-2) : ").strip().lower()

        if choice == "1":
            print("Compilation et entraînement...")
            model = build_model.build_cnn_model()

            # Exemple de données (générées pour tester le script)
            X_train = np.random.rand(100, 64, 64, 3)
            y_train = np.random.randint(0, 2, (100, 2))
            X_val = np.random.rand(20, 64, 64, 3)
            y_val = np.random.randint(0, 2, (20, 2))

            model, history = train_model.compile_and_train_model(model, X_train, y_train, X_val, y_val)
            print("Modèle entraîné avec succès !")
        elif choice == "2":
            print("Tracer les courbes...")
            # Passer l'objet history simulé
            history = {
                'accuracy': [0.6, 0.7, 0.8, 0.85],
                'val_accuracy': [0.55, 0.65, 0.75, 0.80],
                'loss': [0.5, 0.4, 0.3, 0.2],
                'val_loss': [0.6, 0.5, 0.4, 0.35]
            }
            train_model.plot_training_history(history)
        elif choice == "b":
            break
        else:
            print("Choix invalide. Veuillez réessayer.")


def cnn_evaluation_menu():
    while True:
        print("\n=== Partie Évaluation ===")
        print("1 - Évaluer le modèle sur les données de test")
        print("2 - Fine-tuning des hyperparamètres")
        print("3 - Cross-validation")
        print("b - Retour au menu principal")
        choice = input("Choisissez une option (1-3) : ").strip().lower()

        if choice == "1":
            print("Évaluation sur les données de test...")
            X_test = np.random.rand(20, 64, 64, 3)
            y_test = np.random.randint(0, 2, (20, 2))
            model = build_model.build_cnn_model()
            model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
            accuracy = evaluate_model.evaluate_model(model, X_test, y_test)
            print(f"Accuracy sur les données de test : {accuracy:.2f}")
        elif choice == "2":
            print("Fine-tuning des hyperparamètres...")
            X_train = np.random.rand(100, 64, 64, 3)
            y_train = np.random.randint(0, 2, (100, 2))
            X_val = np.random.rand(20, 64, 64, 3)
            y_val = np.random.randint(0, 2, (20, 2))

            param_grid = [
                {'batch_size': 16, 'epochs': 20, 'learning_rate': 0.001},
                {'batch_size': 32, 'epochs': 50, 'learning_rate': 0.0005},
                {'batch_size': 64, 'epochs': 50, 'learning_rate': 0.001}
            ]
            _, results = evaluate_model.fine_tune_model(X_train, y_train, X_val, y_val, param_grid)
            print("Résultats du fine-tuning :")
            for res in results:
                print(res)
        elif choice == "3":
            print("Cross-validation en 5 folds...")
            X = np.random.rand(120, 64, 64, 3)
            y = np.random.randint(0, 2, (120, 2))
            mean_acc, std_acc = evaluate_model.cross_validate_model(X, y, folds=5, epochs=10, batch_size=32)
            print(f"Accuracy moyenne : {mean_acc:.2f}, Écart type : {std_acc:.2f}")
        elif choice == "b":
            break
        else:
            print("Choix invalide. Veuillez réessayer.")


if __name__ == "__main__":
    main()
