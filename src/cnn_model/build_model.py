from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense


def build_cnn_model():
    """
    Construit un réseau de neurones convolutifs (CNN) pour la classification binaire.
    """
    # Étape 1 : Créer un modèle séquentiel
    cnn_model = Sequential()

    # Étape 2 : Ajouter une couche de convolution
    # Kernel : 3x3, Sorties : 4, Padding : 'same', Activation : ReLU
    cnn_model.add(Conv2D(4, (3, 3), padding='same', activation='relu', input_shape=(64, 64, 3)))

    # Étape 3 : Ajouter une couche de MaxPooling (2x2, stride 2)
    cnn_model.add(MaxPooling2D(pool_size=(2, 2), strides=2))
    # Taille de sortie après cette couche : (32, 32, 4)

    # Étape 4 : Ajouter une deuxième couche de convolution
    # Kernel : 3x3, Sorties : 16, Padding : 'same', Activation : ReLU
    cnn_model.add(Conv2D(16, (3, 3), padding='same', activation='relu'))

    # Étape 5 : Ajouter une couche de MaxPooling (2x2, stride 2)
    cnn_model.add(MaxPooling2D(pool_size=(2, 2), strides=2))
    # Taille de sortie après cette couche : (16, 16, 16)

    # Étape 6 : Ajouter une troisième couche de convolution
    # Kernel : 3x3, Sorties : 32, Padding : 'same', Activation : ReLU
    cnn_model.add(Conv2D(32, (3, 3), padding='same', activation='relu'))

    # Étape 7 : Ajouter une couche de MaxPooling (2x2, stride 2)
    cnn_model.add(MaxPooling2D(pool_size=(2, 2), strides=2))
    # Taille de sortie après cette couche : (8, 8, 32)

    # Étape 8 : Ajouter une couche Flatten pour transformer la matrice en vecteur 1D
    cnn_model.add(Flatten())

    # Étape 9 : Ajouter une couche de sortie dense avec activation softmax
    # Taille de sortie : (2,)
    cnn_model.add(Dense(2, activation='softmax'))

    return cnn_model

# Pour tester le modèle
if __name__ == "__main__":
    model = build_cnn_model()
    model.summary()
