import matplotlib.pyplot as plt

def plot_training_history(history):
    """
    Tracer les courbes de loss et d'accuracy en fonction des epochs.
    """
    # Si `history` est un objet History (retourné par model.fit), accédez aux données via history.history
    if hasattr(history, 'history'):
        history = history.history

    # Récupérer les données d'historique
    acc = history['accuracy']
    val_acc = history['val_accuracy']
    loss = history['loss']
    val_loss = history['val_loss']
    epochs = range(len(acc))

    # Tracer accuracy
    plt.figure()
    plt.plot(epochs, acc, label='Training Accuracy')
    plt.plot(epochs, val_acc, label='Validation Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.legend()

    # Tracer loss
    plt.figure()
    plt.plot(epochs, loss, label='Training Loss')
    plt.plot(epochs, val_loss, label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.legend()

    plt.show()
