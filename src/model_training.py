from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import matplotlib.pyplot as plt


def get_untrained_custom_model(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy']):
    model = Sequential()
    # Dense layers
    model.add(Dense(units=64, activation='relu'))
    model.add(Dense(units=32, activation='relu'))
    model.add(Dense(units=16, activation='relu'))
    # model.add(Dropout(rate=0.5))
    # Output layer
    model.add(Dense(units=2, activation='softmax'))
    
    model.compile(optimizer=optimizer, loss=loss , metrics=metrics)
    return model



def train_model(model, x_train, y_train, epochs, x_val, y_val):
    history = model.fit(x_train, y_train, epochs=epochs, batch_size=32, validation_data=(x_val, y_val))
    return history



def plot_training_history(history):
    # Extract data from the history object
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    epochs_range = range(len(acc))

    # Plot training and validation accuracy
    plt.figure(figsize=(12, 6))
    
    # Accuracy subplot
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, acc, label='Training Accuracy')
    plt.plot(epochs_range, val_acc, label='Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.legend(loc='lower right')

    # Loss subplot
    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, loss, label='Training Loss')
    plt.plot(epochs_range, val_loss, label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend(loc='upper right')

    # Saving the plots
    plt.savefig('../results/training_validation_loss_and_accuracy.png')
    
    # Display the plot
    plt.show()

