import tensorflow as tf
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.utils import to_categorical
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

# Load MNIST dataset
(X_train, y_train), (X_test, y_test) = mnist.load_data()

# Convert target labels to categorical format
y_train = to_categorical(y_train, num_classes=10)
y_test = to_categorical(y_test, num_classes=10)

# Reshape the data to fit the model
X_train = X_train.reshape(60000, 28, 28, 1)
X_test = X_test.reshape(10000, 28, 28, 1)

# Normalize pixel values to be between 0 and 1
X_train = X_train.astype('float32') / 255
X_test = X_test.astype('float32') / 255

# Create a sequential model
model = Sequential()

# Add convolutional layer with 32 filters, kernel size 3x3, and ReLU activation
model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)))

# Add max pooling layer with pool size 2x2
model.add(MaxPooling2D((2, 2)))

# Add convolutional layer with 64 filters, kernel size 3x3, and ReLU activation
model.add(Conv2D(64, (3, 3), activation='relu'))

# Add max pooling layer with pool size 2x2
model.add(MaxPooling2D((2, 2)))

# Add flatten layer
model.add(Flatten())

# Add dense layer with 128 units and ReLU activation
model.add(Dense(128, activation='relu'))

# Add output layer with 10 units and softmax activation
model.add(Dense(10, activation='softmax'))

# Compile the model with Adam optimizer and categorical cross-entropy loss
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model on the training data
model.fit(X_train, y_train, epochs=5, batch_size=128, validation_data=(X_test, y_test))

# Evaluate the model on the test data
test_loss, test_acc = model.evaluate(X_test, y_test)
print(f'Test accuracy: {test_acc:.2f}')

# Define a function to load and preprocess a user image
def image_to_array(image_name):
    image_path = f'{image_name}.png'
    width, height = 28, 28  # Replace with the dimensions required by your model
    image = Image.open(image_path).convert('L')  # Convert to grayscale
    image = image.resize((width, height))
    print(image.width)

    # Convert the image to a NumPy array and normalize the pixel values
    image_array = np.asarray(image)
    image_array = image_array.astype('float32') / 255.0  # Normalize the pixel values between 0 and 1
    print(image_array.shape)

    # Reshape the image array to match the input shape of your model
    image_array = image_array.reshape(1, width, height, 1)  # Assumes the input shape is (width, height, 1)
    return image_array, image

# Load the user image and make a prediction
image_name = 'digit0'  # Replace with the name of your image file
image_array, original_image = image_to_array(image_name)
preds = model.predict(image_array)
pred_class = np.argmax(preds, axis=-1)

# Print the predicted class
print(f'Predicted class: {pred_class[0]}')

# Display the original image and the predicted class
fig, ax = plt.subplots(1, 2, figsize=(10, 5))
ax[0].imshow(original_image, cmap='gray')
ax[0].set_title('Original Image')
ax[1].imshow(image_array.reshape(28, 28), cmap='gray')
ax[1].set_title(f'Predicted class: {pred_class[0]}')
plt.show()