import numpy as np
import tensorflow as tf
import larq as lq
import memory_profiler
import psutil

# Load the MNIST dataset
mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train.astype(np.float32) / 255.0, x_test.astype(np.float32) / 255.0

x_train = np.expand_dims(x_train, axis=-1)
x_test = np.expand_dims(x_test, axis=-1)

y_train = tf.keras.utils.to_categorical(y_train)
y_test = tf.keras.utils.to_categorical(y_test)

# Model with a single regular convolutional layer, max pooling, and regular dense layers
model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(10, activation='softmax')
])


model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model with memory tracking
print("Memory usage before training:")
print(psutil.Process().memory_info().rss)

num_epochs = 3

model.fit(x_train, y_train, batch_size=128, epochs=num_epochs, validation_data=(x_test, y_test), 
          verbose=1)

print("Memory usage after training:")
print(psutil.Process().memory_info().rss)

_, accuracy = model.evaluate(x_test, y_test)
print(f'Test accuracy: {accuracy:.4f}')

# Save the trained model weights
model.save_weights('bitnet_mnist_weights.h5')

# Create a sample input for inference
sample_input = np.random.rand(1, 28, 28, 1)

# Perform inference and measure memory usage
@memory_profiler.profile
def perform_inference():
    model.predict(sample_input)

perform_inference()