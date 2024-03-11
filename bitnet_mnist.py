import numpy as np
import tensorflow as tf
import larq as lq
import memory_profiler
import psutil



# Load the MNIST dataset
mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()
#test using binary inputs
# x_train, x_test = np.where(x_train > 0.5, 1.0, -1.0).astype(np.float32), np.where(x_test > 0.5, 1.0, -1.0).astype(np.float32)

#regular inputs
x_train, x_test = x_train.astype(np.float32) / 255.0, x_test.astype(np.float32) / 255.0

x_train = np.expand_dims(x_train, axis=-1)
x_test = np.expand_dims(x_test, axis=-1)
y_train = tf.keras.utils.to_categorical(y_train)
y_test = tf.keras.utils.to_categorical(y_test)
# Model with regular layers, size is way too big
# model = tf.keras.Sequential([
#     tf.keras.layers.Conv2D(64, (3, 3), activation='relu', input_shape=(28, 28, 1)),
#     tf.keras.layers.MaxPooling2D((2, 2)),
#     tf.keras.layers.Flatten(),
#     tf.keras.layers.Dense(256, activation='relu'),
#     tf.keras.layers.Dense(10, activation='softmax')
# ])
# # Model with binary layers using Larq, accuracy way too low but size is only 100kb
model = tf.keras.Sequential([
    lq.layers.QuantConv2D(64, (3, 3), activation='ste_sign', input_shape=(28, 28, 1), kernel_quantizer='ste_sign', kernel_constraint='weight_clip'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.MaxPooling2D((2, 2)),
    lq.layers.QuantConv2D(128, (3, 3), activation='ste_sign', kernel_quantizer='ste_sign', kernel_constraint='weight_clip'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Flatten(),
    lq.layers.QuantDense(256, activation='ste_sign', kernel_quantizer='ste_sign', kernel_constraint='weight_clip'),
    tf.keras.layers.BatchNormalization(),
    lq.layers.QuantDense(10, activation='softmax', kernel_quantizer='ste_sign', kernel_constraint='weight_clip')
])
#model 3 using a combination of binary and regular layers
# model = tf.keras.Sequential([
#     tf.keras.layers.Conv2D(64, (3, 3), activation='relu', input_shape=(28, 28, 1)),
#     tf.keras.layers.MaxPooling2D((2, 2)),
#     lq.layers.QuantConv2D(128, (3, 3), activation='ste_sign', kernel_quantizer='ste_sign', kernel_constraint='weight_clip'),
#     tf.keras.layers.MaxPooling2D((2, 2)),
#     tf.keras.layers.Flatten(),
#     lq.layers.QuantDense(256, activation='ste_sign', kernel_quantizer='ste_sign', kernel_constraint='weight_clip'),
#     tf.keras.layers.Dense(10, activation='softmax')
# ])

#binary optimizer
# optimizer = lq.optimizers.Bop(lr=0.01)
#regular optimizer
optimizer = tf.keras.optimizers.Adam(lr=0.01)

model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model with memory tracking
print("Memory usage before training:")
print(psutil.Process().memory_info().rss / 1024)

num_epochs = 10
batch_size = 64

model.fit(x_train, y_train, batch_size=batch_size, epochs=num_epochs, validation_data=(x_test, y_test), verbose=1)

print("Memory usage after training:")
print(psutil.Process().memory_info().rss / 1024)

_, accuracy = model.evaluate(x_test, y_test)
print(f'Test accuracy: {accuracy:.4f}')

# Save the trained model weights
model.save_weights('bitnet_mnist_weights.h5')

# Create a sample input for inference
sample_input = np.random.rand(1, 28, 28, 1)


@memory_profiler.profile
def profile_memory():
    model.predict(sample_input)


#For normal model evaluation
# Calculate the total number of parameters
# total_params = model.count_params()

# # Calculate the model size in bytes
# model_size_bytes = total_params * 4

# # Convert bytes to kilobytes (KB)
# model_size_kb = model_size_bytes / 1024

# print(f"Regular model size: {model_size_kb:.2f} KB")

#for larq model evaluation
# Calculate the total number of parameters
total_params = model.count_params()

# Calculate the model size in bits
model_size_bits = total_params

# Convert bits to bytes
model_size_bytes = model_size_bits / 8

# Convert bytes to kilobytes (KB)
model_size_kb = model_size_bytes / 1024

print(f"Binarized model size: {model_size_kb:.2f} KB")


# Calculate the model size of half binary fields
# total_params = model.count_params()
# model_size_bytes = total_params * 0.5  # Assuming 50% binarization
# model_size_kb = model_size_bytes / 1024
# print(f"Model size: {model_size_kb:.2f} KB")

profile_memory()