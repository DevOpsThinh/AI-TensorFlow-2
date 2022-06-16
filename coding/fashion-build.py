# Recognizing simple clothing items with Fashion-MNIST dataset
#
# Learner: Nguyen Truong Thinh
# Contact me: nguyentruongthinhvn2020@gmail.com || +84393280504

# %%
from ctypes.wintypes import HINSTANCE
import tensorflow as tf
import numpy as np
import drawing as draw
#%%

# %%
dataset = tf.keras.datasets.fashion_mnist
(train_images, train_labels), (test_images, test_labels) = dataset.load_data()
del dataset
#%%

# %%
draw.show_image(train_images[60])
# %%

# Scale image values (after loading & scaling) so each pixel value is in the [0, 1] interval.
train_images = train_images / 255.0
test_images = test_images / 255.0

draw.show_image(train_images[60])

# %%
draw.draw_set_samples(train_images, train_labels)
# %%

classes = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

# %%
# Training/ Building a simple model
model = tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28)),
    tf.keras.layers.Dense(300, activation='relu'),
    tf.keras.layers.Dense(10, activation='softmax')
])
model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(),
              metrics=['accuracy'])

history = model.fit(train_images, train_labels, epochs=10, validation_split=0.1)

test_loss, test_acc = model.evaluate(test_images, test_labels, verbose=2)

draw.plot_history(history)
# %%
# Saving the fashion model
model.save('fashion')

# %%
new_train_images = tf.expand_dims(train_images, 3)
new_test_images = tf.expand_dims(test_images, 3)
# %%
# Training/ Building an improved model with some extra layers
model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(64, 2, padding='same', activation='relu'),
    tf.keras.layers.MaxPool2D(2),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Conv2D(32, 2, padding='same', activation='relu'),
    tf.keras.layers.MaxPool2D(2),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(256, activation='relu'),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(10, activation='softmax')
])
model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(),
              metrics=['accuracy'])

history = model.fit(new_train_images, train_labels, epochs=10, validation_split=0.1)

test_loss, test_acc = model.evaluate(new_test_images, test_labels, verbose=2)

draw.plot_history(history)
# %%
# Saving the fashion model
model.save('fashion-2')

