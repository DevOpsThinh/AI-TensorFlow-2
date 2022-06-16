# Recognizing simple clothing items with Fashion-MNIST dataset
# 
# Learner: Nguyen Truong Thinh
# Contact me: nguyentruongthinhvn2020@gmail.com || +84393280504

# %%
import tensorflow as tf
import numpy as np
import drawing as draw
# %%
# Loading the CNN model
model = tf.keras.models.load_model('fashion-2')

classes = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

# %%
# Testing the model with a realistic image loaded from disk.
def classify_image_file(imageFile):
    #Load, scale the image down to 28px*28px and make it grayscale
    img = tf.keras.preprocessing.image.load_img(imageFile,
        color_mode='grayscale',
        target_size=(28, 28),
        interpolation='bicubic')
    # Scale the image's pixel values from the [0,255] to the [0.1] interval.
    img = tf.keras.preprocessing.image.img_to_array(img) / 255.0
    classify_image(img)

def classify_image(subject):
    draw.show_image(subject)
    # Create a batch from this single image.
    subject = tf.expand_dims(subject, 0)
    # Get the first prediction from the sing-image batch.
    score = model.predict(subject)[0]
    print(
        "I'm convinced this is a {} ({:.2f}% confidence)."
        .format(classes[np.argmax(score)], 100 * np.max(score))
    )
    print("But it could also be: ")
    for i in range(0, score.shape[0]):
        print("{:5.2f}% says it's a {}".format(score[i] * 100, classes[i]))
# %%
# Evaluating the fashion-2 with a realistic clothing image by A convolutional neutral network
classify_image_file("shirt.png")
# %%