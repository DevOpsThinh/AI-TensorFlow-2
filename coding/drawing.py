# Some valuable functions to examine our dataset
#
# Learner: Nguyen Truong Thinh
# Contact me: nguyentruongthinhvn2020@gmail.com || +84393280504

import matplotlib.pyplot as plt

def draw_image(img, label=''):
    plt.imshow(img)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.xlabel(label)

def show_image(img):
    plt.figure()
    draw_image(img)
    plt.colorbar()
    plt.show()

def draw_set_samples(images, labels):
    plt.figure(figsize=(5, 5))
    for i in range(9):
        plt.subplot(3, 3, i+1)
        draw_image(images[i], labels[i])
    plt.show()

def plot(history, label, val_label):
    plt.plot(history.history[label], label=label)
    plt.plot(history.history[val_label], label=val_label)
    plt.xlabel('Epoch')
    plt.legend()
    plt.grid(True)

def plot_history(history):
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plot(history, 'accuracy', 'val_accuracy')
    plt.subplot(1, 2, 2)
    plot(history, 'loss', 'val_loss')
    plt.show()
