import tensorflow as tf
from tensorflow.keras import datasets, layers, models, losses
import matplotlib.pyplot as plt

(train_imgs, train_lbls), (test_imgs, test_lbls) = datasets.cifar10.load_data()

train_imgs, test_imgs = train_imgs/255.0, test_imgs/255.0

class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck']

plt.figure(figsize = (10,10))
for i in range(25):
    plt.subplot(5,5,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(train_imgs[i])
    plt.xlabel(class_names[train_lbls[i][0]])
plt.show()

model = models.Sequential()
model.add(layers.Conv2D(32, (3,3), activation="relu", input_shape=(32, 32, 3)))
model.add(layers.MaxPooling2D((2, 2)))

model.add(layers.Conv2D(64, (3,3), activation="relu"))
model.add(layers.MaxPooling2D((2, 2)))

model.add(layers.Conv2D(32, (3,3), activation="relu"))

model.add(layers.Flatten())
model.add(layers.Dense(units=64, activation="relu"))
model.add(layers.Dense(10))

model.summary()

model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True) ,
              metrics=["accuracy"])

model_history = model.fit(train_imgs, train_lbls, validation_data=(test_imgs, test_lbls), epochs=3)

plt.plot(model_history.history['accuracy'], label="accuracy")
plt.plot(model_history.history['val_accuracy'], label="val_accuracy")
plt.xlabel("epochs")
plt.ylabel("Accuracy")
plt.ylim([0.5, 1])
plt.legend(loc='lower right')


test_loss, test_acc = model.evaluate(test_imgs,  test_lbls, verbose=2)

print("Test accuracy: ", test_acc * 100)
