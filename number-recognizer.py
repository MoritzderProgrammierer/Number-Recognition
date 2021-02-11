import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

mnist = tf.keras.datasets.mnist # 28x28 images of handwritten digits from 0 to 9

(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Scaling between 0 and 1, makes it easier for the network to learn 
x_train = tf.keras.utils.normalize(x_train, axis=1)
x_test = tf.keras.utils.normalize(x_test, axis=1)


# Load model
model = tf.keras.models.load_model("epic_num_reader.model")

predictions = model.predict([x_test])
print(predictions)

print(np.argmax(predictions[10]))

plt.imshow(x_test[10])
plt.show()