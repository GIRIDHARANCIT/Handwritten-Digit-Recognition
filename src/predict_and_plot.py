import tensorflow as tf
import matplotlib.pyplot as plt

def predict_and_show():
    model = tf.keras.models.load_model("digit_model.h5")
    (_, _), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    x_test = x_test / 255.0

    predictions = model.predict(x_test)

    for i in range(5):
        plt.imshow(x_test[i], cmap=plt.cm.binary)
        plt.title(f"Predicted: {tf.argmax(predictions[i]).numpy()}, Actual: {y_test[i]}")
        plt.show()

if __name__ == "__main__":
    predict_and_show()
