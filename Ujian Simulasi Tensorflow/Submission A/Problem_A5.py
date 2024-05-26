# =======================================================================================
# PROBLEM A5
#
# Build and train a neural network model using the Sunspots.csv dataset.
# Use MAE as the metrics of your neural network model.
# We provided code for normalizing the data. Please do not change the code.
# Do not use lambda layers in your model.
#
# The dataset used in this problem is downloaded from kaggle.com/robervalt/sunspots
#
# Desired MAE < 0.15 on the normalized dataset.
# ========================================================================================

import csv
import tensorflow as tf
import numpy as np
import urllib

# DO NOT CHANGE THIS CODE
def windowed_dataset(series, window_size, batch_size, shuffle_buffer):
    series = tf.expand_dims(series, axis=-1)
    ds = tf.data.Dataset.from_tensor_slices(series)
    ds = ds.window(window_size + 1, shift=1, drop_remainder=True)
    ds = ds.flat_map(lambda w: w.batch(window_size + 1))
    ds = ds.shuffle(shuffle_buffer)
    ds = ds.map(lambda w: (w[:-1], w[1:]))
    return ds.batch(batch_size).prefetch(1)


def solution_A5():
    data_url = 'https://github.com/dicodingacademy/assets/raw/main/Simulation/machine_learning/sunspots.csv'
    urllib.request.urlretrieve(data_url, 'sunspots.csv')

    time_step = []
    sunspots = []

    with open('sunspots.csv') as csvfile:
      reader = csv.reader(csvfile, delimiter=',')
      next(reader)
      for row in reader:
        sunspots.append(int(row[0]))
        time_step.append(float(row[2]))

    series=  # YOUR CODE HERE

    # Normalization Function. DO NOT CHANGE THIS CODE
    min=np.min(series).astype(float)
    max=np.max(series).astype(float)
    series -= min
    series /= max
    time=np.array(time_step)

    # DO NOT CHANGE THIS CODE
    split_time=3000


    time_train=  time[:split_time]
    x_train=  series[:split_time]
    time_valid=  time[split_time:]
    x_valid=  series[split_time:]


    # DO NOT CHANGE THIS CODE
    window_size=30
    batch_size=32
    shuffle_buffer_size=1000


    train_set=windowed_dataset(x_train, window_size=window_size,
                               batch_size=batch_size, shuffle_buffer=shuffle_buffer_size)


    model=tf.keras.models.Sequential([
        #tf.keras.layers.Conv1D(filters=128, kernel_size=5, strides=1, padding='causal', activation='relu',input_shape=[None, 1]),
        tf.keras.layers.LSTM(256, input_shape=[None, 1]),
        #tf.keras.layers.LSTM(30),
        #tf.keras.layers.Dense(10, activation='relu'),
        #tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(1)
    ])
    learning_rate = 1e-5

    # Set the optimizer 
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

    # Set the training parameters
    model.compile(loss=tf.keras.losses.Huber(),
                  optimizer=optimizer,
                  metrics=["mae"])

    history = model.fit(train_set, epochs=100)

    forecast_series = series[split_time-window_size:-1]

    dataset = tf.data.Dataset.from_tensor_slices(series)

    dataset = dataset.window(window_size, shift=1, drop_remainder=True)

    dataset = dataset.flat_map(lambda w: w.batch(window_size))

    dataset = dataset.batch(batch_size).prefetch(1)

    forecast = model.predict(dataset)

    results = forecast.squeeze()

    forecast_valid = results[-len(x_valid):]
    mae = tf.keras.metrics.mean_absolute_error(x_valid, forecast_valid).numpy()
    print("mae" + str(mae))
    
    return model


# The code below is to save your model as a .h5 file.
# It will be saved automatically in your Submission folder.
if __name__ == '__main__':
    # DO NOT CHANGE THIS CODE
    model=solution_A5()
    model.save("model_A5.h5")
