import tensorflow as tf


def rmse_loss(y_true, y_pred):
    """
    Root Mean Squared Error (RMSE) loss function.

    Args:
        y_true (tf.Tensor): True values.
        y_pred (tf.Tensor): Predicted values.

    Returns:
        tf.Tensor: RMSE loss.

    """
    return tf.sqrt(tf.reduce_mean(tf.square(y_true - y_pred)))


def train_autoencoder(model, data, path_to_save):
    """
    Train an autoencoder model.

    Args:
        model (tf.keras.Model): Autoencoder model.
        data (tf.Tensor): Input data for training.
        path_to_save (str): Path to save the trained model.

    Returns:
        None

    """
    model.compile(optimizer=tf.keras.optimizers.Adam(),
                  loss=rmse_loss,
                  metrics=['mae', 'mse'])

    def exponential_decay(lr0, s):
        def exponential_decay_fn(epoch):
            return lr0 * 0.1 ** (epoch / s)

        return exponential_decay_fn

    #  learning rate will decrease by a factor of 10 every 20 epochs.
    exponential_decay_fn = exponential_decay(lr0=0.001, s=20)
    lr_scheduler = tf.keras.callbacks.LearningRateScheduler(exponential_decay_fn)
    model.fit(data, data, epochs=1, batch_size=100, callbacks=[lr_scheduler])
    model.save(path_to_save)
