import time
import numpy as np
import tensorflow as tf
from IPython import display
from matplotlib import pyplot as plt

from self_supervised_loss import loss


@tf.function
def train_step(input_image, generator, generator_optimizer):
    with tf.GradientTape() as gen_tape:
        gen_output, lamb = generator(input_image, training=True)
        gen_loss = loss(lamb=lamb, prediction=gen_output, noisy=input_image)
    generator_gradients = gen_tape.gradient(gen_loss, generator.trainable_variables)
    generator_optimizer.apply_gradients(zip(generator_gradients, generator.trainable_variables))
    return gen_loss


def train(gen_model, gen_opt, inputs, inputs_valid, n_epochs,
          model_save_directory,
          loss_save_directory, x_valid):
    value_loss = np.zeros((n_epochs, 2), dtype=np.float32)
    count1 = 0

    for epoch in range(n_epochs):
        start = time.time()
        count = 1
        for x_train in zip(inputs):
            gen_loss = train_step(x_train, gen_model, gen_opt)
            value_loss[epoch, 0] = gen_loss.numpy() + value_loss[epoch, 0]
            count = count + 1

        value_loss[epoch, 0] = value_loss[epoch, 0] / count

        count = 1
        for x_valid in zip(inputs_valid):
            prediction_valid, lamb_valid = gen_model(x_valid, training=False)
            gen_loss = loss(lamb_valid, prediction_valid, x_valid)
            value_loss[epoch, 1] = gen_loss.numpy() + value_loss[epoch, 1]
            count = count + 1
        value_loss[epoch, 1] = value_loss[epoch, 1] / count

        display.clear_output(wait=True)

        current_lr = tf.keras.backend.eval(gen_model.optimizer.lr)
        print('learning rate:', current_lr)
        if np.remainder(count1 + 1, 10) == 0:
            if current_lr > 1e-7:
                update_lr = current_lr * 0.8
                tf.keras.backend.set_value(gen_model.optimizer.learning_rate, update_lr)

        if epoch == 0:
            gen_model.save_weights(model_save_directory, overwrite=True)
            print('model is saved')
        else:
            if value_loss[epoch, 1] <= np.min(value_loss[0:epoch, 1]):
                gen_model.save_weights(model_save_directory, overwrite=True)
                print('model is saved')
                count1 = 0
            else:
                count1 = count1 + 1
        if count1 == 100:
            print('Training is stopped')
            break

        np.savetxt(loss_save_directory, value_loss, delimiter=",")

        print('>%d, tloss[%.3f]  vloss[%.3f]' % (epoch + 1, value_loss[epoch, 0], value_loss[epoch, 1]))
        print('Time for epoch {} is {} sec'.format(epoch + 1, time.time() - start))

        ix = np.random.randint(0, len(x_valid))
        predictions, _ = gen_model(x_valid[ix:ix + 1], training=False)
        fig = plt.figure(fig_size=(20, 15))
        plt.subplot(1, 2, 1)
        plt.imshow(x_valid[ix, :, :, 0], cmap='magma')
        plt.axis('off')
        plt.subplot(1, 2, 2)
        plt.imshow(predictions[0, :, :, 0], cmap='magma')
        plt.axis('off')
        plt.show()
    return value_loss
