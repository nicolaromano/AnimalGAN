import numpy as np
import tensorflow.keras as keras
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


class AnimalGAN:
    """
    A class to generate our GAN. This takes care of everything, including
    building the networks, training and prediction.
    """

    def __init__(self,
                 discriminator_input_shape: tuple,
                 generator_input_shape: tuple,
                 generator_output_shape: tuple,
                 discriminator_leakyReLU_alpha: float = 0.2,
                 discriminator_learning_rate: float = 0.0002,
                 generator_learning_rate: float = 0.0002,
                 latent_dim: int = 100):
        """
        Initialize the GAN.
        """

        self.latent_dim = latent_dim
        self.discriminator = self.__build_discriminator(discriminator_input_shape,
                                                        discriminator_leakyReLU_alpha,
                                                        discriminator_learning_rate)
        generator_initial_shape = tuple(
            self.discriminator.layers[-4].output.shape[1:])

        self.generator = self.__build_generator(latent_dim,
                                                generator_initial_shape)

        self.GAN = self.__build_GAN(self.generator, self.discriminator)

    def __build_generator(self, latent_dim, initial_shape):
        """
        Build the generator network.

        The generator network takes a latent vector as input and generates
        an image. 

        param latent_dim: The latent vector dimension.
        param initial_shape: The shape of the first feature layer.
        return: The generator network.
        """

        model = keras.Sequential()
        model.add(keras.layers.Dense(
            np.prod(initial_shape), input_dim=latent_dim))
        model.add(keras.layers.LeakyReLU(alpha=0.2))
        model.add(keras.layers.Reshape(initial_shape))
        # Scale up
        for i in range(3):
            model.add(keras.layers.Conv2DTranspose(
                int(initial_shape[2] / 2**i), (5, 5), strides=(2, 2), padding='same'))
            model.add(keras.layers.LeakyReLU(alpha=0.2))
        model.add(keras.layers.Conv2DTranspose(
            3, (5, 5), strides=(2, 2), padding='same', activation='tanh'))
        return model

    def __build_discriminator(self, input_shape: tuple,
                              leakyReLU_alpha: float = 0.2,
                              learning_rate: float = 0.0002) -> keras.Model:
        """
        Build the discriminator network.

        The network consists of four convolutional layers (Conv + MaxPool)
        followed by a fully connected layer. This discriminator is used to
        distinguish between real and fake images.

        LeakyReLU is used as the activation function for the convolutional
        layers.

        param input_shape: The shape of the input.
        param leakyReLU_alpha: The alpha value for the leaky ReLU.
        param learning_rate: The learning rate for the optimizer.
        return: The discriminator network.
        """

        model = keras.Sequential()

        model.add(keras.layers.Input(shape=input_shape))

        # First convolutional layer
        model.add(keras.layers.Conv2D(filters=64,
                                      kernel_size=(5, 5),
                                      padding='same'))
        model.add(keras.layers.LeakyReLU(alpha=leakyReLU_alpha))
        model.add(keras.layers.MaxPool2D(pool_size=(2, 2)))

        # Second convolutional layer
        model.add(keras.layers.Conv2D(filters=64,
                                      kernel_size=(5, 5),
                                      padding='same'))
        model.add(keras.layers.LeakyReLU(alpha=leakyReLU_alpha))
        model.add(keras.layers.MaxPool2D(pool_size=(2, 2)))

        # Third convolutional layer
        model.add(keras.layers.Conv2D(filters=128,
                                      kernel_size=(5, 5),
                                      padding='same'))
        model.add(keras.layers.LeakyReLU(alpha=leakyReLU_alpha))
        model.add(keras.layers.MaxPool2D(pool_size=(2, 2)))

        # Fourth convolutional layer
        model.add(keras.layers.Conv2D(filters=128,
                                      kernel_size=(5, 5),
                                      padding='same'))
        model.add(keras.layers.LeakyReLU(alpha=leakyReLU_alpha))
        model.add(keras.layers.MaxPool2D(pool_size=(2, 2)))

        model.add(keras.layers.Flatten())
        model.add(keras.layers.Dropout(0.4))
        model.add(keras.layers.Dense(1, activation='sigmoid'))

        opt = keras.optimizers.Adam(learning_rate=learning_rate, beta_1=0.5)
        model.compile(loss='binary_crossentropy',
                      optimizer=opt,
                      metrics=['accuracy'])

        return model

    def __build_GAN(self, generator, discriminator):
        """
        Build the GAN.

        The GAN is a combination of the generator and the discriminator.

        param generator: The generator network.
        param discriminator: The discriminator network.
        """

        discriminator.trainable = False
        model = keras.Sequential()
        model.add(generator)
        model.add(discriminator)
        return model

    def __latent_samples(self, num_samples: int):
        """
        Generate a latent vector.

        param n: The size of the latent vector.
        return: The latent vector.
        """

        return np.random.normal(size=(num_samples, self.latent_dim))
        
    
    def model_summary(self):
        """
        Print the model summary.
        """
        print(self.discriminator.summary())
        print(self.generator.summary())
        print(self.GAN.summary())

    def train(self):
        """
        Train the GAN.
        """
        pass

    def predict(self):
        """
        Predict the GAN.
        """
        pass
