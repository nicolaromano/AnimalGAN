import numpy as np
import matplotlib.pyplot as plt
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow.keras as keras
from tqdm import tqdm
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
        self.batch_size = 1
        self.train_ds = None  # This will be set by the train method.
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

        # This only freezes the weights of the discriminator in the GAN.
        # The discriminator is still trained separately, but won't be
        # affected when training the GAN.
        discriminator.trainable = False
        model = keras.Sequential()
        model.add(generator)
        model.add(discriminator)

        opt = keras.optimizers.Adam(learning_rate=0.0002, beta_1=0.5)
        model.compile(loss='binary_crossentropy', optimizer=opt)

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

    def __get_real_samples(self):
        """
        Gets a batch of real images.

        return: A tuple of (images, labels).
        """

        assert self.train_ds is not None, "No training dataset selected."

        # get a batch of real images
        images_batch = self.train_ds.take(1)

        images = None
    
        for image, _ in images_batch:
            images = image if images is None else np.concatenate((images, image))

        labels = np.ones((images.shape[0], 1))
        return images, labels

    def __get_fake_samples(self, n):
        """
        Gets fake images.

        param n: The number of images to get.
        return: A tuple of (images, labels).
        """

        # Generate a latent vector
        latent_vector = self.__latent_samples(n)
        # Generate fake images
        images = self.generator.predict(latent_vector)
        # Create labels for the images
        labels = np.zeros((n, 1))
        return images, labels

    def train(self, training_images_dir: str,
              epochs: int, batch_size: int = 50,
              output_filename: str = None,
              plot_loss: bool = True,
              save_images: bool = True) -> None:
        """
        Train the GAN.
        This first trains the discriminator, then trains the generator (by training the whole GAN, with the discriminator frozen).

        param: training_images_dir: The directory containing the training images.
        param: epochs: The number of epochs to train.
        param: batch_size: The batch size to use.
        param: output_filename: The filename to save the model to. Defaults to None.
        param: plot_loss: Whether to plot the loss (of the last batch in each epoch). Defaults to True.
        param: save_images: Whether to save an example image at the end of each epoch. Defaults to True.
        """

        self.batch_size = batch_size

        # Load the training images
        self.train_ds = keras.utils.image_dataset_from_directory(training_images_dir,
                                                                 batch_size=batch_size//2,
                                                                 shuffle=True)

        discriminator_loss_1 = np.zeros((epochs, 1))
        discriminator_loss_2 = np.zeros((epochs, 1))
        generator_loss = np.zeros((epochs, 1))

        for e in range(epochs):
            for b in tqdm(range(len(self.train_ds)), desc=f"Epoch {e+1}/{epochs}"):
                X_real, y_real = self.__get_real_samples()
                discriminator_loss_1[e], _ = self.discriminator.train_on_batch(
                    X_real, y_real)
                X_fake, y_fake = self.__get_fake_samples(batch_size // 2)
                discriminator_loss_2[e], _ = self.discriminator.train_on_batch(
                    X_fake, y_fake)
                X_gan = self.__latent_samples(batch_size)
                y_gan = np.ones((batch_size, 1))
                generator_loss[e] = self.GAN.train_on_batch(X_gan, y_gan)

            if save_images:
                image = self.generate_image()
                plt.imsave(f"GAN_output_epoch_{e}.png", image)

            if output_filename is not None:
                self.GAN.save(f"{output_filename}_epoch_{e+1}_GAN")
                self.generator.save(f"{output_filename}_epoch_{e+1}_generator")
        
        if plot_loss:
            plt.plot(self.discriminator_loss_1, label='Discriminator loss 1')
            plt.plot(self.discriminator_loss_2, label='Discriminator loss 2')
            plt.plot(self.generator_loss, label='Generator loss')
            plt.legend()

    def generate_image(self) -> np.array:
        """
        Generate an image.

        return: The generated image.
        """

        latent_vector = self.__latent_samples(1)
        return self.generator.predict(latent_vector)