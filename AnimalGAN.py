import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import matplotlib.pyplot as plt
import numpy as np
import tensorflow.keras as keras
from tqdm import tqdm
import shutil


class AnimalGAN:
    """
    A class to generate our GAN. This takes care of everything, including
    building the networks, training and prediction.
    """

    def __init__(self,
                 input_shape: tuple,
                 training_images_dir: str,
                 batch_size: int = 32,
                 latent_dim: int = 100):
        """
        Initialize the GAN.

        param input_shape: The shape of the input.
        param: training_images_dir: The directory containing the training images.
        param: batch_size: The batch size to use. Default is 32.
        param latent_dim: The latent vector dimension. Default is 100.
        """

        self.latent_dim = latent_dim
        self.batch_size = batch_size

        self.image_size = (input_shape[0], input_shape[1])
        # Load the training images
        self.train_ds = keras.utils.image_dataset_from_directory(training_images_dir,
                                                                 image_size=self.image_size,
                                                                 batch_size=batch_size//2,
                                                                 shuffle=True)

        self.num_classes = len(self.train_ds.class_names)

        self.discriminator = self.__build_discriminator(input_shape)
        generator_initial_shape = tuple(
            self.discriminator.layers[-4].output.shape[1:])

        self.generator = self.__build_generator(latent_dim,
                                                generator_initial_shape)

        self.GAN = self.__build_GAN(self.generator, self.discriminator)

    def __build_generator(self, latent_dim, initial_shape):
        """
        Build the conditional generator network.

        The generator network takes a latent vector and a label as inputs and
        outputs an image.

        param latent_dim: The latent vector dimension.
        param initial_shape: The shape of the first feature layer.
        return: The generator network.
        """

        # The label of our class
        label_input = keras.layers.Input(shape=(1,), name='label_input_gen')
        # Put the label into an embedding layer
        embedding = keras.layers.Embedding(
            self.num_classes, latent_dim)(label_input)
        dense = keras.layers.Dense(
            initial_shape[0]*initial_shape[1])(embedding)
        dense = keras.layers.Reshape(
            (initial_shape[0], initial_shape[1], 1))(dense)

        # Concatenate the label with the latent vector
        latent_input = keras.layers.Input(
            shape=(latent_dim,), name='latent_input_gen')

        dense2 = keras.layers.Dense(np.prod(initial_shape))(latent_input)
        dense2 = keras.layers.LeakyReLU(alpha=0.2)(dense2)
        dense2 = keras.layers.Reshape(initial_shape)(dense2)

        concatenated = keras.layers.concatenate([dense, dense2])

        print("Generator input shape (concatenated): ", concatenated.shape)

        # Upsample using Conv2DTranspose
        convtr1 = keras.layers.Conv2DTranspose(
            filters=64, kernel_size=2, strides=2, padding='same')(concatenated)
        print("Conv 1: ", convtr1.shape)
        convtr2 = keras.layers.Conv2DTranspose(
            filters=32, kernel_size=2, strides=2, padding='same', activation='sigmoid')(convtr1)
        print("Conv 2: ", convtr2.shape)
        convtr3 = keras.layers.Conv2DTranspose(
            filters=16, kernel_size=2, strides=2, padding='same', activation='sigmoid')(convtr2)
        print("Conv 3: ", convtr3.shape)

        # Return the model
        return keras.models.Model(inputs=[latent_input, label_input], outputs=convtr3)

    def __build_discriminator(self, input_shape: tuple) -> keras.Model:
        """
        Build the conditional discriminator network. 

        This accepts a label and an image as input and outputs a single value.
        The network consists of four convolutional layers (Conv + MaxPool)
        followed by a fully connected layer. This discriminator is used to
        distinguish between real and fake images.

        LeakyReLU is used as the activation function for the convolutional
        layers.

        param input_shape: The shape of the input.
        return: The discriminator network.
        """
        # The label of our class
        label_input = keras.layers.Input(shape=(1,), name='label_input')
        # Put the label into an embedding layer
        embedding = keras.layers.Embedding(
            self.num_classes, self.latent_dim)(label_input)
        dense = keras.layers.Dense(
            input_shape[0] * input_shape[1])(embedding)
        dense = keras.layers.Reshape(
            (input_shape[0], input_shape[1], 1))(dense)
        # Concatenate the label with the image
        model_input = keras.layers.Input(shape=input_shape)
        concat = keras.layers.Concatenate()([model_input, dense])
        # First convolutional layer
        conv1 = keras.layers.Conv2D(
            64, (5, 5), padding='same',
            name='discr_conv1')(concat)
        conv1 = keras.layers.LeakyReLU(alpha=0.2)(conv1)
        print(f"Conv1 - {conv1.shape}")
        pool1 = keras.layers.MaxPooling2D((2, 2),
                                          name='discr_pool1')(conv1)
        print(f"Pool1 - {pool1.shape}")
        # Second convolutional layer
        conv2 = keras.layers.Conv2D(
            64, (3, 3), padding='same',
            name='discr_conv2')(pool1)
        conv2 = keras.layers.LeakyReLU(alpha=0.2)(conv2)
        print(f"Conv2 - {conv2.shape}")
        pool2 = keras.layers.MaxPooling2D((2, 2), name='discr_pool2')(conv2)
        print(f"Pool2 - {pool2.shape}")
        # Third convolutional layer
        conv3 = keras.layers.Conv2D(
            128, (3, 3), padding='same',
            name='discr_conv3')(pool2)
        conv3 = keras.layers.LeakyReLU(alpha=0.2)(conv3)
        print(f"Conv3 - {conv3.shape}")
        pool3 = keras.layers.MaxPooling2D((2, 2), strides=(2, 2),
                                          name='discr_pool3')(conv3)
        print(f"Pool3 - {pool3.shape}")
        # # Fourth convolutional layer
        # conv4 = keras.layers.Conv2D(
        #     128, (3, 3), padding='same', name = 'discr_conv4')(pool3)
        # conv4 = keras.layers.LeakyReLU(alpha=0.2)(conv4)
        # print(f"Conv4 - {conv4.shape}")
        # pool4 = keras.layers.MaxPooling2D((2, 2), strides=(2, 2),
        #     name = 'discr_pool4')(conv4)
        # print(f"Pool4 - {pool4.shape}")
        # Flatten the output
        flat = keras.layers.Flatten()(pool3)
        dropout = keras.layers.Dropout(0.4)(flat)
        # Fully connected layer
        out = keras.layers.Dense(1, activation='sigmoid')(flat)

        model = keras.Model(inputs=[model_input, label_input], outputs=out)
        opt = keras.optimizers.Adam(learning_rate=0.0002, beta_1=0.5)
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

        gen_noise, gen_label = generator.input
        gen_output = generator.output

        model = keras.Model(
            inputs=[gen_noise, gen_label], outputs=gen_output)

        opt = keras.optimizers.Adam(learning_rate=0.0002, beta_1=0.5)
        model.compile(loss='binary_crossentropy', optimizer=opt)

        return model

    def __latent_samples(self, num_samples: int) -> np.ndarray:
        """
        Generate a latent vector and random classes

        param num_samples: The number of samples to generate.
        return: The latent vector.
        """
        return [np.random.normal(size=(num_samples, self.latent_dim)),
                np.random.randint(0, self.num_classes, size=(num_samples, 1))]

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

        return: A tuple of ([images, class_label], 1).
        """

        assert self.train_ds is not None, "No training dataset selected."

        # get a batch of real images
        images_batch = self.train_ds.take(1)

        images = None
        labels = None

        for image, label in images_batch:
            image = image / 255.0
            images = image if images is None else np.concatenate(
                (images, image))
            labels = label if labels is None else np.concatenate(
                (labels, label))

        # These are real images, so they are coded as 1. We are adding some noise (random number between 0.9 and 1)
        y = np.random.random(size=(images.shape[0], 1)) * 0.1 + 0.9
        return [images, labels], y

    def __get_fake_samples(self, n):
        """
        Gets fake images.

        param n: The number of images to get.
        return: A tuple of (images, labels).
        """

        # Generate a latent vector
        latent_vector, class_labels = self.__latent_samples(n)
        # Generate fake images
        images = self.generator.predict(latent_vector)
        # These are fake images, so they are coded as 0.
        # We are adding some noise (random number between 0 and 0.1)
        y = np.random.random(size=(n, 1)) * 0.1
        return [images, class_labels], y

    def train(self,
              epochs: int, batch_size: int = 50,
              output_dir: str = './output',
              continue_training: bool = False,
              starting_epoch: int = 0,
              saved_model_name: str = None,
              model_name: str = None,
              save_images: bool = True) -> None:
        """
        Train the GAN.
        This first trains the discriminator, then trains the generator (by training the whole GAN, with the discriminator frozen).

        param: epochs: The number of epochs to train.
        param: output_dir: The directory to save the output (images and models).
        param: continue_training: Whether to continue training from a previous model.
        param: starting_epoch: The epoch to start training from (this is just for the file name)
        param: saved_model_name: The name of the saved model to restart training from. Only used if continue_training is True. Should be in output_dir.
        param: model_name: The name of the model. This will have the current epoch appended to it. Only the last model will be kept.
        param: save_images: Whether to save an example image at the end of each epoch. Defaults to True.
        """

        if continue_training:
            # Load the saved models
            self.generator = keras.models.load_model(
                os.path.join(output_dir, saved_model_name) + '_generator')
            self.GAN = keras.models.load_model(
                os.path.join(output_dir, saved_model_name) + '_GAN')
            # Extract the discriminator from the GAN
            self.discriminator = self.GAN.layers[-1]
            self.discriminator.trainable = True

        for e in range(starting_epoch, starting_epoch + epochs):
            print(f"Epoch {e+1}/{epochs}")
            for _ in range(len(self.train_ds)):
                [X_real, labels_real], y_real = self.__get_real_samples()
                discriminator_loss_1, _ = self.discriminator.train_on_batch(
                    [X_real, labels_real], y_real)
                [X_fake, labels_fake], y_fake = self.__get_fake_samples(
                    batch_size // 2)
                discriminator_loss_2, _ = self.discriminator.train_on_batch(
                    X_fake, y_fake)
                [X_gan, labels_gan] = self.__latent_samples(batch_size)
                y_gan = np.random.random(size=(batch_size, 1)) * 0.1 + 0.9
                generator_loss = self.GAN.train_on_batch(
                    [X_gan, labels_gan], y_gan)
                print(
                    f"Discriminator loss: {discriminator_loss_1}/{discriminator_loss_2}")
                print(f"Generator loss: {generator_loss}")

            if save_images:
                image = self.generate_image()
                plt.imsave(f"{output_dir}/GAN_output_epoch_{e}.png", image)

            if output_dir is not None and model_name is not None:
                self.GAN.save(f"{output_dir}/{model_name}_epoch_{e+1}_GAN")
                self.generator.save(
                    f"{output_dir}/{model_name}_epoch_{e+1}_generator")
                # Delete the previous model, if it exists
                if os.path.exists(f"{output_dir}/{model_name}_epoch_{e}_GAN"):
                    shutil.rmtree(f"{output_dir}/{model_name}_epoch_{e}_GAN")
                if os.path.exists(f"{output_dir}/{model_name}_epoch_{e}_generator"):
                    shutil.rmtree(
                        f"{output_dir}/{model_name}_epoch_{e}_generator")

    def generate_image(self) -> np.array:
        """
        Generate an image.

        return: The generated image.
        """

        latent_vector = self.__latent_samples(1)
        return self.generator.predict(latent_vector)[0]
