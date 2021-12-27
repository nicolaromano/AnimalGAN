import tensorflow.keras as keras
from tensorflow.python.keras.layers import convolutional

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
                generator_learning_rate: float = 0.0002):
        """
        Initialize the GAN.
        """
        
        self.discriminator = self.build_discriminator(discriminator_input_shape,
                                discriminator_leakyReLU_alpha,
                                discriminator_learning_rate)
        # self.generator = self.build_generator()
        # self.GAN = self.build_GAN(self.generator, self.discriminator)

    def build_generator(self):
        """
        Build the generator network.
        """
        pass

    def build_discriminator(self, input_shape: tuple, 
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
        model.add(keras.layers.Conv2D(filters = 64, 
                    kernel_size = (5,5), 
                    padding='same'))
        model.add(keras.layers.LeakyReLU(alpha=leakyReLU_alpha))
        model.add(keras.layers.MaxPool2D(pool_size=(2,2)))

        # Second convolutional layer
        model.add(keras.layers.Conv2D(filters = 64, 
                    kernel_size = (5,5), 
                    padding='same'))
        model.add(keras.layers.LeakyReLU(alpha=leakyReLU_alpha))
        model.add(keras.layers.MaxPool2D(pool_size=(2,2)))

        # Third convolutional layer
        model.add(keras.layers.Conv2D(filters = 128, 
                    kernel_size = (5,5), 
                    padding='same'))
        model.add(keras.layers.LeakyReLU(alpha=leakyReLU_alpha))
        model.add(keras.layers.MaxPool2D(pool_size=(2,2)))

        # Fourth convolutional layer
        model.add(keras.layers.Conv2D(filters = 128, 
                    kernel_size = (5,5), 
                    padding='same'))
        model.add(keras.layers.LeakyReLU(alpha=leakyReLU_alpha))
        model.add(keras.layers.MaxPool2D(pool_size=(2,2)))

        model.add(keras.layers.Flatten())
        model.add(keras.layers.Dropout(0.4))
        model.add(keras.layers.Dense(1, activation='sigmoid'))
    
        opt = keras.optimizers.Adam(lr=learning_rate, beta_1=0.5)
        model.compile(loss='binary_crossentropy', 
                    optimizer=opt, 
                    metrics=['accuracy'])
	    
        return model

    def model_summary(self):
        """
        Print the model summary.
        """
        print(self.discriminator.summary())
        # print(self.generator.summary())
        # print(self.GAN.summary())

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

