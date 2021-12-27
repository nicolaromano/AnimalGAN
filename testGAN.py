import AnimalGAN as GAN


test = GAN.AnimalGAN(discriminator_input_shape = (128, 128, 3),
                     generator_input_shape = (128, 128, 3),
                     generator_output_shape= (128, 128, 3))

test.model_summary()