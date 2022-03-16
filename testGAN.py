import AnimalGAN as GAN

test = GAN.AnimalGAN(input_shape=(64, 64, 3),
                training_images_dir="images_small",
                latent_dim=100,
                verbose=True)

test.train(epochs=10, batch_size=256,
           output_dir="out", model_name="testGAN")