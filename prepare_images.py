# Resizes and pads the images to 256x256x3
# Original images are from https://www.kaggle.com/iamsouravbanerjee/animal-image-dataset-90-different-animals

import os
import matplotlib.pyplot as plt
from skimage.transform import resize
from numpy import pad
from tqdm import tqdm

original_path = "animal_images_original/"
output_path = "images/"

output_size = 256

# Read the original images
for subdir in os.listdir(original_path):
    print(f"Processing images in {subdir}")
    # Create output directory
    output_subdir = os.path.join(output_path, subdir)
    if not os.path.exists(output_subdir):
        os.makedirs(output_subdir)

    for filename in tqdm(os.listdir(original_path + subdir)):
        # Read the image
        image = plt.imread(original_path + subdir + "/" + filename)
        # Resize the image and pad to square
        if (image.shape[0] > image.shape[1]):
            image = resize(image, (output_size, int(image.shape[1]/image.shape[0]*output_size)))
        else:
            image = resize(image, (int(image.shape[0]/image.shape[1]*output_size), output_size))

        # Pad the image to square
        xpad1 = (output_size - image.shape[0])//2
        xpad2 = output_size - image.shape[0] - xpad1
        ypad1 = (output_size - image.shape[1])//2
        ypad2 = output_size - image.shape[1] - ypad1
        image = pad(image, ((xpad1, xpad2), (ypad1, ypad2), (0,0)), mode='constant', constant_values=0)

        # Save the image
        plt.imsave(output_path + subdir + "/" + filename, image)



