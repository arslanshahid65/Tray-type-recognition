import numpy as np
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img

datagen = ImageDataGenerator(
    rotation_range = 90,
    width_shift_range = 0.2,
    height_shift_range = 0.2,
    brightness_range = [0.5, 1.5],
    shear_range = 0.2,
    zoom_range = 0.1,
    fill_mode = 'nearest'
)

image = load_img('image_name.jpg') #loaded from default directory
image_array = img_to_array(image)
print(np.shape(image_array))
image_array = image_array.reshape((1,) + image_array.shape)
print(np.shape(image_array))

i = 0

for batch in datagen.flow(image_array, batch_size=1, save_to_dir='Directory_name', save_prefix = 'Round trans 3', save_format = 'jpeg'):  
    #save_to_dir in a default directory
    i += 1
    if i > 100:
        break   