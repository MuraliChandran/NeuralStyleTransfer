import functools
import os

from matplotlib import gridspec
import matplotlib.pylab as plt
import tensorflow as tf
import tensorflow_hub as hub

print("TF Version: ", tf.__version__)
print("TF-Hub version: ", hub.__version__)
print("Eager mode enabled: ", tf.executing_eagerly())
print("GPU available: ", tf.test.is_gpu_available())

#Library way of loading images
def n_load_img(path_to_img):

      image_path = tf.keras.utils.get_file(os.path.basename(path_to_img)[-128:], path_to_img)
      max_dim = 512
      img = tf.io.read_file(image_path)
      img = tf.image.decode_image(img, channels=3)
      img = tf.image.convert_image_dtype(img, tf.float32)

      shape = tf.cast(tf.shape(img)[:-1], tf.float32)
      long_dim = max(shape)
      scale = max_dim / long_dim

      new_shape = tf.cast(shape * scale, tf.int32)

      img = tf.image.resize(img, new_shape)
      img = img[tf.newaxis, :]
      return img


def show_n(images, titles=('',)):
    n = len(images)
    image_sizes = [image.shape[1] for image in images]
    w = (image_sizes[0] * 6) // 320
    plt.figure(figsize=(w  * n, w))
    gs = gridspec.GridSpec(1, n, width_ratios=image_sizes)
    for i in range(n):
      plt.subplot(gs[i])
      plt.imshow(images[i][0], aspect='equal')
      plt.axis('off')
      plt.title(titles[i] if len(titles) > i else '')
    plt.show()

if __name__ =='__main__':

    content_image_url = 'https://upload.wikimedia.org/wikipedia/commons/0/00/Tuebingen_Neckarfront.jpg'  # @param {type:"string"}
    style_image_url = 'https://upload.wikimedia.org/wikipedia/commons/0/0a/The_Great_Wave_off_Kanagawa.jpg'  # @param {type:"string"}
    output_image_size = 384  # @param {type:"integer"}

    # The content image size can be arbitrary.
    content_img_size = (output_image_size, output_image_size)

    style_img_size = (256, 256)  # Recommended to keep it at 256.

    content_image = n_load_img(content_image_url)
    style_image = n_load_img(style_image_url)

    style_image = tf.nn.avg_pool(style_image, ksize=[3, 3], strides=[1, 1], padding='SAME')
    #show_n([content_image, style_image], ['Content image', 'Style image'])

    hub_handle = 'https://tfhub.dev/google/magenta/arbitrary-image-stylization-v1-256/2'
    hub_module = hub.load(hub_handle)

    outputs = hub_module(tf.constant(content_image), tf.constant(style_image))
    stylized_image = outputs[0]

    show_n([content_image, style_image, stylized_image],
           titles=['Original content image', 'Style image', 'Stylized image'])