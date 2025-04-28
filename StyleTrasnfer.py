from pathlib import Path
import requests
import os
import random
from PIL import Image
import matplotlib.pyplot as plt
from tqdm.auto import tqdm

# useful files
files = {
    "images.zip":"https://github.com/Q-b1t/nst_demo_resources/raw/main/images/images.zip",
}
# create the import directory
imports_path = Path("imports")
imports_path.mkdir(exist_ok=True,parents=True)
paths = dict()

# import the files
for f,raw in files.items():
  new_path = imports_path / f
  req = requests.get(raw)
  with open(new_path,"wb") as fl:
    fl.write(req.content)
    print(f"File {raw} written to {new_path},")
    paths[f] = new_path



# get the image paths as a list
image_path = Path("images")
style_path = image_path / "style"
content_path = image_path / "content"

style_images = list(style_path.glob("*.jpg"))
content_images = list(content_path.glob("*.jpg"))



style_image = random.choice(style_images)
content_image = random.choice(content_images)

plt.figure(figsize = (10,5))
plt.subplot(1,2,1)
plt.title("Content Image")
plt.imshow(plt.imread(content_image))
plt.axis(False)
plt.subplot(1,2,2)
plt.title("Style Image")
plt.imshow(plt.imread(style_image))
plt.axis(False)

import numpy as np
import tensorflow as tf


# compute the gram matrix of the features
def gram_matrix(x):
  #x = tf.convert_to_tensor(x, tf.int32)
  x = tf.transpose(x,perm = (2,0,1))
  features = tf.reshape(x,(tf.shape(x)[0],-1))
  gram = tf.matmul(features,tf.transpose(features))
  return gram

# compute the style cost function
def style_cost_function(style_image,generated_image):
  S = gram_matrix(style_image)
  C = gram_matrix(generated_image)
  channels = 3
  size = img_nrows * img_ncols
  return tf.reduce_sum(tf.square(S - C)) / (4.0 * (channels ** 2) * (size ** 2))

def content_cost_function(base_image,generated_image):
  return tf.reduce_sum(tf.square(tf.subtract(generated_image,base_image)))


### TF model 

from tensorflow.keras.models import Model
from tensorflow.keras.applications import VGG19
from tensorflow.keras.applications.vgg19 import preprocess_input
from tensorflow.keras.preprocessing.image import load_img,img_to_array
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.optimizers.schedules import ExponentialDecay

pretrained_vgg_model = VGG19(include_top = False, weights="imagenet")

pretrained_vgg_model.summary()

# extract the layers' name (for reference only)
model_layers = [layer.name for layer in pretrained_vgg_model.layers]
model_layers
# create a dictionary that maps the layers name to each feature extractor
model_outputs = {layer.name : layer.output for layer in pretrained_vgg_model.layers}
# feature extractor
feature_extractor = Model(inputs=pretrained_vgg_model.inputs, outputs=model_outputs)

# define the style layers
style_layers = [
    "block1_conv1",
    "block2_conv1",
    "block3_conv1",
    "block4_conv1",
    "block5_conv1",
    ]
# define the content layer
content_layer = "block5_conv2"

# define the style and content weights
content_weight = 2.5e-8
style_weight = 1.0e-6

def loss_function(combination_image, base_image, style_reference_image):
  # 1. Combine all the images in the same tensioner.
  input_tensor = tf.concat(
      [base_image, style_reference_image, combination_image], axis=0
  )

  # 2. Get the values in all the layers for the three images.
  features = feature_extractor(input_tensor)

  #3. Inicializar the loss

  loss = tf.zeros(shape=())

  # 4. Extract the content layers + content loss
  layer_features = features[content_layer]
  base_image_features = layer_features[0, :, :, :]
  combination_features = layer_features[2, :, :, :]

  loss = loss + content_weight * content_cost_function(
      base_image_features, combination_features
  )
  # 5. Extraer the style layers + style loss
  for layer_name in style_layers:
      layer_features = features[layer_name]
      style_reference_features = layer_features[1, :, :, :]
      combination_features = layer_features[2, :, :, :]
      sl = style_cost_function(style_reference_features, combination_features)
      loss += (style_weight / len(style_layers)) * sl

  return loss

@tf.function
def compute_loss_and_grads(generated_image, base_image, style_image):
    with tf.GradientTape() as tape:
        loss = loss_function(generated_image, base_image, style_image)
    grads = tape.gradient(loss, generated_image)
    return loss, grads

    """
def preprocess_image(image_path):
  # load the image into a tensot
  img = load_img(image_path, target_size=(img_nrows, img_ncols))
  # turn the image into a numpy array
  img = img_to_array(img)
  # add a batch dimention
  img = np.expand_dims(img, axis=0)
  # preprocess according to the vgg model's specification
  img = preprocess_input(img)
  return tf.convert_to_tensor(img)

"""


def preprocess_image(img):
  # load the image into a tensot
  img = img.resize((img_ncols,img_nrows))
  # turn the image into a numpy array
  img = img_to_array(img)
  # add a batch dimention
  img = np.expand_dims(img, axis=0)
  # preprocess according to the vgg model's specification
  img = preprocess_input(img)
  return tf.convert_to_tensor(img)


def deprocess_image(x):
    # Cconert to array
    x = x.reshape((img_nrows, img_ncols, 3))
    # mean = 0
    x[:, :, 0] += 103.939
    x[:, :, 1] += 116.779
    x[:, :, 2] += 123.68
    # convert to rgb
    x = x[:, :, ::-1]
    # normalize
    x = np.clip(x, 0, 255).astype("uint8")
    return x

# store the loss curve values
loss_values = list()

from PIL import Image, ImageChops 
from diffusers.utils import load_image
## test 1 image 
image_path = "/home/muzzi/Image2image generation/Anomaly detection images/customer data/1 (402)_insulator_4.jpg"
style_path = "/home/muzzi/Image2image generation/edges.jpg"

content_image = load_image(image_path).convert('RGB')

style_image = load_image(style_path).convert('RGB')

base_pil_image = Image.open(image_path)
generated_pil_image = Image.open(image_path)
style_pil_image = Image.open(style_path)

width, height = base_pil_image.size


img_nrows = 400
img_ncols = int(width * img_nrows / height)

optimizer = SGD(
    ExponentialDecay(
        initial_learning_rate=100.0, decay_steps=100, decay_rate=0.96
    )
)

base_image = preprocess_image(base_pil_image)
style_reference_image = preprocess_image(style_pil_image)
combination_image = tf.Variable(preprocess_image(generated_pil_image))

print(base_image.shape,style_reference_image.shape,combination_image.shape)

epochs = 4

for i in tqdm(range(epochs)):
    loss, grads = compute_loss_and_grads(
        combination_image, base_image, style_reference_image
    )
    optimizer.apply_gradients([(grads, combination_image)])
    if i % 100 == 0:
        print(f"ITERATION: {i} | LOSS: {loss:.3f}")
    loss_values.append(loss)

plt.figure()
plt.plot(loss_values)
plt.title("Loss Curve")
plt.ylabel("Loss")
plt.xlabel("Epochs")


generated_image_numpy = combination_image.numpy()
generated_image_numpy = deprocess_image(generated_image_numpy)
#generated_image_numpy = np.squeeze(generated_image_numpy,axis = 0)
generated_image_numpy.shape

plt.imshow(generated_image_numpy)
plt.axis(False)
plt.savefig("generated_image.png")