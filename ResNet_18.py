"""This is a simple example for building ML model demos using streamlit.

Streamlit is required to run this file. For more info check go/streamlit-doc
"""


import numpy as np
import pandas as pd
from PIL import Image
import requests
import streamlit as st
import torch
from torchvision import transforms


# Function to output Pytorch predictions
def predict(inp):
  inp = transforms.ToTensor()(inp).unsqueeze(0)
  with torch.no_grad():
    prediction = torch.nn.functional.softmax(model(inp)[0], dim=0)
    confidences = {labels[i]: float(prediction[i]) for i in range(1000)}
  return confidences


if __name__ == "__main__":
  model = torch.hub.load(
      "pytorch/vision:v0.6.0", "resnet18", pretrained=True).eval()
  response = requests.get("https://git.io/JJkYN")
  labels = response.text.split("\n")

  st.set_page_config(
      page_title="Demo ResNet18",
      layout="wide",
      page_icon="https://pytorch.org/assets/images/pytorch-logo.png")

  st.title("Deep Residual Neural Networks (ResNets)")
  st.markdown("[Kaiming He](https://scholar.google.com/citations?user=DhtAFkwAAAAJ&hl=en), 2015")

  with st.expander("Learn more about the model"):
    st.markdown("""
    ### Summary
    **Residual Networks**, or **ResNets**, learn residual functions with reference 
    to the layer inputs, instead of learning unreferenced functions. Instead 
    of hoping each few stacked layers directly fit a desired underlying mapping, 
    residual nets let these layers fit a residual mapping. They stack residual 
    blocks ontop of each other to form network: e.g. a ResNet-18 has eighteen layers using these blocks.
    
    ### ResNet-18 Description
    ResNet-18 is a convolutional neural network that is 18 layers deep. You can 
    load a pretrained version of the network trained on more than a million images 
    from the ImageNet database [1]. The pretrained network can classify images into 
    1000 object categories, such as keyboard, mouse, pencil, and many animals. As a result, 
    the network has learned rich feature representations for a wide range of images. The network 
    has an image input size of 224-by-224.
      """)

    st.latex(r"\sigma(z_i) = \frac{e^{z_{i}}}{\sum_{j=1}^K e^{z_{j}}} \ \ \ for\ i=1,2,\dots,K")
    st.markdown("Above is the formula for the Softmax activation function even though it's not used here to show how you can easily use LaTeX with Streamlit. Below is another random example:")
    st.latex(r"""JS(\hat{y} || y) = \frac{1}{2}(KL(y||\frac{y+\hat{y}}{2}) + KL(\hat{y}||\frac{y+\hat{y}}{2}))""")
    st.text("")

    col1, col2, col3, col4 = st.columns([1,1,1,1]) # The list indicates the size proportion of each column
    if col1.button("Show Paramaters"):
      st.image("https://www.researchgate.net/publication/343249978/figure/tbl1/AS:918089503870976@1595901003064/Architecture-and-parameters-of-ResNet-18.png")
    if col2.button("Show Architecture"):
      st.image("https://blog.it-logix.ch/wp-content/uploads/2021/01/resnet18.png")
    if col4.button("Hide"):
      st.text("")
    if col3.button("Show Code"):
      rescode = """def resnet(x, blocks_per_layer, num_classes=1000):
    x = layers.ZeroPadding2D(padding=3, name='conv1_pad')(x)
    x = layers.Conv2D(filters=64, kernel_size=7, strides=2, use_bias=False, kernel_initializer=kaiming_normal, name='conv1')(x)
    x = layers.BatchNormalization(momentum=0.9, epsilon=1e-5, name='bn1')(x)
    x = layers.ReLU(name='relu1')(x)
    x = layers.ZeroPadding2D(padding=1, name='maxpool_pad')(x)
    x = layers.MaxPool2D(pool_size=3, strides=2, name='maxpool')(x)

    x = make_layer(x, 64, blocks_per_layer[0], name='layer1')
    x = make_layer(x, 128, blocks_per_layer[1], stride=2, name='layer2')
    x = make_layer(x, 256, blocks_per_layer[2], stride=2, name='layer3')
    x = make_layer(x, 512, blocks_per_layer[3], stride=2, name='layer4')

    x = layers.GlobalAveragePooling2D(name='avgpool')(x)
    initializer = keras.initializers.RandomUniform(-1.0 / math.sqrt(512), 1.0 / math.sqrt(512))
    x = layers.Dense(units=num_classes, kernel_initializer=initializer, bias_initializer=initializer, name='fc')(x)

    return x"""
      st.code(rescode, language="python")

  st.subheader("Want to see Resnet18 in action?")
  file = st.file_uploader("Upload an image to see PyTorch pre-trained model in action.", type=["jpg","png"])

  if file:
    img = Image.open(file)

    # Resizing the image
    basewidth = 250
    wpercent = (basewidth / float(img.size[0]))
    hsize = int((float(img.size[1]) * float(wpercent)))
    img = img.resize((basewidth, hsize), Image.ANTIALIAS)

    left, right = st.columns(2)
    left.markdown("### Here is the image you've selected")
    left.image(img)

    pred = predict(img)

    # Get items from the prediction dictionary and create dataframe with them
    df = pd.DataFrame(pred.items(), columns=["Label", "Confidence Level"])
    print(np.shape(df))

    # Sort values so that I can print the name of the one with > confidence lvl
    df = df.sort_values(by=["Confidence Level"], ascending=False)
    df = df.reset_index()

    obj = str(df["Label"][0])
    right.markdown("### The model thinks there's a " + obj + " in your pic")

    # Create a slider that stores number of labels the user sees
    n = right.slider(
        "How many labels do you want to see?",
        min_value=1,
        max_value=10,
        value=2,
        disabled=False)

    show = df.head(n)
    show["Confidence Level"] = show["Confidence Level"] * 100
    show["Confidence Level"] = show["Confidence Level"].round(decimals=2)

    right.write(show[["Label", "Confidence Level"]])
