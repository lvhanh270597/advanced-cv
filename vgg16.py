from keras.applications.vgg16 import VGG16, preprocess_input
from keras.models import Model

# VGG16 model with default parameters, which includes the 3 fully-connected
# layers at the top of the network and uses pre-trained weights on ImageNet
model = VGG16()

# Intermediate layer model, which is the same as VGG16 model except that it
# uses the first fully-connected layers as its output
model_fc6 = Model(inputs=model.input,
                  outputs=model.get_layer('fc1').output)


def extract_fc6_features(images, verbose=False):
    """Extracts image features using VGG16 model.

    Args:
        images: list of images of shape (224, 224, 3)
        verbose: whether to output progress

    Returns:
        A 4096-dimensional vector for each of the images
    """

    global model_fc6

    # From Keras docstring:
    # ...
    # mode: One of "caffe", "tf" or "torch".
    # - caffe: will convert the images from RGB to BGR,
    #     then will zero-center each color channel with
    #     respect to the ImageNet dataset,
    #     without scaling.
    # - tf: will scale pixels between -1 and 1,
    #     sample-wise.
    # - torch: will scale pixels between 0 and 1 and then
    #     will normalize each channel with respect to the
    #     ImageNet dataset.
    X = preprocess_input(images, mode='caffe')

    if verbose:
        return model_fc6.predict(X, batch_size=1, verbose=1)

    return model_fc6.predict(X)
