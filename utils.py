import os

import numpy as np
from PIL import Image


def read_process_images(dir, labels, return_paths=False):
    """Prepares data to feed VGG16 model.

    Uses labels as the name of the directories containing the images.
    Converts each of the images to have the shape of 224x224x3, which is
    required by the (default) VGG16 model.

    Args:
        dir: where to read images from
        labels: list of image labels
        return_paths: whether to return file paths

    Returns:
        X, y: converted dataset
        paths: image file paths if return_paths is True
    """

    X = []
    y = []
    paths = []

    for i, label in enumerate(labels):
        for fn in os.listdir(os.path.join(dir, label)):
            fp = os.path.join(dir, label, fn)
            img = Image.open(fp)

            # Ensure 224x224x3
            assert len(img.getbands()) == 3
            img = img.resize((224, 224))

            X.append(np.array(img))
            y.append(i)
            paths.append(fp)

    if return_paths:
        return np.array(X), np.array(y), np.array(paths)

    return np.array(X), np.array(y)
