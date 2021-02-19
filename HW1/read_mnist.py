import numpy as np


def read_labels(filename):
    with open(filename,'rb') as f:
        magic = f.read(4)
        magic = int.from_bytes(magic,'big')
        print("Magic is:", magic)

        nolab = f.read(4)
        nolab = int.from_bytes(nolab,'big')
        print("Num of labels is:", nolab)
        labels = [f.read(1) for i in range(nolab)]
        labels = np.array([int.from_bytes(label, 'big') for label in labels])
    return labels


def read_images(filename):
    with open(filename,'rb') as f:
        magic = f.read(4)
        magic = int.from_bytes(magic,'big')
        print("Magic is:", magic)

        noimg = f.read(4)
        noimg = int.from_bytes(noimg,'big')
        print("Number of images is:", noimg)

        norow = f.read(4)
        norow = int.from_bytes(norow,'big')
        print("Number of rows is:", norow)

        nocol = f.read(4)
        nocol = int.from_bytes(nocol,'big')
        print("Number of cols is:", nocol)

        images = []
        for i in range(noimg):
            rows = []
            for r in range(norow):
                cols = []
                for c in range(nocol):
                    cols.append(int.from_bytes(f.read(1), 'big'))
                rows.append(np.array(cols))
            images.append(np.array(rows))

    return images


def print_ascii_number(image):
    for row in image:
        for col in row:
            print('.' if col <= 127 else '#', end='')
        print()


def preprocess_images(images, percent=1):
    img_flat = [img.flatten() for img in images]
    img_norm = np.zeros((len(img_flat), img_flat[0].shape[0]), dtype=float)
    for i, img in enumerate(img_flat):
        #temp = img / 255
        #temp.reshape(len(temp), 1)
        img_norm[i, :] = img / 255
    if percent < 1 and percent > 0:
        subindex = int(percent * img_norm.shape[0])
        img_norm = img_norm[:subindex, :]
    elif percent == 1:
        pass
    else:
        raise ValueError(f"Percent must be set to a value in range (0, 1], not {percent}")
    return img_norm


def preprocess_labels(arr, percent=1):
    size = len(np.unique(arr))
    ohe = np.zeros((len(arr), size), dtype=int)
    for i, val in enumerate(arr):
        ohe[i, val] = 1
    if percent < 1 and percent > 0:
        subindex = int(percent * ohe.shape[0])
        ohe = ohe[:subindex, :]
    elif percent == 1:
        pass
    else:
        raise ValueError(f"Percent must be set to a value in range (0, 1], not {percent}")
    return ohe
