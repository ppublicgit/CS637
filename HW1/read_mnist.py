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
