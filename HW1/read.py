from PIL import Image
import numpy as np


def read_labels_from_file(filename):
    with open(filename,'rb') as f: #use gzip to open the file in read binary mode
        magic = f.read(4) # magic number is the first 4 bytes
        magic = int.from_bytes(magic,'big') # Convert bytes to integers.
        print("Magic is:", magic) # print to console

        # the same as above but with labels
        nolab = f.read(4)
        nolab = int.from_bytes(nolab,'big')
        print("Num of labels is:", nolab)
        # for looping through labels
        labels = [f.read(1) for i in range(nolab)]
        labels = [int.from_bytes(label, 'big') for label in labels]
    return labels
print()
train_labels = read_labels_from_file("mnist/train-labels-idx1-ubyte")
test_labels = read_labels_from_file("mnist/t10k-labels-idx1-ubyte")

def read_images_from_file(filename):
    with open(filename,'rb') as f:
        magic = f.read(4)
        magic = int.from_bytes(magic,'big')
        print("Magic is:", magic)

        # Number of images in next 4 bytes
        noimg = f.read(4)
        noimg = int.from_bytes(noimg,'big')
        print("Number of images is:", noimg)

        # Number of rows in next 4 bytes
        norow = f.read(4)
        norow = int.from_bytes(norow,'big')
        print("Number of rows is:", norow)

        # Number of columns in next 4 bytes
        nocol = f.read(4)
        nocol = int.from_bytes(nocol,'big')
        print("Number of cols is:", nocol)

        images = [] # create array
        #for loop
        for i in range(noimg):
            rows = []
            for r in range(norow):
                cols = []
                for c in range(nocol):
                    cols.append(int.from_bytes(f.read(1), 'big')) # append the current byte for every column
                rows.append(cols) # append columns array for every row
            images.append(rows) # append rows for every image
    return images
print() #line break

# Call the functions and run them to read the files
train_images = read_images_from_file("mnist/train-images-idx3-ubyte")
test_images = read_images_from_file("mnist/t10k-images-idx3-ubyte")

# Question 2 - Output an image to the console
# Output the third image in the training set to the console.
# Do this by representing any pixel value less than 128 as a full stop and any other pixel value as a hash symbol.
for row in train_images[4999]:
    for col in row:
        print('.' if col <= 127 else '#', end='')
    print()

# Question 3 - Output the image files as PNGs
# Download the image and label files.
# Have Python decompress and read them byte by byte into appropriate data structures in memory.
img = Image.fromarray(np.array(train_images[4999]).astype('uint8'))
img = img.convert('RGB') # convert into rgb format
img.show() # display image in window
img.save('train-4999-2.png') # save the image file as png
