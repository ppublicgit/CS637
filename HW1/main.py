import numpy as np
import time

import mlp
import read_mnist


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


def main():
    train_images = read_mnist.read_images("mnist/train_images")
    test_images  = read_mnist.read_images("mnist/test_images")
    train_labels = read_mnist.read_labels("mnist/train_labels")
    test_labels  = read_mnist.read_labels("mnist/test_labels")

    train_images = preprocess_images(train_images, 0.1)
    test_images  = preprocess_images(test_images, 0.2)

    train_labels = preprocess_labels(train_labels, 0.1)
    test_labels  = preprocess_labels(test_labels, 0.2)

    start = time.time()

    num_epochs, batch_size = 1000, 1000
    nn = mlp.MLP(shape=(train_images.shape[1], 20, 20,test_labels.shape[1]), batch_size=batch_size, loss="softmax", num_epochs=num_epochs, eta=0.001, progress_epoch=10)

    nn.train(train_images, train_labels)

    runtime = time.time() - start

    pre_train = nn.predict(train_images, train_labels)
    pre_test = nn.predict(test_images, test_labels)

    print(f"Train Datasize: {train_images.shape[0]}")
    print(f"Test Datasize: {test_images.shape[0]}")
    print(f"Train Score: {nn.score(pre_train, train_labels)*100:.2f}% Correct")
    print(f"Test  Score: {nn.score(pre_test, test_labels)*100:2f}% Correct")

    print(f"Total Train Time for {num_epochs} epochs, {batch_size} batch size : {runtime:.2f} seconds")

    return



if __name__ == "__main__":
    main()
