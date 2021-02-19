import numpy as np
import time

import mlp
import read_mnist





def main():
    train_images = read_mnist.read_images("mnist/train_images")
    test_images  = read_mnist.read_images("mnist/test_images")
    train_labels = read_mnist.read_labels("mnist/train_labels")
    test_labels  = read_mnist.read_labels("mnist/test_labels")

    train_images = read_mnist.preprocess_images(train_images, 0.1)
    test_images  = read_mnist.preprocess_images(test_images, 0.2)

    train_labels = read_mnist.preprocess_labels(train_labels, 0.1)
    test_labels  = read_mnist.preprocess_labels(test_labels, 0.2)

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
