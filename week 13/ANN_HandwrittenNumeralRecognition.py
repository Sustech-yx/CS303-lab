import os
import numpy as np
import pickle

from utils import xavier, gaussian, relu, d_relu, softmax, getLabel
import matplotlib
import matplotlib.pyplot as plt

print(matplotlib.get_backend())
# matplotlib.use('Agg')

'''
num_input = 784
num_layer1 = 500
num_layer2 = 300
num_layer3 = 100
num_output = 10
'''


def RandomInitial(x, y):
    w = np.random.rand(x, y)
    b = np.random.random()
    b = b * np.ones((1, y))
    return w, b


def Parameter_initialization(num_input, num_layer1, num_layer2, num_layer3, num_output):
    Parameter = {}

    w1 = xavier(num_input, num_layer1)
    w2 = xavier(num_layer1, num_layer2)
    w3 = xavier(num_layer2, num_layer3)
    w4 = xavier(num_layer3, num_output)
    # w1, _ = RandomInitial(num_input, num_layer1)
    # w2, _ = RandomInitial(num_layer1, num_layer2)
    # w3, _ = RandomInitial(num_layer2, num_output)
    # print(w1, w2, w3)
    # Your code starts here
    # Please Initialize all parameters used in ANN-Hidden Layers with Xavier
    b1 = np.ones(500)
    b2 = np.ones(300)
    b3 = np.ones(100)
    b4 = np.ones(10)

    # Your code ends here
    Parameter['w1'] = w1
    Parameter['b1'] = b1
    Parameter['w2'] = w2
    Parameter['b2'] = b2
    Parameter['w3'] = w3
    Parameter['b3'] = b3
    Parameter['w4'] = w4
    Parameter['b4'] = b4
    return Parameter


def Hidden_Layer(x, w, b, batch_size):
    # Your code starts here
    z = np.dot(x, w)
    a = relu(z + b)
    # Your code ends here
    return a


def Output_Layer(x, w, b, batch_size):
    # Your code starts here
    z = np.dot(x, w)
    a = softmax(z + b)
    # Your code ends here
    return a


def Loss(label, logits):
    # label : Actual label BATCHSIZE * class
    # logits : The predicted results of your model
    # Your code starts here
    # BATCHSIZE = label.shape()[0]
    # count = 0
    # for index, l in enumerate(label):
    #     if l[logits[index]] == 1:
    #         count += 1
    # Your code ends here
    # print(label.shape)
    # print(logits.shape)
    # tmp = np.dot(label, np.log(logits).T)
    # loss = - np.sum(tmp)
    loss = -np.sum(label * np.log(logits))
    return loss


def Back_propagation(logits, label, w1, b1, w2, b2, w3, b3, w4, b4, a3, a2, a1, image_blob):
    # label : Actual label BATCHSIZE * class
    # logits : The predicted results of your model
    # Your code starts here
    # d_w1, d_w2, d_w3, d_b1, d_b2, d_b3 = 0, 0, 0, 0, 0, 0

    delta = logits - label
    d_w4 = np.dot(a3.T, delta)
    d_b4 = np.sum(delta, axis=0)

    error1 = np.dot(delta, w4.T)
    delta = error1 * d_relu(a3)
    d_w3 = np.dot(a2.T, delta)
    d_b3 = np.sum(delta, axis=0)

    error2 = np.dot(delta, w3.T)
    delta = error2 * d_relu(a2)
    d_w2 = np.dot(a1.T, delta)
    d_b2 = np.sum(delta, axis=0)

    error3 = np.dot(delta, w2.T)
    delta = error3 * d_relu(a1)
    d_w1 = np.dot(image_blob.T, delta)
    d_b1 = np.sum(delta, axis=0)
    # Your code ends here
    return d_w1, d_b1, d_w2, d_b2, d_w3, d_b3, d_w4, d_b4


if __name__ == '__main__':
    (train_images, train_labels, test_images, test_labels) = pickle.load(open('data.pkl', 'rb'), encoding='latin1')
    # train_images: 10000 * 784
    # test_images: 1000 * 784
    # train_labels: 10000
    # test_labels: 1000
    EPOCH = 100
    ITERS = 100
    BATCHSIZE = 100
    LR_BASE = 0.2
    k = 0.0005  # lambda
    num_input = 784
    num_layer1 = 500
    num_layer2 = 300
    num_layer3 = 100
    num_output = 10
    ### 1. Data preprocessing: normalize all pixels to [0,1) by dividing 256
    train_images = train_images / 256.0
    test_images = test_images / 256.0
    print(type(train_images[0][0]))

    ### 2. Weight initialization: Xavier
    Parameter = Parameter_initialization(num_input, num_layer1, num_layer2, num_layer3, num_output)
    w1, b1, w2, b2, w3, b3, w4, b4 = Parameter['w1'], Parameter['b1'], Parameter['w2'], Parameter['b2'],\
                                     Parameter['w3'], Parameter['b3'], Parameter['w4'], Parameter['b4']

    ### 3. training of neural network
    loss = np.zeros(EPOCH)  # save the loss of each epoch
    accuracy = np.zeros(EPOCH)  # save the accuracy of each epoch
    for epoch in range(0, EPOCH):
        if epoch <= EPOCH / 2:
            lr = LR_BASE
        else:
            lr = LR_BASE / 10.0
        for iters in range(0, ITERS):
            image_blob = train_images[iters * BATCHSIZE:(iters + 1) * BATCHSIZE, :]  # 100*784
            label_blob = train_labels[iters * BATCHSIZE:(iters + 1) * BATCHSIZE]  # 100*1
            label = getLabel(label_blob, BATCHSIZE, num_output)

            # Forward propagation  Hidden Layer
            a1 = Hidden_Layer(image_blob, w1, b1, BATCHSIZE)
            a2 = Hidden_Layer(a1, w2, b2, BATCHSIZE)
            a3 = Hidden_Layer(a2, w3, b3, BATCHSIZE)
            # Forward propagation  output Layer
            a4 = Output_Layer(a3, w4, b4, BATCHSIZE)
            # if np.count_nonzero(a4) != 1000:
            #     print(a4)
            # compute loss
            loss_tmp = Loss(label, a4)
            if iters % 100 == 99:
                loss[epoch] = loss_tmp
                print('Epoch ' + str(epoch + 1) + ': ')
                print(loss_tmp)
            # Back propagation
            d_w1, d_b1, d_w2, d_b2, d_w3, d_b3, d_w4, d_b4 =\
                Back_propagation(a4, label, w1, b1, w2, b2, w3, b3, w4, b4, a3, a2, a1, image_blob)

            # Gradient update
            w1 = w1 - lr * d_w1 / BATCHSIZE - lr * k * w1
            b1 = b1 - lr * d_b1 / BATCHSIZE
            w2 = w2 - lr * d_w2 / BATCHSIZE - lr * k * w2
            b2 = b2 - lr * d_b2 / BATCHSIZE
            w3 = w3 - lr * d_w3 / BATCHSIZE - lr * k * w3
            b3 = b3 - lr * d_b3 / BATCHSIZE
            w4 = w4 - lr * d_w4 / BATCHSIZE - lr * k * w4
            b4 = b4 - lr * d_b4 / BATCHSIZE

            # Testing for accuracy
            if iters % 100 == 99:
                # print(d_w1, d_w2, d_w3)
                z1 = np.dot(test_images, w1) + np.tile(b1, (1000, 1))
                a1 = relu(z1)
                z2 = np.dot(a1, w2) + np.tile(b2, (1000, 1))
                a2 = relu(z2)
                z3 = np.dot(a2, w3) + np.tile(b3, (1000, 1))
                a3 = relu(z3)
                z4 = np.dot(a3, w4) + np.tile(b4, (1000, 1))
                a4 = softmax(z4)  # 1000*10
                predict = np.argmax(a4, axis=1)
                print('Accuracy: ')
                accuracy[epoch] = 1 - np.count_nonzero(predict - test_labels) * 1.0 / 1000
                print(accuracy[epoch])

    ### 4. Plot
    plt.figure(figsize=(12, 5))
    ax1 = plt.subplot(121)
    ax1.plot(np.arange(EPOCH) + 1, loss[0:], 'r', label='Loss', linewidth=2)
    plt.xlabel('Epoch', fontsize=16)
    plt.ylabel('Loss on trainSet', fontsize=16)
    plt.grid()
    ax2 = plt.subplot(122)
    ax2.plot(np.arange(EPOCH) + 1, accuracy[0:], 'b', label='Accuracy', linewidth=2)
    plt.xlabel('Epoch', fontsize=16)
    plt.ylabel('Accuracy on trainSet', fontsize=16)
    plt.grid()
    plt.tight_layout()
    plt.savefig('figure.pdf', dbi=1200)
    plt.show()
