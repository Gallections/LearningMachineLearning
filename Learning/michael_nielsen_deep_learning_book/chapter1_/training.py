import network
import mnist_loader
training_data, validation_data, test_data = \
mnist_loader.load_data_wrapper()

# Network of three layers, with one hidden layer
# net = network.Network([784, 30, 10])
# net.SGD(training_data, 30, 10, 3.0, test_data=test_data)


# Network of no hiddne layer, only two layers at a time
# net2 = network.Network([784, 10])
# net2.SGD(training_data, 40, 10, 2.7, test_data=test_data)


# Network of 2 hidden layers, so 4 layers in total
net3 = network.Network([784, 30, 30, 10])
net3.SGD(training_data, 30, 10, 3.0, test_data=test_data)