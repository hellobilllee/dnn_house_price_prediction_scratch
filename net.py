from fully_connected_layer import FullyConnectedlayer
from activation_layer import ActivationLayer
from loss_layer import Losslayer
from initializer import Initializer
from optimizer import Optimizer
from metrics_calculator import MetricCalculator
import numpy as np
class DNNNet:
    def __init__(self, optimizer = Optimizer.batch_gradient_descent_fixed, initializer = Initializer.xavier, batch_size=16, weights_decay=0.001):
        self.optimizer = optimizer
        self.initializer = initializer
        self.batch_size = batch_size
        self.weights_decay = weights_decay
        self.fc1 = FullyConnectedlayer(13,16,self.batch_size, self.weights_decay)
        self.ac1 = ActivationLayer('relu')
        self.fc2 = FullyConnectedlayer(16,1,self.batch_size, self.weights_decay)
        self.loss = Losslayer("LeastSquareLoss")

    def forward_train(self,input_data, input_label):
        self.fc1.get_inputs_for_forward(input_data)
        self.fc1.forward()
        self.ac1.get_inputs_for_forward(self.fc1.outputs)
        self.ac1.forward()
        self.fc2.get_inputs_for_forward(self.ac1.outputs)
        self.fc2.forward()

        print("predict label: \n ", np.concatenate((self.fc2.outputs[:10], input_label[:10]), axis=1))
        self.loss.get_inputs_for_loss(self.fc2.outputs)
        self.loss.get_label_for_loss(input_label)
        self.loss.compute_loss()
        print("loss: ",self.loss.loss)


    def backward_train(self):
        self.loss.compute_gradient()
        self.fc2.get_inputs_for_backward(self.loss.grad_inputs)
        self.fc2.backward()
        self.ac1.get_inputs_for_backward(self.fc2.grad_inputs)
        self.ac1.backward()
        self.fc1.get_inputs_for_backward(self.ac1.grad_inputs)
        self.fc1.backward()

    def predict(self,input_data):
        self.fc1.get_inputs_for_forward(input_data)
        self.fc1.forward()
        self.ac1.get_inputs_for_forward(self.fc1.outputs)
        self.ac1.forward()

        self.fc2.get_inputs_for_forward(self.ac1.outputs)
        self.fc2.forward()
        return self.fc2.outputs

    def eval(self,input_data, input_label):
        self.fc1.update_batch_size(input_data.shape[0])
        self.fc1.get_inputs_for_forward(input_data)
        self.fc1.forward()
        self.ac1.get_inputs_for_forward(self.fc1.outputs)
        self.ac1.forward()
        self.fc2.update_batch_size(input_data.shape[0])
        self.fc2.get_inputs_for_forward(self.ac1.outputs)
        self.fc2.forward()
        print("predict: \n ",self.fc2.outputs[:10])
        print("label: \n", input_label[:10])
        metric = MetricCalculator(label=input_label, predict=self.fc2.outputs)
        metric.get_mae()
        metric.get_mse()
        metric.get_rmse()
        metric.print_metrics()

    def update(self):
        self.fc1.update(self.optimizer)
        self.fc2.update(self.optimizer)

    def initial(self):
        self.fc1.initialize_weights(self.initializer)
        self.fc2.initialize_weights(self.initializer)
