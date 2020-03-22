import numpy as np
class FullyConnectedlayer:
    def __init__(self, num_neuron_inputs, num_neuron_outputs, batch_size=16,weights_decay=0.001):
        self.num_neuron_inputs = num_neuron_inputs
        self.num_neuron_outputs = num_neuron_outputs
        self.inputs = np.zeros((batch_size, num_neuron_inputs))
        self.outputs = np.zeros((batch_size, num_neuron_outputs))
        self.weights = np.zeros((num_neuron_inputs, num_neuron_outputs))
        self.bias = np.zeros(num_neuron_outputs)
        self.weights_previous_direction = np.zeros((num_neuron_inputs, num_neuron_outputs))
        self.bias_previous_direction = np.zeros(num_neuron_outputs)
        self.grad_weights = np.zeros((batch_size, num_neuron_inputs, num_neuron_outputs))
        self.grad_bias = np.zeros((batch_size, num_neuron_outputs))
        self.grad_inputs = np.zeros((batch_size, num_neuron_inputs))
        self.grad_outputs = np.zeros((batch_size, num_neuron_outputs))
        self.batch_size = batch_size
        self.weights_decay = weights_decay

    def initialize_weights(self, initializer):
        self.weights = initializer(self.num_neuron_inputs, self.num_neuron_outputs)

    # 在正向传播过程中,用于获取输入;
    def get_inputs_for_forward(self, inputs):
        self.inputs = inputs

    def forward(self):
        self.outputs = self.inputs.dot(self.weights)+ np.tile(self.bias, (self.batch_size, 1))

    # 在反向传播过程中,用于获取输入;
    def get_inputs_for_backward(self, grad_outputs):
        self.grad_outputs = grad_outputs

    def backward(self):
        # 求权值的梯度,求得的结果是一个三维的数组,因为有多个样本;
        for i in np.arange(self.batch_size):
            self.grad_weights[i, :] = np.tile(self.inputs[i, :], (1, 1)).T.dot(np.tile(self.grad_outputs[i, :], (1, 1))) + self.weights * self.weights_decay
        # 求偏置的梯度;
        self.grad_bias = self.grad_outputs
        # 求输入的梯度;
        self.grad_inputs = self.grad_outputs.dot(self.weights.T)

    def update(self, optimizer):
        # 权值与偏置的更新;
        grad_weights_average = np.mean(self.grad_weights, 0)
        grad_bias_average = np.mean(self.grad_bias, 0)
        (self.weights, self.weights_previous_direction) = optimizer(self.weights, grad_weights_average,self.weights_previous_direction)
        (self.bias, self.bias_previous_direction) = optimizer(self.bias,grad_bias_average, self.bias_previous_direction)

    def update_batch_size(self,batch_size):
        self.batch_size = batch_size
