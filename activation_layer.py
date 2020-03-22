from activation_function import ActivationFunction
class ActivationLayer:
    def __init__(self, activation_function_name):
        self.actfunc = ActivationFunction()
        if activation_function_name == 'sigmoid':
            self.activation_function = self.actfunc.sigmoid
            self.der_activation_function = self.actfunc.der_sigmoid
        elif activation_function_name == 'tanh':
            self.activation_function = self.actfunc.tanh
            self.der_activation_function = self.actfunc.der_tanh
        elif activation_function_name == 'relu':
            self.activation_function = self.actfunc.relu
            self.der_activation_function = self.actfunc.der_relu
        elif activation_function_name == 'linear':
            self.activation_function = self.actfunc.identity
            self.der_activation_function = self.actfunc.der_identity
        else:
            print('wrong activation function')
        self.inputs = 0
        self.outputs = 0
        self.grad_inputs = 0
        self.grad_outputs = 0

    def get_inputs_for_forward(self, inputs):
        self.inputs = inputs

    def forward(self):
        self.outputs = self.activation_function(self.inputs)

    def get_inputs_for_backward(self, grad_outputs):
        self.grad_outputs = grad_outputs

    def backward(self):
        self.grad_inputs = self.grad_outputs * self.der_activation_function(self.inputs)