from loss_function import LossFunction
class Losslayer:
    def __init__(self, loss_function_name):
        self.lossfunc = LossFunction()
        self.inputs = 0
        self.loss = 0
        self.grad_inputs = 0
        if loss_function_name == 'SoftmaxLogloss':
            self.loss_function = self.lossfunc.softmax_logloss
            self.der_loss_function = self.lossfunc.der_softmax_logloss
        elif loss_function_name == 'LeastSquareLoss':
            self.loss_function = self.lossfunc.least_square_loss
            self.der_loss_function = self.lossfunc.der_least_square_loss
        else:
            print("wrong loss function")
    def get_label_for_loss(self, label):
        self.label = label

    def get_inputs_for_loss(self, inputs):
        self.inputs = inputs

    def compute_loss(self):
        self.loss = self.loss_function(self.inputs, self.label)

    def compute_gradient(self):
        self.grad_inputs = self.der_loss_function(self.inputs, self.label)