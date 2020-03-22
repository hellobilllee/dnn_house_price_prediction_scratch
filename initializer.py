import numpy as np
import scipy.stats as stats
class Initializer:
    # xavier 初始化方法
    def xavier(self, num_neuron_inputs, num_neuron_outputs):
        temp1 = np.sqrt(6) / np.sqrt(num_neuron_inputs + num_neuron_outputs + 1)
        weights = stats.uniform.rvs(-temp1, 2 * temp1, (num_neuron_inputs, num_neuron_outputs))
        return weights