import numpy as np


class MLP():
    """Multi Layer Perceptron"""

    def __init__(self, **kwargs):
        self.shape              = kwargs.get("shape", None)
        self.activation         = kwargs.get("activation", "sigmoid")
        self.hidden_activation  = kwargs.get("hidden_activation", "sigmoid")
        self.eta                = kwargs.get("eta", 0.0001)
        self.weight_init        = kwargs.get("weight_init", "random")
        self.batch_size         = kwargs.get("batch_size", 25)
        self.loss               = kwargs.get("loss", "mse")

        self.weights            = []
        self.hidden_layers      = []


        self.act_funcs          = {"sigmoid": lambda x: 1/(1+np.exp(-x)),
                                   "relu"   : lambda x: 0 if x<=0 else x,
                                   "step"   : lambda x: 1 if x>0.5 else 0
                                   }

        self.act_funcs_derivs   = {"sigmoid" : lambda x : self.act_funcs["sigmoid"](x) * \
                                   (1-self.act_funcs["sigmoid"](x)),
                                   "relu" : lambda x : 0 if x<=0 else 1,
                                   "step" : lambda x : 0
                                   }

        self.init_weight_funcs  =  {"random" : lambda x, y : np.random.normal(0, 0.01, ((x+1), y))}

        self.loss_fn            = {"mse" : lambda yhat, y : 0.5 * sum((yhat-y)**2)
                                   }

        if self.shape is not None:
            self._setup_architecture()

        return


    def _check_valid_attributes(self, data_size):
        if self.activation not in self.act_funcs.keys():
            raise ValueError((f"Invalid activation passed : {self.activation}. "
                              "Activation must be one of {self.act_funcs.keys}"))

        if self.hidden_activation not in self.act_funcs.keys():
            raise ValueError((f"Invalid hidden_activation passed : {self.hidden_activation}. "
                              "Activation must be one of {self.act_funcs.keys}"))

        if self.weight_init not in self.init_weight_funcs.keys():
            raise ValueError((f"Invalid weight init function : {self.weight_init}."
                              " Must be one of {self.init_weight_funcs.keys()"))

        if not isinstance(self.batch_size, int) or self.batch_size <=0 or self.batch_size > data_size:
            raise ValueError((f"Batch size must be an integer, that is > 0 and <= total data size"
                              f". Batch size was set to {self.batch_size} and data size is {data_size}"))


    def _check_valid_data(self, inputs, outputs):
        try:
            inputs.shape[0]
        except:
            raise TypeError(f"Inputs must be a 2-d numpy array, not a {type(inputs)}")
        try:
            outputs.shape[0]
        except:
            raise TypeError(f"Outputs must be a 2-d numpy array, not a {type(outputs)}")

        if len(inputs.shape) !=2 or len(outputs.shape) != 2:
            raise ValueError((f"Inputs and outputs must be 2-d numpy array. "
                              "Input shpae is {inputs.shape} and outputs shape is {outputs.shape}"))

        if inputs.shape[1] != self.shape[0] or outputs.shape[1] != self.shape[-1]:
            raise ValueError(("Size of either input or output does not much architcture shape. "
                              f"Input has size {inputs.shape[0]}, outputs has size {outputs.shape[0]} "
                              f"and architecture shape is {self.shape}"))

        if inputs.shape[0] != outputs.shape[0]:
            raise ValueError(("The number of batched inputs does not match the number of batched outputs."
                              f" Input batch size is {inputs.shape[1]} and output batch size is {outputs.shape[1]}"))

    def _setup_architecture(self):
        if not isinstance(self.shape, tuple) and  len(self.shape) < 2:
            raise ValueError((f"Invalid shape attribute passed : {self.shape} of type {type(self.shape)}"
                              "Shape attribute of MLP must be of type tuple and size of 2 or greater. "
                              "Where first value is size of input (note do not specify biases, they are "
                              "automatically added) and the last is the size of the output"))

        for i in range(len(self.shape)-2):
            self.weights.append(self.init_weight_funcs[self.weight_init](self.shape[i], self.shape[i+1]))
            self.hidden_layers.append(None)#append(np.zeros(self.shape[i+1], dtype=float))

            self.weights.append(self.init_weight_funcs[self.weight_init](self.shape[-2], self.shape[-1]))

        return


    def _add_bias(self, inputs):
        bias = np.ones((inputs.shape[0], 1), dtype=float)
        return np.concatenate((bias, inputs), axis=1)


    def train(self, inputs, outputs, **kwargs):

        self.shape      = kwargs.get("shape", self.shape)
        self.activation = kwargs.get("activation", self.activation)
        self.hidden_activation = kwargs.get("hidden_activation", self.hidden_activation)
        self.eta        = kwargs.get("eta", self.eta)
        self.weight_init = kwargs.get("weight_init", self.weight_init)
        self.batch_size  = kwargs.get("batch_size", self.batch_size)

        self._check_valid_data(inputs, outputs)

        self._check_valid_attributes(inputs.shape[0])




        breakpoint()
        batched = 0
        while batched < len(inputs):
            batched_next = batched + self.batch_size
            batch_in = inputs[batched:batched_next]
            batch_out = outputs[batched:batched_next]
            batch_yhat = self._forward(batch_in, batch_out)
            self._backward(batch_yhat, batch_out)
            batched = batched_next

        return


    def _forward(self, inputs, _):
        next_in = inputs
        next_in = self._add_bias(next_in)
        for level in range(len(self.shape) - 1):
            weights = self.weights[level].reshape(next_in.shape[1], self.shape[level+1])
            if level != len(self.shape)- 2:
                self.hidden_layers[level] = next_in.dot(weights)
                next_in = self.hidden_layers[level]
                next_in = self._add_bias(self.hidden_layers[level])
            else:
                outputs = self._classify(next_in.dot(weights))

        return outputs


    def _classify(self, arr):
        out = np.zeros_like(arr)
        for i in range(len(arr)):
            cmax = arr[i,0]
            imax = 0
            for j in range(len(arr[i])):
                if arr[i, j] > cmax:
                    cmax = arr[i, j]
                    imax = j
            out[i, imax] = 1

        return out

    def _backward(self, yhat, y):
        breakpoint()
        #loss = self.loss_fn[self.loss](yhat, y)
        dldy = yhat - y
        for level in range(-1, -len(self.shape)+1, -1):
            dldw = 1
            self.weights[level] = self.weights[level] - self.eta * dldw

        return


    def predict(self, inputs):
        return
