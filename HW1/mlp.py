import numpy as np


class MLP():
    """Multi Layer Perceptron"""

    def __init__(self, **kwargs):
        self.shape      = kwargs.get("shape", None)
        self.activation = kwargs.get("activation", "sigmoid")
        self.hidden_activation = kwargs.get("hidden_activation", "sigmoid")
        self.eta        = kwargs.get("eta", None)
        self.weight_init = kwargs.get("weight_init", "random")

        self.weights    = []

        self.act_funcs = {"sigmoid": lambda x: 1/(1+np.exp(-x)),
                          "relu"   : lambda x: 0 if x<=0 else x,
                          "step"   : lambda x: 1 if x>0.5 else 0
                          }

        self.act_funcs_derivs = {"sigmoid" : lambda x : self.act_funcs["sigmoid"](x) * \
                                 (1-self.act_funcs["sigmoid"](x)),
                                 "relu" : lambda x : 0 if x<=0 else 1,
                                 "step" : lambda x : 0
                                 }

        self.init_weight_funcs =  {"random" : lambda x, y : np.random.normal((x,y))}


        if self.activation not in self.act_funcs.keys():
            raise ValueError((f"Invalid activation passed : {self.activation}. "
                              "Activation must be one of {self.act_funcs.keys}"))

        if self.hidden_activation not in self.act_funcs.keys():
            raise ValueError((f"Invalid hidden_activation passed : {self.hidden_activation}. "
                              "Activation must be one of {self.act_funcs.keys}"))

        if self.weight_init not in self.init_weight_funcs.keys():
            raise ValueError((f"Invalid weight init function : {self.weight_init}."
                              " Must be one of {self.init_weight_funcs.keys()"))

        if self.shape is not None:
            self._setup_archiitecture()

        return


    def _setup_archtiecture(self):
        if not isinstance(self.shape, tuple) and  len(self.shape) < 2:
            raise ValueError((f"Invalid shape attribute passed : {self.shape} of type {type(self.shape)}"
                              "Shape attribute of MLP must be of type tuple and size of 2 or greater. "
                              "Where first value is size of input (note do not specify biases, they are "
                              "automatically added) and the last is the size of the output"))

        for i in range(len(self.shape)):
            self.weights.append(self.init_weight_func["self.weight_init"](num_in, num_out))

        return


    def forward(inputs, outputs):
        return


    def backward():
        return
