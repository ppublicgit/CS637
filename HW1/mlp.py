import numpy as np


class MLP():
    """Multi Layer Perceptron"""

    def __init__(self, shape, **kwargs):
        self.shape              = shape#kwargs.get("shape", None)
        self.activation         = kwargs.get("activation", "sigmoid")
        self.hidden_activation  = kwargs.get("hidden_activation", "sigmoid")
        self.eta                = kwargs.get("eta", 0.0001)
        self.weight_init        = kwargs.get("weight_init", "random")
        self.batch_size         = kwargs.get("batch_size", 25)
        self.loss               = kwargs.get("loss", "mse")
        self.num_epochs         = kwargs.get("num_epochs", 1000)

        self.weights            = []
        self.hidden_layers      = []
        self.hidden_layers_act  = []


        self.act_funcs          = {"sigmoid": lambda x : 1/(1+np.exp(-x)),
                                   #"relu"   : lambda x: 0 if x<=0 else x,
                                   #"step"   : lambda x: 1 if x>0.5 else 0
                                   }

        self.act_funcs_derivs   = {"sigmoid" : lambda x : np.multiply(x, 1-x)
                                   #"relu" : lambda x : 0 if x<=0 else 1,
                                   #"step" : lambda x : 0
                                   }

        self.init_weight_funcs  =  {"random" : lambda x, y : np.random.normal(0, 0.01, ((x+1), y)),
                                    "zero" : lambda x, y : np.zeros((x+1, y), dtype=float)}

        self.loss_fn            = {"mse" : lambda yhat, y : 0.5 * sum((yhat-y)**2),
                                   "softmax" : lambda yhat, y : self._soft_max(yhat, y)
                                   }

        self.loss_fn_derivs     = {"mse" : lambda yhat, y : yhat - y,
                                   "softmax" : lambda  yhat, y : 1
                                   }

        self._setup_architecture()

        return


    def _soft_max(self, yhat, y):
        #normalize = np.sum(np.exp(x).T, axis=1).reshape(1, np.shape(x)[1])
        loss = 0
        for i in range(y.shape[1]):
            normalize = np.sum(np.exp(yhat[:, i]))
            norm_yhat = np.zeros(y.shape[0], dtype=float)
            for j in range(y.shape[0]):
                norm_yhat[j] = np.exp(yhat[j, i])/normalize
            index = np.where(y[:, i] == 1)[0]
            loss += -np.log(norm_yhat[index])
        return loss/len(yhat)


    def _check_valid_attributes(self, data_size):
        if self.activation not in self.act_funcs.keys():
            raise ValueError((f"Invalid activation passed : {self.activation}. "
                              "Activation must be one of {self.act_funcs.keys}"))

        ha_valid = [ha in self.act_funcs.keys() for ha in self.hidden_activation]
        if not all(ha_valid):
            raise ValueError((f"Invalid hidden_activation passed : {self.hidden_activation}. "
                              "Activation must be one of {self.act_funcs.keys}"))

        if self.weight_init not in self.init_weight_funcs.keys():
            raise ValueError((f"Invalid weight init function : {self.weight_init}."
                              " Must be one of {self.init_weight_funcs.keys()"))

        if not isinstance(self.batch_size, int) or self.batch_size <=0 or self.batch_size > data_size:
            raise ValueError((f"Batch size must be an integer, that is > 0 and <= total data size"
                              f". Batch size was set to {self.batch_size} and data size is {data_size}"))

        if not isinstance(self.num_epochs, int) or self.num_epochs <= 0:
            raise ValueError(("Invalid number of epochs set for training NN."
                              " Set number of epochs to an integer greater than 0. "
                              f"num_epochs was set to : {self.num_epochs}"))

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
            self.weights.append(self.init_weight_funcs[self.weight_init](self.shape[i], self.shape[i+1]).T)
            self.hidden_layers.append(None)#append(np.zeros(self.shape[i+1], dtype=float))
            self.hidden_layers_act.append(None)

        self.weights.append(self.init_weight_funcs[self.weight_init](self.shape[-2], self.shape[-1]).T)

        return


    def _add_bias(self, inputs):
        bias = np.ones((1, inputs.shape[1]), dtype=float)
        return np.concatenate((bias, inputs), axis=0)


    def _activate(self, level):
        act_fn = self.act_funcs[self.hidden_activation[level]]
        return act_fn(self.hidden_layers[level])


    def _set_hidden_act(self, ha):
        if isinstance(ha, str):
            self.hidden_activation = [ha] * (len(self.shape)-2)
        elif len(ha) != len(self.shape) - 2:
            raise ValueError(("Length of specified hidden activation functions ({len(ha)}). "
                              f"Does not match number of hidden layers ({len(self.shape)-2}). "
                              f"Either set hidden activation to a string to use the same "
                              "activation function for each hidden layer, or set to a list of "
                              "strings for the activation function of choice for each hidden "
                              "layer."))
        else:
            self.hidden_activation = ha
        return


    def train(self, inputs, outputs, **kwargs):

        #self.shape      = kwargs.get("shape", self.shape)
        self.activation = kwargs.get("activation", self.activation)
        self._set_hidden_act(
            kwargs.get("hidden_activation", self.hidden_activation)
            )
        self.eta        = kwargs.get("eta", self.eta)
        self.weight_init = kwargs.get("weight_init", self.weight_init)
        self.batch_size  = kwargs.get("batch_size", self.batch_size)
        self.num_epochs = kwargs.get("num_epochs", self.num_epochs)

        self._check_valid_data(inputs, outputs)

        self._check_valid_attributes(inputs.shape[0])

        inputs_T = inputs.T
        outputs_T = outputs.T


        for i in range(self.num_epochs):
            batched = 0
            while batched < len(inputs):
                batched_next = batched + self.batch_size
                batch_in = inputs_T[:, batched:batched_next]
                batch_out = outputs_T[:, batched:batched_next]
                batch_yhat = self._forward(batch_in)
                self._backward(batch_yhat, batch_out, batch_in)
                batched = batched_next
            if (i+1) % 100 == 0:
                    print(f"Epoch : {i-1}")
                    loss = self.loss_fn[self.loss](batch_yhat, batch_out)
                    print(f"Loss : {loss}")

        return


    def _forward(self, inputs):
        next_in = inputs
        next_in = self._add_bias(next_in)
        for level in range(len(self.shape) - 1):
            weights = self.weights[level]#.reshape(next_in.shape[1], self.shape[level+1])
            if level != len(self.shape)- 2:
                self.hidden_layers[level] = weights.dot(next_in)
                self.hidden_layers_act[level] = self._activate(level)
                next_in = self.hidden_layers_act[level]
                next_in = self._add_bias(self.hidden_layers_act[level])
            else:
                #outputs = self._classify(weights.dot(next_in))
                outputs = weights.dot(next_in)

        return outputs


    def _classify(self, arr):
        out = np.zeros_like(arr)
        for j in range(arr.shape[1]):
            cmax = arr[0, j]
            imax = 0
            for i in range(arr.shape[0]):
                if arr[i, j] > cmax:
                    cmax = arr[i, j]
                    imax = i
            out[imax, j] = 1
        return out

    def _backward(self, yhat, y, inputs):
        #loss = self.loss_fn[self.loss](yhat, y)
        dldy = self.loss_fn_derivs[self.loss](yhat, y)
        if len(self.hidden_layers) == 0:
            next_in = self._add_bias(inputs)
            dldwlast = dldy.dot(next_in.T)
            self.weights[-1] = self.weights[-1] - self.eta * dldwlast
            return

        level = -1
        next_in = self._add_bias(self.hidden_layers_act[level])
        dldw = dldy.dot(next_in.T)

        wprev = self.weights[level][:, 1:]
        dado = self.act_funcs_derivs[self.hidden_activation[level]](
            self.hidden_layers_act[level]
        )
        #dadz = self._add_bias(dadz)
        dlda = wprev.T.dot(dldy)
        dldo = np.multiply(dlda, dado)

        self.weights[level] = self.weights[level] - self.eta * dldw

        level -= 1
        #for level in range(-2, -len(self.shape)+1, -1):
        while level != -len(self.shape) + 1:
            next_in = self._add_bias(self.hidden_layers_act[level])
            dldw = dldo.dot(next_in.T)

            wprev = self.weights[level][:, 1:]
            dado = self.act_funcs_derivs[self.hidden_activation[level]](
                self.hidden_layers_act[level]
            )
            dlda = wprev.T.dot(dldo)
            dldo = np.multiply(dlda, dado)

            self.weights[level] = self.weights[level] - self.eta * dldw

            level -= 1

        next_in = self._add_bias(inputs)
        dldw = dldo.dot(next_in.T)
        self.weights[level] = self.weights[level] - self.eta * dldw

        return


    def predict(self, inputs, outputs):
        predictions = np.zeros_like(outputs)
        for i in range(outputs.shape[0]):
            predict = self._classify(self._forward(inputs[i].reshape(inputs.shape[1], 1)))
            predictions[i, :] = predict.T
        return predictions


    def score(self, predictions, targets):
        correct = 0
        for i in range(predictions.shape[0]):
            if any(predictions[i] != targets[i]):
                continue
            correct += 1

        return correct/predictions.shape[0]
