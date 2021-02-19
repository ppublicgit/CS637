import numpy as np
from copy import deepcopy


class MLP():
    """Multi Layer Perceptron"""

    def __init__(self, shape, **kwargs):
        self.shape              = shape#kwargs.get("shape", None)
        self.activation         = kwargs.get("activation", "sigmoid")
        self.hidden_activation  = kwargs.get("hidden_activation", "sigmoid")
        self.eta                = kwargs.get("eta", 0.0001)
        self.weight_init        = kwargs.get("weight_init", "random")
        self.batch_size         = kwargs.get("batch_size", 50)
        self.loss               = kwargs.get("loss", "mse")
        self.num_epochs         = kwargs.get("num_epochs", 10000)
        self.progress_epoch     = kwargs.get("progress_epoch", 1000)
        self.track_epoch        = kwargs.get("track_epoch", False)
        self.optimization       = kwargs.get("optimization", "basic")

        self.weights            = []
        self.hidden_layers      = []
        self.hidden_layers_act  = []


        self.act_funcs          = {"sigmoid": lambda x : 1/(1+np.exp(-x)),
                                   "relu"   : lambda x: self._relu(x, False),
                                   "linear" : lambda x: x
                                   }

        self.act_funcs_derivs   = {"sigmoid" : lambda x : np.multiply(x, 1-x),
                                   "relu" : lambda x : self._relu(x, True),
                                   "linear" : lambda x: 1
                                   }

        self.init_weight_funcs  =  {"random" : lambda x, y : np.random.normal(0, 1, ((x+1), y)),
                                    "zero" : lambda x, y : np.zeros((x+1, y), dtype=float)}

        self.loss_fn            = {"mse" : lambda yhat, y : 0.5 * sum((yhat-y)**2),
                                   "softmax" : lambda yhat, y : self._softmax(yhat, y, False),
                                   "hinge" : lambda yhat, y : self._hinge(yhat, y)
                                   }

        self.loss_fn_derivs     = {"mse" : lambda yhat, y : yhat - y,
                                   "softmax" : lambda  yhat, y : self._softmax(yhat, y, True),
                                   "hinge" : lambda yhat, y : self._hinge_deriv(yhat, y)
                                   }

        self.optimization_func  = {"basic" : lambda old, deriv, eta : old - eta * deriv
                                   }

        self._setup_architecture()

        return


    def _relu(self, arr, deriv=False):
        ret = np.zeros_like(arr, dtype=float)
        for i in range(arr.shape[0]):
            for j in range(arr.shape[1]):
                if arr[i,j] > 0 and deriv:
                    ret[i,j] = 1
                elif arr[i, j] > 0:
                    ret[i,j] = arr[i,j]
        return ret


    def _softmax(self, yhat, y, deriv=False):
        #breakpoint()
        summed_prob = np.sum(np.exp(yhat), axis=0)
        if not deriv:
            ret = np.zeros_like(summed_prob, dtype=float)
        else:
            ret = np.zeros_like(yhat, dtype=float)
        for j in range(y.shape[1]):
            for i in range(y.shape[0]):
                if y[i, j] == 1:
                    index = i
                    break
            if not deriv:
                ret[j] = -np.log(np.exp(yhat[index, j])/ summed_prob[j])
            else:
                for i in range(y.shape[0]):
                    if i == index:
                        ret[i, j] = -(1-np.exp(yhat[i, j])/summed_prob[j])
                    else:
                        ret[i, j] = np.exp(yhat[i, j])/ summed_prob[j]
        return ret


    def _hinge(self, yhat, y):
        ret = np.zeros(y.shape[1], dtype=float)
        for j in range(y.shape[1]):
            loss = 0
            index = np.where(y[:, j] == 1)[0][0]
            for i in range(y.shape[0]):
                if i != index:
                    loss += max(0, yhat[i, j] - yhat[index, j] + 1)
            ret[j] = loss
        return ret


    def _hinge_deriv(self, yhat, y):
        dldyhat = np.zeros_like(yhat, dtype=float)
        for j in range(y.shape[1]):
            count = 0
            index = np.where(y[:, j] == 1)[0][0]
            for i in range(y.shape[0]):
                if i != index:
                    if yhat[i, j] - yhat[index, j] + 1 > 0:
                        dldyhat[i, j] = 1
                        count += 1
            dldyhat[index, j] = -count
        return dldyhat

    def _check_valid_attributes(self, data_size):
        if self.activation not in self.act_funcs.keys():
            raise ValueError((f"Invalid activation passed : {self.activation}. "
                              f"Activation must be one of {self.act_funcs.keys()}"))

        ha_valid = [ha in self.act_funcs.keys() and ha != "softmax" for ha in self.hidden_activation]
        if not all(ha_valid):
            raise ValueError((f"Invalid hidden_activation passed : {self.hidden_activation}. "
                              f"Activation must be one of {self.act_funcs.keys}"))

        if self.weight_init not in self.init_weight_funcs.keys():
            raise ValueError((f"Invalid weight init function : {self.weight_init}."
                              f" Must be one of {self.init_weight_funcs.keys()}"))

        if not isinstance(self.batch_size, int) or self.batch_size <=0 or self.batch_size > data_size:
            raise ValueError((f"Batch size must be an integer, that is > 0 and <= total data size"
                              f". Batch size was set to {self.batch_size} and data size is {data_size}"))

        if not isinstance(self.num_epochs, int) or self.num_epochs <= 0:
            raise ValueError(("Invalid number of epochs set for training NN."
                              " Set number of epochs to an integer greater than 0. "
                              f"num_epochs was set to : {self.num_epochs}"))

        if not isinstance(self.progress_epoch, int) or self.progress_epoch < 1:
            raise ValueError(("Invalid progress_epoch set. Progress epoch must be an "
                              "integer greater than or equal to 0. Not {self.progress_epoch"))

        if (self.loss == "softmax" or self.loss == "hinge") \
           and self.activation != "linear":
            print("Setting output activation to linear since softmax or hinge loss was chosen.")
            self.activation = "linear"

        if self.optimization not in self.optimization_func.keys():
            raise ValueError(f"Invalid optimaztion method set ({self.optimization}).")

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
            raise ValueError(("The number of batched inputs does not match "
                              "the number of batched outputs."
                              f" Input batch size is {inputs.shape[1]} "
                              f"and output batch size is {outputs.shape[1]}"))


    def _check_valid_val_data(self, val_data):
        if len(val_data) != 2:
            raise ValueError((f"Size of val data is not 2. Must be an iterable of size 2 "
                             f"with the first value a 2-d numpy array of inputs and the second"
                              "value as a 2-d numpy array of outputs."))
        val_inputs = val_data[0]
        val_outputs = val_data[1]
        try:
            val_inputs.shape[0]
        except:
            raise TypeError(f"Val inputs must be a 2-d numpy array, not a {type(val_inputs)}")
        try:
            val_outputs.shape[0]
        except:
            raise TypeError(f"Val outputs must be a 2-d numpy array, not a {type(val_outputs)}")

        if len(val_inputs.shape) !=2 or len(val_outputs.shape) != 2:
            raise ValueError((f"Val inputs and val outputs must be 2-d numpy array. "
                              "Val input shpae is {val_inputs.shape} and val_outputs "
                              f"shape is {val_outputs.shape}"))

        if val_inputs.shape[1] != self.shape[0] or val_outputs.shape[1] != self.shape[-1]:
            raise ValueError(("Size of either val input or val output does not much architcture shape. "
                              f"Val nput has size {val_inputs.shape[0]}, val_outputs has size "
                              f" {val_outputs.shape[0]} "
                              f"and architecture shape is {self.shape}"))

        if val_inputs.shape[0] != val_outputs.shape[0]:
            raise ValueError(("The number of batched val_inputs does not match "
                              "the number of batched val_outputs."
                              f" Input batch size is {val_inputs.shape[1]} "
                              f"and output batch size is {val_outputs.shape[1]}"))


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
        self.eta            = kwargs.get("eta", self.eta)
        self.weight_init    = kwargs.get("weight_init", self.weight_init)
        self.batch_size     = kwargs.get("batch_size", self.batch_size)
        self.num_epochs     = kwargs.get("num_epochs", self.num_epochs)
        self.progress_epoch = kwargs.get("progress_epoch", self.progress_epoch)
        self.track_epoch    = kwargs.get("track_epoch", self.track_epoch)
        self.optimization   = kwargs.get("optimization", self.optimization)
        val_data            = kwargs.get("val_data", None)

        self._check_valid_data(inputs, outputs)

        if self.track_epoch and val_data is not None:
            self._check_valid_val_data(val_data)

        self._check_valid_attributes(inputs.shape[0])

        inputs_T = inputs.T
        outputs_T = outputs.T

        if self.track_epoch:
            if val_data is not None:
                inputs_val_T = val_data[0].T
                outputs_val_T = val_data[1].T
                self._epoch_perf_val = np.zeros(self.num_epochs, dtype=float)
            else:
                self._epoch_perf_val = None
            self._epoch_perf = np.zeros(self.num_epochs, dtype=float)
        else:
            self._epoch_perf = None

        self._weights_epoch = []
        for i in range(self.num_epochs):
            self._weights_epoch.append(deepcopy(self.weights))
            batched = 0
            while batched < len(inputs):
                batched_next = batched + self.batch_size
                batch_in = inputs_T[:, batched:batched_next]
                batch_out = outputs_T[:, batched:batched_next]
                batch_yhat = self._forward(batch_in)
                self._backward(batch_yhat, batch_out, batch_in)
                batched = batched_next
            if self.progress_epoch and (i+1) % self.progress_epoch == 0:
                print(f"Epoch : {i-1}")
                ave_loss = sum(self.loss_fn[self.loss](batch_yhat, batch_out))/self.batch_size
                print(f"Ave. Loss : {ave_loss}")
            if self.track_epoch:
                epoch_yhat = self._forward(inputs_T)
                self._epoch_perf[i] = sum(self.loss_fn[self.loss](
                    epoch_yhat, outputs_T))/inputs_T.shape[1]
                if val_data is not None:
                    epoch_yhat_val = self._forward(inputs_val_T)
                    self._epoch_perf_val[i] = sum(self.loss_fn[self.loss](
                        epoch_yhat_val, outputs_val_T))/inputs_val_T.shape[1]
        self._weights_epoch.append(deepcopy(self.weights))
        return


    def set_weights(self, epoch):
        self.weights = self._weights_epoch[epoch]
        return

    def get_epoch_performance(self):
        if self._epoch_perf is None:
            raise ValueError("Epoch performarnce was not tracked")
        return self._epoch_perf, self._epoch_perf_val


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
                out_act = self.act_funcs[self.activation](outputs)

        return out_act


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

        self.weights[level] = self.optimization_func[self.optimization(
            self.weights[level], dldw, self.eta
            )

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

            self.weights[level] = self.optimization_func[self.optimization(
            self.weights[level], dldw, self.eta
            )

            level -= 1

        next_in = self._add_bias(inputs)
        dldw = dldo.dot(next_in.T)
        self.weights[level] = self.optimization_func[self.optimization(
            self.weights[level], dldw, self.eta
            )

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
