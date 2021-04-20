import numpy as np
from copy import deepcopy

np.random.seed(42)

class RNN():
    def __init__(self, shape, **kwargs):
        self.shape              = shape
        self.activation         = kwargs.get("activation", "tanh")
        self.learning_rate      = kwargs.get("learning_rate", 1e-1)
        self.weight_init        = kwargs.get("weight_init", "rnn")
        self.num_epochs         = kwargs.get("num_epochs", 10000)
        self.progress_epoch     = kwargs.get("progress_epoch", 100)
        self.track_epoch        = kwargs.get("track_epoch", False)
        self.seq_length         = kwargs.get("seq_length", 25)

        self.alphabet           = None
        self.alphabet2idx       = {}
        self.idx2alphabet       = {}

        self.weights            = []
        self.weights_hidden     = []
        self.hidden_layers      = []
        self.hidden_layers_act  = []
        self.hidden_previous    = []
        self.biases             = []
        self.memories           = []
        self.memories_hidden    = []
        self.memories_biases    = []

        self.act_funcs          = {"tanh": lambda x : np.tanh(x)
        }

        self.init_weight_funcs  =  {"random" : lambda x, y : np.random.normal(0, 1, ((x+1), y)),
                                    "rnn" : lambda x,y : np.random.randn(x, y) * 0.01,
                                    "zero" : lambda x, y : np.zeros((x+1, y), dtype=float)}


    def _setup_architecture(self):
        """Setup the neural network's architcecture

        Initialize the neural network structure to be used for weights and
        hidden layers
        """

        self.shape = (len(self.alphabet),) + self.shape + (len(self.alphabet),)

        for i in range(len(self.shape)-2):
            self.weights.append(self.init_weight_funcs[self.weight_init](self.shape[i+1], self.shape[i]))
            self.weights_hidden.append(self.init_weight_funcs[self.weight_init](self.shape[i+1], self.shape[i+1]))
            self.hidden_layers.append(None)
            self.hidden_layers_act.append(None)
            self.biases.append(np.zeros((self.shape[i+1], 1)))
            self.hidden_previous.append(np.zeros((self.shape[i+1], 1)))

        self.weights.append(self.init_weight_funcs[self.weight_init](self.shape[-1], self.shape[-2]))
        self.biases.append(np.zeros((self.shape[-1], 1)))

        for i in range(len(self.weights)):
            self.memories.append(np.zeros_like(self.weights[i]))
        for i in range(len(self.weights_hidden)):
            self.memories_hidden.append(np.zeros_like(self.weights_hidden[i]))
        for i in range(len(self.biases)):
            self.memories_biases.append(np.zeros_like(self.biases[i]))


    def train(self, data):
        self.alphabet = sorted(list(set(data)))
        for idx, char in enumerate(self.alphabet):
            self.alphabet2idx[char] = idx
            self.idx2alphabet[idx] = char

        self._setup_architecture()

        self.losses = [-np.log(1.0/len(self.alphabet)) * self.seq_length] # loss at time 0

        seq = 0
        for i in range(self.num_epochs):
            if seq + self.seq_length + 1 >= len(data): # reset the sequencing of data
                seq = 0
                for i in range(len(self.shape)-2):
                    self.hidden_previous[i] = np.zeros((self.shape(i+1), 1))

            inputs = [self.alphabet2idx[char] for char in data[seq:seq+self.seq_length]]
            targets = [self.alphabet2idx[char] for char in data[seq+1:seq+self.seq_length+1]]


            back_loss = self._backward(inputs, targets)
            loss = self.losses[-1] * 0.999 + back_loss * 0.001
            self.losses.append(loss)

            seq += self.seq_length
            if i % self.progress_epoch == 0:
                print(f"iter {i}, loss: {loss}")
                gen = self.generate(self.idx2alphabet[inputs[0]], 200)
                print('----\n %s \n----' % (gen, ))
                breakpoint()



    def _backward(self, inputs, targets):
        X, Hs, Y, P = {}, [], {}, {}

        for i in range(len(self.shape)-2):
            Hs.append({})
            Hs[i][-1] = deepcopy(self.hidden_previous[i])

        loss = 0

        #forward pass the to predict chars
        for i in range(len(inputs)):
            X[i] = np.zeros((len(self.alphabet), 1))
            X[i][inputs[i]] = 1
            H_input = []
            for j in range(len(Hs)):
                H_input.append(Hs[j][i-1])

            H_output, Y[i], P[i] = self._forward(X[i], H_input)
            for j in range(len(Hs)):
                Hs[j][i] = H_output[j]

            loss += -np.log(P[i][targets[i],0])

        dWs = []
        dWhy = np.zeros_like(self.weights[-1])
        dBy = np.zeros_like(self.biases[-1])
        dWhhs = []
        dBhs  = []
        dHnexts = []

        for i in range(len(self.weights_hidden)):
            dWhhs.append(np.zeros_like(self.weights_hidden[i]))
            dBhs.append(np.zeros_like(self.biases[i]))
            dHnexts.append(np.zeros_like(Hs[i][0]))

        for i in range(len(self.weights)-1):
            dWs.append(np.zeros_like(self.weights[i]))

        for i in reversed(range(len(inputs))):
            dY = np.copy(P[i])
            dY[targets[i]] -= 1
            dWhy += dY.dot(Hs[-1][i].T)
            dBy += dY
            dNext = dY

            for j in reversed(range(len(self.shape)-2)):
                dH = self.weights[j+1].T.dot(dNext) + dHnexts[j]
                dHraw = (1 - Hs[j][i] * Hs[j][i]) * dH
                dBhs[j] += dHraw
                dWhhs[j] += dHraw.dot(Hs[j][i-1].T)
                dHnexts[j] = self.weights_hidden[j].T.dot(dHraw)
            dWxh += dHraw.dot(X[i].T)

        # clip to solve gradient explosion
        for grad in [dWxh, dWhy, dBy]:
            np.clip(grad, -5, 5, out=grad)
        for grad in dBhs:
            np.clip(grad, -5, 5, out=grad)
        for grad in dWhhs:
            np.clip(grad, -5, 5, out=grad)

        for i in range(len(Hs)):
            self.hidden_previous[i] = deepcopy(Hs[i][len(inputs)-1])


        self.update_network(dWs, dWhhs, dWhy, dBhs, dBy)

        return loss


    def update_network(self, dWxh, dWhhs, dWhy, dBhs, dBy):

        derivatives = dWs + [dWhy]
        for i in range(len(self.weights)):
            self.memories[i] += derivatives[i] * derivatives[i]
            self.weights[i] += -self.learning_rate * derivatives[i] / np.sqrt(self.memories[i] + 1e-8)

        for i in range(len(self.weights_hidden)):
            self.memories_hidden[i] += dWhhs[i] * dWhhs[i]
            self.weights_hidden[i] += -self.learning_rate * dWhhs[i] / np.sqrt(self.memories_hidden[i] + 1e-8)

        derivatives = dBhs + [dBy]
        for i in range(len(self.biases)):
            self.memories_biases[i] += derivatives[i] * derivatives[i]
            self.biases[i] += -self.learning_rate * derivatives[i] / np.sqrt(self.memories_biases[i] + 1e-8)


    def _forward(self, inputs, hidden_in):
        hidden = []
        hidden_act = []

        hid = self.weights[0].dot(inputs) + self.weights_hidden[0].dot(hidden_in[0]) + self.biases[0]
        hidden.append(hid)
        hid_act = self.act_funcs[self.activation](hid)
        hidden_act.append(hid_act)

        for i in range(1, len(self.weights)-1):
            hid = self.weights[i].dot(hidden_act[i-1]) + self.weights_hidden.dot(hidden_in[i]) + self.biases[i]
            hidden.append(hid)
            hid_act = self.act_funcs[self.activation](hid)
            hidden_act.append(hid_act)

        output = self.weights[-1].dot(hidden_act[-1]) + self.biases[-1]
        output_norm = np.exp(output) / np.sum(np.exp(output))

        return hidden_act, output, output_norm


    def generate(self, char, num_chars):
        prev_char_idx = self.alphabet2idx[char]
        X = np.zeros((len(self.alphabet), 1))
        X[prev_char_idx] = 1
        Hs = self.hidden_previous
        indexes = []
        for i in range(num_chars):
            Hs, _, probability = self._forward(X, Hs)
            next_char_idx = np.random.choice(range(len(self.alphabet)), p=probability.ravel())
            X[prev_char_idx] = 0
            X[next_char_idx] = 1
            prev_char_idx = next_char_idx
            indexes.append(next_char_idx)
        chars = []
        for idx in indexes:
            chars.append(self.idx2alphabet[idx])
        return ''.join(chars)


if __name__ == "__main__":
    data = open('input.txt', 'r').read() # should be simple plain text file

    rnn = RNN((100,))
    rnn.train(data)
