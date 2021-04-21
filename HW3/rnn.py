import numpy as np
from copy import deepcopy
import os
import pickle
import glob

np.random.seed(42)

class RNN():
    def __init__(self, shape, **kwargs):
        self.shape              = shape
        self.activation         = kwargs.get("activation", "tanh")
        self.learning_rate      = kwargs.get("learning_rate", 1e-1)
        self.weight_init        = kwargs.get("weight_init", "rnn")
        self.num_epochs         = kwargs.get("num_epochs", 10000)
        self.progress_epoch     = kwargs.get("progress_epoch", 100)
        self.track_epoch        = kwargs.get("track_epoch", True)
        self.save_progress      = kwargs.get("save_progress", False)
        self.seq_length         = kwargs.get("seq_length", 25)
        self.filename_states    = kwargs.get("filename_states", None)

        self.alphabet           = None
        self.alphabet2idx       = {}
        self.idx2alphabet       = {}

        self.weights            = []
        self.weights_hidden     = []
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

        self.state_dir = (os.path.join(os.getcwd(), "states"))
        self.progress_loaded = False

        self.epoch_start = 0

    def save_nn(self, epoch):
        if self.filename_states is None:
            print("filename_states is not set. It must be set to save nn progress.")
            ret = input("what would you like to call your filename prefix for your saved states? ")
            self.filename_states = ret

        attributes = [self.shape, self.activation, self.learning_rate,
                      self.num_epochs, self.progress_epoch, self.track_epoch,
                      self.seq_length, self.alphabet, self.alphabet2idx, self.idx2alphabet]

        neurons = [self.weights, self.weights_hidden, self. hidden_previous,
                   self.biases, self.memories, self.memories_hidden,
                   self.memories_biases]

        state_info = [self.losses[-1], epoch, self.texts[-1]]

        saved_info = (attributes, neurons, state_info)

        out = open(os.path.join(self.state_dir, self.filename_states + f"_{epoch}.p"), "wb")
        pickle.dump(saved_info, out)
        out.close()
        return


    def load_nn(self, filename_state, epoch):
        if not os.path.exists(os.path.join(self.state_dir, filename_state + f"_{epoch}.p")):
            raise ValueError(f"Filename {filename_state} does not exist")
        infile = open(os.path.join(self.state_dir, filename_state + f"_{epoch}.p"), "rb")
        loaded_info = pickle.load(infile)

        self.filename_states = filename_state
        self.progress_loaded = True

        attributes, neurons, state_info = loaded_info

        self.shape, self.activation, self.learning_rate, \
        self.num_epochs, self.progress_epoch, self.track_epoch, \
        self.seq_length, self.alphabet, self.alphabet2idx, self.idx2alphabet = attributes

        self.weights, self.weights_hidden, self. hidden_previous, \
        self.biases, self.memories, self.memories_hidden, \
        self.memories_biases = neurons

        self.losses = [state_info[0]]
        self.back_losses = [state_info[0]]
        self.texts = [state_info[2]]
        self.epoch_start = state_info[1]+1

        infile.close()

        print("=" * 10)
        print(f"Loading rnn state from file {filename_state}")
        print(f"Last error : {self.losses[-1]} \t Last epoch : {self.epoch_start-1}")
        print(f"Last text:\n{self.texts[-1]}")
        print("=" * 10)
        return

    def _setup_architecture(self):
        """Setup the neural network's architcecture

        Initialize the neural network structure to be used for weights and
        hidden layers
        """

        self.shape = (len(self.alphabet),) + self.shape + (len(self.alphabet),)

        for i in range(len(self.shape)-2):
            self.weights.append(self.init_weight_funcs[self.weight_init](self.shape[i+1], self.shape[i]))
            self.weights_hidden.append(self.init_weight_funcs[self.weight_init](self.shape[i+1], self.shape[i+1]))
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
        if not self.progress_loaded:
            self.alphabet = sorted(list(set(data)))
            for idx, char in enumerate(self.alphabet):
                self.alphabet2idx[char] = idx
                self.idx2alphabet[idx] = char

            self._setup_architecture()

            self.back_losses = [-np.log(1.0/len(self.alphabet)) * self.seq_length] # loss at time 0
            self.losses = []
            self.texts = []
        else:
            alphabet = sorted(list(set(data)))
            if alphabet != self.alphabet:
                raise ValueError("Input file alphabet does not match rnn's loaded alphabet")

        seq = 0
        for i in range(self.num_epochs):
            if seq + self.seq_length + 1 >= len(data): # reset the sequencing of data
                seq = 0
                for i in range(len(self.shape)-2):
                    self.hidden_previous[i] = np.zeros((self.shape[i+1], 1))

            inputs = [self.alphabet2idx[char] for char in data[seq:seq+self.seq_length]]
            targets = [self.alphabet2idx[char] for char in data[seq+1:seq+self.seq_length+1]]


            back_loss = self._backward(inputs, targets)
            loss = self.back_losses[-1] * 0.999 + back_loss * 0.001
            self.back_losses.append(loss)
            seq += self.seq_length
            if (i + self.epoch_start) % self.progress_epoch == 0:
                self.losses.append(loss)
                print("")
                print(f"iter {self.epoch_start + i}, loss: {loss}")
                gen = self.generate(self.idx2alphabet[inputs[0]], 200)
                print('----\n %s \n----' % (gen, ))
                if self.track_epoch or self.save_progress:
                    self.texts.append(gen)
                if self.save_progress:
                    self.save_nn(i+self.epoch_start)
            elif (i + self.epoch_start) % int(self.progress_epoch/10) == 0:
                print(".", end="", flush=True)




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
        dWy = np.zeros_like(self.weights[-1])
        dBy = np.zeros_like(self.biases[-1])
        dWhs = []
        dBhs  = []
        dHnexts = []

        for i in range(len(self.weights_hidden)):
            dWhs.append(np.zeros_like(self.weights_hidden[i]))
            dBhs.append(np.zeros_like(self.biases[i]))
            dHnexts.append(np.zeros_like(Hs[i][0]))

        for i in range(len(self.weights)-1):
            dWs.append(np.zeros_like(self.weights[i]))

        for i in reversed(range(len(inputs))):
            dY = np.copy(P[i])
            dY[targets[i]] -= 1
            dWy += dY.dot(Hs[-1][i].T)
            dBy += dY
            dNext = dY

            for j in reversed(range(len(self.shape)-2)):
                dH = self.weights[j+1].T.dot(dNext) + dHnexts[j]
                dNext = dH
                dHraw = (1 - Hs[j][i] * Hs[j][i]) * dH
                dBhs[j] += dHraw
                dWhs[j] += dHraw.dot(Hs[j][i-1].T)
                dHnexts[j] = self.weights_hidden[j].T.dot(dHraw)

                if j != 0:
                    dWs[j]+= dHraw.dot(Hs[j][i].T)

            dWs[0] += dHraw.dot(X[i].T)

        # clip to solve gradient explosion
        dWs = dWs + [dWy]
        dBs = dBhs + [dBy]
        for grad in dWs:
            np.clip(grad, -5, 5, out=grad)
        for grad in dBs:
            np.clip(grad, -5, 5, out=grad)
        for grad in dWhs:
            np.clip(grad, -5, 5, out=grad)

        for i in range(len(Hs)):
            self.hidden_previous[i] = deepcopy(Hs[i][len(inputs)-1])

        self.update_network(dWs, dWhs, dBs)

        return loss


    def update_network(self, dWs, dWhs, dBs):

        for i in range(len(self.weights)):
            self.memories[i] += dWs[i] * dWs[i]
            self.weights[i] += -self.learning_rate * dWs[i] / np.sqrt(self.memories[i] + 1e-8)

        for i in range(len(self.weights_hidden)):
            self.memories_hidden[i] += dWhs[i] * dWhs[i]
            self.weights_hidden[i] += -self.learning_rate * dWhs[i] / np.sqrt(self.memories_hidden[i] + 1e-8)

        for i in range(len(self.biases)):
            self.memories_biases[i] += dBs[i] * dBs[i]
            self.biases[i] += -self.learning_rate * dBs[i] / np.sqrt(self.memories_biases[i] + 1e-8)


    def _forward(self, inputs, hidden_in):
        hidden = []
        hidden_act = []

        hid = self.weights[0].dot(inputs) + self.weights_hidden[0].dot(hidden_in[0]) + self.biases[0]
        hidden.append(hid)
        hid_act = self.act_funcs[self.activation](hid)
        hidden_act.append(hid_act)

        for i in range(1, len(self.weights)-1):
            hid = self.weights[i].dot(hidden_act[i-1]) + self.weights_hidden[i].dot(hidden_in[i]) + self.biases[i]
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


def load_state_perfs(filename):
    import glob
    files = glob.glob(os.path.join(os.getcwd(), "states", f"{filename}*.p"))
    losses, texts, epochs = [], [], []
    for f in files:
        infile = open(f, "rb")
        loaded_info = pickle.load(infile)

        _, _, state_info = loaded_info

        losses.append(state_info[0])
        texts.append(state_info[2])
        epochs.append(state_info[1])

        infile.close()

    return losses, texts, epochs


def save_state_perfs_to_pickle(filename, losses, texts, epochs):
    outfile = open(filename, "wb")
    save = (losses, texts, epochs)
    pickle.dump(save, outfile)
    outfile.close()


def load_perfs_only_file(filename):
    infile = open(filename, "rb")
    load = pickle.load(infile)

    losses, texts, epochs = load[0], load[1], load[2]

    infile.close()

    return losses, texts, epochs


if __name__ == "__main__":
    data = open('input.txt', 'r').read() # should be simple plain text file

    rnn = RNN((128,128,128,),
              num_epochs=1000000,
              progress_epoch=1000,
              track_epoch=True,
              filename_states="",
              seq_length=100,
              save_progress=False)
    #rnn.load_nn("rnn_shakespeare_long", 0)
    rnn.train(data)
