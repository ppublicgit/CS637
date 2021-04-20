import numpy as np
from copy import deepcopy

np.random.seed(42)

class RNN_VANILLA:
    def __init__(self, **kwargs):
        self.num_iterations = kwargs.get("num_iterations", 1000000)
        self.alphabet = None
        self.alphabet2idx = {}
        self.idx2alphabet = {}
        self.num_hidden = kwargs.get("num_hidden", 100)
        self.seq_length = kwargs.get("seq_length", 25)
        self.learning_rate = kwargs.get("learning_rate", 1e-1)

        self.Wxh             = None
        self.Whh             = None
        self.Why             = None
        self.Bh              = None
        self.By              = None
        self.hidden_previous = None


    def setup_architecture(self):
        self.Wxh = np.random.randn(self.num_hidden, len(self.alphabet)) * 0.01
        self.Whh = np.random.randn(self.num_hidden, self.num_hidden) * 0.01
        self.Why = np.random.randn(len(self.alphabet), self.num_hidden) * 0.01
        self.Bh  = np.zeros((self.num_hidden, 1))
        self.By  = np.zeros((len(self.alphabet), 1))

        self.mWxh = np.zeros_like(self.Wxh)
        self.mWhh = np.zeros_like(self.Whh)
        self.mWhy = np.zeros_like(self.Why)
        self.mBh  = np.zeros_like(self.Bh)
        self.mBy  = np.zeros_like(self.By)


    def train(self, data):
        self.alphabet = sorted(list(set(data)))
        for idx, char in enumerate(self.alphabet):
            self.alphabet2idx[char] = idx
            self.idx2alphabet[idx] = char

        self.setup_architecture()

        self.losses = [-np.log(1.0/len(self.alphabet)) * self.seq_length] # loss at time 0
        self.hidden_previous = np.zeros((self.num_hidden, 1)) # initialize the hidden layer RNN memory

        seq = 0
        for i in range(self.num_iterations):
            if seq + self.seq_length + 1 >= len(data): # reset the sequencing of data
                seq = 0
                self.hidden_previous = np.zeros((self.num_hidden, 1))

            # get inputs and targets for the sequence of chars to learn from
            inputs = [self.alphabet2idx[char] for char in data[seq:seq+self.seq_length]]
            targets = [self.alphabet2idx[char] for char in data[seq+1:seq+self.seq_length+1]]


            back_loss = self.backward(inputs, targets)
            loss = self.losses[-1] * 0.999 + back_loss * 0.001
            self.losses.append(loss)

            seq += self.seq_length
            if i % 1000 == 0:
                print(f"iter {i}, loss: {loss}")
                gen = self.generate(self.idx2alphabet[inputs[0]], 200)
                print('----\n %s \n----' % (gen, ))
        return


    def forward(self, inputs, hidden_in):
        # pass inputs into through hidden layer
        hidden = self.Wxh.dot(inputs) + self.Whh.dot(hidden_in) + self.Bh
        # activate hidden layer
        hidden_act = np.tanh(hidden)
        # pass hidden layer through to output layer
        output = self.Why.dot(hidden_act) + self.By
        # activate (normalize) outputs
        output_norm = np.exp(output) / np.sum(np.exp(output))
        return hidden_act, output, output_norm


    def backward(self, inputs, targets):
        X, H, Y, P = {}, {}, {}, {}
        H[-1] = deepcopy(self.hidden_previous)
        loss = 0

        #forward pass the to predict chars
        for i in range(len(inputs)):
            X[i] = np.zeros((len(self.alphabet), 1))
            X[i][inputs[i]] = 1
            H[i], Y[i], P[i] = self.forward(X[i], H[i-1])
            loss += -np.log(P[i][targets[i],0])

        #backward pass to update weights
        # initialize derivatives
        dWxh, dWhh, dWhy = np.zeros_like(self.Wxh), np.zeros_like(self.Whh), np.zeros_like(self.Why)
        dBh, dBy = np.zeros_like(self.Bh), np.zeros_like(self.By)
        dHnext = np.zeros_like(H[0])

        for i in reversed(range(len(inputs))):#[*range(len(inputs))][::-1]:
            dY = np.copy(P[i])
            dY[targets[i]] -= 1
            dWhy += dY.dot(H[i].T)
            dBy += dY
            dH = self.Why.T.dot(dY) + dHnext
            dHraw = (1 - H[i] * H[i]) * dH
            dBh += dHraw
            dWxh += dHraw.dot(X[i].T)
            dWhh += dHraw.dot(H[i-1].T)
            dHnext = self.Whh.T.dot(dHraw)

        # clip to solve gradient explosion
        for grad in [dWxh, dWhh, dWhy, dBh, dBy]:
            np.clip(grad, -5, 5, out=grad)


        self.hidden_previous = deepcopy(H[len(inputs)-1])


        self.update_network(dWxh, dWhh, dWhy, dBh, dBy)

        return loss

    def update_network(self, dWxh, dWhh, dWhy, dBh, dBy):
        #
        for node, derivative, memory in zip([self.Wxh, self.Whh, self.Why, self.Bh, self.By],
                                            [dWxh, dWhh, dWhy, dBh, dBy],
                                            [self.mWxh, self.mWhh, self.mWhy, self.mBh, self.mBy]):
            memory += derivative * derivative
            node += -self.learning_rate * derivative / np.sqrt(memory + 1e-8)


    def generate(self, char, num_chars):
        prev_char_idx = self.alphabet2idx[char]
        X = np.zeros((len(self.alphabet), 1))
        X[prev_char_idx] = 1
        hidden = self.hidden_previous
        indexes = []
        for i in range(num_chars):
            hidden, _, probability = self.forward(X, hidden)
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

    rnn = RNN_VANILLA()
    rnn.train(data)
