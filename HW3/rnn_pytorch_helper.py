import numpy as np
import torch
from torch import nn
import torch.nn.functional as F


def one_hot_encode(arr, num_labels):
    # initialize
    one_hot = np.zeros((np.multiply(*arr.shape), num_labels), dtype=np.float32)
    # Fill the appropriate elements with ones
    one_hot[np.arange(one_hot.shape[0]), arr.flatten()] = 1.
    # reshape to original shape
    one_hot = one_hot.reshape((*arr.shape, num_labels))
    return one_hot


# Defining method to make mini-batches for training
def get_batches(arr, batch_size, seq_length):
    """Generator for batching data

    array is dataset to batch
    batch_size is the numbber of sequences in a batch
    seq_length is the length of the inputs for each row in batch
    """

    # determine number of batches we can make from data
    batch_size_total = batch_size * seq_length
    n_batches = len(arr)//batch_size_total

    # Keep only enough characters to make full batches
    arr = arr[:n_batches * batch_size_total]
    # Reshape into batch_size rows
    arr = arr.reshape((batch_size, -1))

    # iterate through the array, one sequence at a time
    for n in range(0, arr.shape[1], seq_length):
        # The features
        x = arr[:, n:n+seq_length]
        # The targets, shifted by one
        y = np.zeros_like(x)
        try:
            y[:, :-1], y[:, -1] = x[:, 1:], arr[:, n+seq_length]
        except IndexError:
            y[:, :-1], y[:, -1] = x[:, 1:], arr[:, 0]
        yield x, y


def train_lstm(net, data,
          opt, criterion,
          epochs=10, batch_size=10,
          seq_length=50, clip=5,
          val_frac=0.1):
    # set network to training mode
    net.train()

    # create training and validation data
    val_idx = int(len(data)*(1-val_frac))
    data, val_data = data[:val_idx], data[val_idx:]

    counter = 0
    n_chars = len(net.alphabet)

    train_losses_ret, val_losses_ret, samples = [], [], []
    # loop through our epochs
    for e in range(epochs):
        # initialize hidden state
        h = net.init_hidden(batch_size)
        first_batch = True
        # loop through all the batches of the data
        for x, y in get_batches(data, batch_size, seq_length):
            counter += 1
            # One-hot encode our data and convert to torch tensors
            x = one_hot_encode(x, n_chars)
            inputs, targets = torch.from_numpy(x), torch.from_numpy(y)

            # Creating new variables for the hidden state, otherwise
            # we'd backprop through the entire training history
            h = tuple([each.data for each in h])

            # zero accumulated gradients
            net.zero_grad()

            # forward pass
            output, h = net(inputs, h)

            # calculate loss and backpropagation
            loss = criterion(output, targets.view(batch_size*seq_length).long())
            loss.backward()
            # clip gradient to prevent exploding gradients for RNN/LSTMs
            nn.utils.clip_grad_norm_(net.parameters(), clip)
            opt.step()

            # loss stats
            if first_batch:
                # Get validation loss
                # get val hidden initial state
                val_h = net.init_hidden(batch_size)
                val_losses = []
                # set network to eval mode
                net.eval()
                # loop through batches
                for x, y in get_batches(val_data, batch_size, seq_length):
                    # One-hot encode our data and make them Torch tensors
                    x = one_hot_encode(x, n_chars)
                    x, y = torch.from_numpy(x), torch.from_numpy(y)

                    # Creating new variables for the hidden state, otherwise
                    # we'd backprop through the entire training history
                    val_h = tuple([each.data for each in val_h])

                    inputs, targets = x, y

                    # forward pass and calculate loss
                    output, val_h = net(inputs, val_h)
                    val_loss = criterion(output, targets.view(batch_size*seq_length).long())

                    val_losses.append(val_loss.item())


                # print update of performance
                print("Epoch: {}/{}...".format(e+1, epochs),
                      "Step: {}...".format(counter),
                      "Loss: {:.4f}...".format(loss.item()),
                      "Val Loss: {:.4f}".format(np.mean(val_losses)))

                sample = generate(net, 200, prime='A', top_k=5)
                # save loss info and set first batch to false to not trigger validation until next epoch
                train_losses_ret.append(loss.item())
                val_losses_ret.append(np.mean(val_losses))
                samples.append(sample)
                first_batch = False

                net.train() # reset to train mode after iterationg through validation data

    return train_losses_ret, val_losses_ret, samples


def generate(net, size, prime='The', top_k=None):
    """Generate a sequence of characters

    net is the neural network
    size is the number of chars to generate
    prime is the first sequence of chars to kick off the generation
    """
    net.eval() # eval mode

    # First off, run through the prime characters
    chars = [ch for ch in prime]
    h = net.init_hidden(1)
    for ch in prime:
        char, h = generate_helper(net, ch, h, top_k=top_k)

    chars.append(char)

    # Now pass in the previous character and get a new one
    for ii in range(size):
        char, h = generate_helper(net, chars[-1], h, top_k=top_k)
        chars.append(char)

    return ''.join(chars)


# Defining a method to generate the next character
def generate_helper(net, char, h=None, top_k=None):
        ''' Given a character, predict the next character.
            Returns the predicted character and the hidden state.
        '''

        # tensor inputs
        x = np.array([[net.char2int[char]]])
        x = one_hot_encode(x, len(net.alphabet))
        inputs = torch.from_numpy(x)

        # detach hidden state from history
        h = tuple([each.data for each in h])
        # get the output of the model
        out, h = net(inputs, h)

        # get the character probabilities
        p = F.softmax(out, dim=1).data

        # get top characters
        if top_k is None:
            top_ch = np.arange(len(net.alphabet))
        else:
            p, top_ch = p.topk(top_k)
            top_ch = top_ch.numpy().squeeze()

        # select the likely next character with some element of randomness
        p = p.numpy().squeeze()
        char = np.random.choice(top_ch, p=p/p.sum())

        # return the encoded value of the predicted char and the hidden state
        return net.int2char[char], h


def save_torch_state(filename):
    state = {"char2int": net.char2int,
             "int2char": int2char,
             "alphabet": net.alphabet,
             "state_dict": net.state_dict()}
    torch.save(state, filename)


def load_torch_state(filename):
    state = torch.load(filename)

    new = LSTM(state["alphabet"])

    new.load_state_dict(state["state_dict"])
    new.int2char = state["int2char"]
    new.char2int = state["char2int"]

    return new


class LSTM(nn.Module):

    def __init__(self, tokens, n_hidden=512, n_layers=3, drop_prob=0.5, lr=0.001):
        super().__init__()
        self.drop_prob = drop_prob
        self.n_layers = n_layers
        self.n_hid = n_hidden
        self.lr = lr

        # creating alphabet and encoders
        self.alphabet = tokens
        self.int2char = dict(enumerate(self.alphabet))
        self.char2int = {ch: ii for ii, ch in self.int2char.items()}

        # create the lstm layers
        self.lstm = nn.LSTM(len(self.alphabet), self.n_hid, self.n_layers, dropout=self.drop_prob, batch_first=True)

        # create a dropout layer for post lstm
        self.dropout = nn.Dropout(self.drop_prob)

        # create a fully connected feed forward layer
        self.fc = nn.Linear(self.n_hid, len(self.alphabet))

    def forward(self, x, hidden):
        """Forward pass

        x is the inputs
        hidden are the hidden node states
        """
        # pass through the lstm layer
        lstm_out, hidden = self.lstm(x, hidden)
        # pass through dropout layer
        dropout_out = self.dropout(lstm_out)
        # reshape outputs for fully connected layer
        out = dropout_out.contiguous().view(-1, self.n_hid)
        # pass through the fully connected layer
        out = self.fc(out)
        # return final output and the hidden states
        return out, hidden

    def init_hidden(self, batch_size):
        """ Initialize hidden state"""
        # Create two new tensors with sizes n_layers x batch_size x n_hidden,
        # initialized to zero, for hidden state and cell state of LSTM
        weight = next(self.parameters()).data

        hidden = (weight.new(self.n_layers, batch_size, self.n_hid).zero_(),
                  weight.new(self.n_layers, batch_size, self.n_hid).zero_())
        return hidden
