import numpy as np
import mlp



iris_types = {"Setosa": 0,
              "Virginica": 2,
              "Versicolor": 1
              }

nn = mlp.MLP(shape=(4,4,3), batch_size=1, loss="mse", num_epochs=100, eta=0.01)


def get_outputs(op):
    ret = np.zeros((len(op), 3), dtype=int)
    for i in range(len(op)):
        idx = iris_types[op[i][-1]]
        ret[i, idx] = 1
    return ret


def read_csv(rl):
    ret = []
    for i in range(len(rl)):
        split = rl[i].split(",")
        if i != len(rl)-1:
            split[-1] = split[-1][:-1]
        ret.append(split)
    return ret


def get_inputs(ret, ip):
    for i in range(len(ip)):
        for j in range(len(ip[0])-1):
            ret[i, j] = ip[i][j]
    return ret

with open("iris.csv", "r") as f:
    rl = f.readlines()[1:]
    lines = read_csv(rl)
    inputs = np.zeros((len(lines), len(lines[0])-1), dtype=float)
    inp = get_inputs(inputs, lines)
    out = get_outputs(lines)

breakpoint()

indices = [*range(len(inp))]

np.random.seed(42)
np.random.shuffle(indices)

in_shuffled = inp[indices]
out_shuffled = out[indices]

nn.train(in_shuffled, out_shuffled)

predictions = nn.predict(in_shuffled, out_shuffled)

print(nn.score(predictions, out_shuffled))

breakpoint()
