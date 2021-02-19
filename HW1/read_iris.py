import numpy as np

iris_types = {"Setosa": 0,
              "Virginica": 2,
              "Versicolor": 1
              }

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

def read_iris(shuffle=True, test=True):
    with open("iris/iris.csv", "r") as f:
        rl = f.readlines()[1:]
        lines = read_csv(rl)
        inputs = np.zeros((len(lines), len(lines[0])-1), dtype=float)
        inp = get_inputs(inputs, lines)
        out = get_outputs(lines)
    num = len(inp)
    test_split = int(num*0.75)
    if shuffle and test:
        indices = [*range(len(inp))]

        np.random.seed(42)
        np.random.shuffle(indices)

        in_shuffled = inp[indices]
        out_shuffled = out[indices]

        ret_in = in_shuffled[:test_split]
        ret_test_in = in_shuffled[test_split:]
        ret_out = out_shuffled[:test_split]
        ret_test_out = out_shuffled[test_split:]
    else:
        ret_in = inp[:test_split]
        ret_test_in = inp[test_split:]
        ret_out = out[:test_split]
        ret_test_out = out[test_split:]


    return ret_in, ret_out, ret_test_in, ret_test_out
