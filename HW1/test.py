import mlp


a = mlp.MLP(shape=(4,3,5))

inputs = np.zeros((4,10))

outputs = np.zeros((5, 10))

breakpoint()

a.train(inputs, outputs)
