import QCNN_circuit
import pennylane as qml
from pennylane import numpy as np
import autograd.numpy as anp
from tqdm import tqdm

def square_loss(labels, predictions):
    loss = 0
    for l, p in zip(labels, predictions):
        loss = loss + (l - p) ** 2
    loss = loss / len(labels)
    return loss

def cross_entropy(labels, predictions):
    loss = 0
    for l, p in zip(labels, predictions):
        c_entropy = l * (anp.log(p[l])) + (1 - l) * anp.log(1 - p[1 - l])
        loss = loss + c_entropy
    return -1 * loss

def cost(params, X, Y, U, U_params, embedding_type, circuit, cost_fn):
    predictions = [QCNN_circuit.QCNN(x, params, U, U_params, embedding_type, cost_fn=cost_fn) for x in X]
    if cost_fn == 'mse':
        loss = square_loss(Y, predictions)
    elif cost_fn == 'cross_entropy':
        loss = cross_entropy(Y, predictions)
    return loss

# Circuit training parameters
steps = 1460
learning_rate = 0.01
batch_size = 64     
def circuit_training(X_train, Y_train, U, U_params, embedding_type, circuit, cost_fn):
    if U == 'U_SU4_no_pooling' or U == 'U_SU4_1D' or U == 'U_9_1D':
        total_params = U_params * 3
    elif U =='U_SU4_1D_double':
        total_params = 63
    elif U =='U_SU4_1D_tf':
        total_params = 75
    else:
        total_params = U_params * 3 + 2 * 3
    
    params = np.random.randn(total_params, requires_grad=True)
    #opt = qml.NesterovMomentumOptimizer(stepsize=learning_rate)
    opt = qml.AdamOptimizer(stepsize=learning_rate)
    loss_history = []

    for it in tqdm(range(0,steps)):
        batch_index = np.random.randint(0, len(X_train), (batch_size,))
        X_batch = [X_train[i] for i in batch_index]
        Y_batch = [Y_train[i] for i in batch_index]
        params, cost_new = opt.step_and_cost(lambda v: cost(v, X_batch, Y_batch, U, U_params, embedding_type, circuit, cost_fn),params)
        loss_history.append(cost_new)
        if it % 10 == 0:
            print("iteration: ", it, " cost: ", cost_new)
    
    print('loss_history:', loss_history)
    print("---------------------------")
    print('training params:',params)
    with open(r'C:\Users\charu\Desktop\Projects\MNIST_QCNN\Result_BRATS\loss_history.txt', "a") as f:
        f.write(str(loss_history))
        f.write('\n')
    f.close()
    with open(r'C:\Users\charu\Desktop\Projects\MNIST_QCNN\Result_BRATS\training_params.txt', "a") as f:
        f.write(str(params))
        f.write('\n')
    f.close()
   
    return loss_history, params

