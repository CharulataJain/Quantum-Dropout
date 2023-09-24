import pennylane as qml
import unitary
import embedding

# Quantum Circuits for Convolutional layers
def conv_layer1(U, params):
    U(params, wires=[0, 15])
    for i in range(0, 14, 2):
        U(params, wires=[i, i + 1])
    for i in range(1, 12, 2):
        U(params, wires=[i, i + 1])

def conv_layer2(U, params):
    U(params, wires=[0, 10])
    U(params, wires=[0, 2])
    U(params, wires=[4, 6])
    U(params, wires=[8, 10])
    U(params, wires=[2, 4])
    U(params, wires=[0, 13])
    U(params, wires=[6, 14])
    U(params, wires=[0, 11])
    U(params, wires=[12, 14])
    U(params, wires=[13, 15])

def conv_layer3(U, params):
    U(params, wires=[0, 8])
    U(params, wires=[2, 6])
    U(params, wires=[4, 8])
    U(params, wires=[10, 14])
    U(params, wires=[12, 15])

# Quantum Circuits for Pooling layers
def pooling_layer1(V, params):
    for i in range(0, 14, 2):
        V(params, wires=[i + 1, i])

def pooling_layer2(V, params):
    V(params, wires=[2, 0])
    V(params, wires=[6, 4])
    V(params, wires=[10, 8])

def pooling_layer3(V, params):
    V(params, wires=[0, 16])
    V(params, wires=[4, 16])
    V(params, wires=[8, 16])

def QCNN_structure(U, params, U_params):
    param1 = params[0:U_params]
    param2 = params[U_params: 2 * U_params]
    param3 = params[2 * U_params: 3 * U_params]
    param4 = params[3 * U_params: 3 * U_params + 2]
    param5 = params[3 * U_params + 2: 3 * U_params + 4]
    param6 = params[3 * U_params + 4: 3 * U_params + 6]

    # Pooling Ansatz1 is used by default
    conv_layer1(U, param1)
    pooling_layer1(unitary.Pooling_ansatz1, param4)
    conv_layer2(U, param2)
    pooling_layer2(unitary.Pooling_ansatz1, param5)
    conv_layer3(U, param3)
    pooling_layer3(unitary.Pooling_ansatz1, param6)


def QCNN_structure_without_pooling(U, params, U_params):
    param1 = params[0:U_params]
    param2 = params[U_params: 2 * U_params]
    param3 = params[2 * U_params: 3 * U_params]

    conv_layer1(U, param1)
    conv_layer2(U, param2)
    conv_layer3(U, param3)

def QCNN_1D_circuit(U, params, U_params):
    param1 = params[0: U_params]
    param2 = params[U_params: 2 * U_params]
    param3 = params[2 * U_params: 3 * U_params]

    for i in range(0, 10, 2):
        U(param1, wires=[i, i + 1])
    for i in range(1, 9, 2):
        U(param1, wires=[i, i + 1])

    U(param2, wires=[2, 3])
    U(param2, wires=[4, 5])
    U(param3, wires=[3, 4])

def QCNN_1D_circuit_tf(U, params, U_params):
    param1 = params[0: U_params]
    param2 = params[U_params: U_params * 2]
    param3 = params[U_params * 2: U_params * 3]
    param4 = params[U_params * 3: U_params * 3 + 12]
    param5 = params[U_params * 3 + 12: U_params * 3 + 24]
    param6 = params[U_params * 3 + 24: U_params * 3 + 36]

    for i in range(0, 14, 2):
        U(param1, wires=[i, i + 1])
    for i in range(1, 13, 2):
        U(param1, wires=[i, i + 1])

    for i in range(0, 14, 2):
        unitary.U_SO4(param4, wires=[i, i + 1])
    for i in range(1, 13, 2):
        unitary.U_SO4(param4, wires=[i, i + 1])

    U(param2, wires=[2, 3])
    U(param2, wires=[4, 5])
    U(param2, wires=[6, 7])
    U(param2, wires=[8, 9])
    U(param2, wires=[10, 11])
    U(param2, wires=[12, 13])

    unitary.U_SO4(param5, wires=[2, 3])
    unitary.U_SO4(param5, wires=[4, 5])
    unitary.U_SO4(param5, wires=[6, 7])
    unitary.U_SO4(param5, wires=[8, 9])
    unitary.U_SO4(param5, wires=[10, 11])
    unitary.U_SO4(param5, wires=[12, 13])

    U(param3, wires=[3, 4])
    unitary.U_SO4(param6, wires=[3, 4])
    U(param3, wires=[5, 6])
    unitary.U_SO4(param6, wires=[5, 6])

    # Additional gates for remaining qubits


def QCNN_1D_circuit_double(U, params, U_params):
    param1 = params[0: U_params]
    param2 = params[U_params: U_params * 2]
    param3 = params[U_params * 2: U_params * 3]
    param4 = params[U_params * 3: U_params * 3 + 12]
    param5 = params[U_params * 3 + 12: U_params * 3 + 24]
    param6 = params[U_params * 3 + 24: U_params * 3 + 36]

    for i in range(0, 10, 2):
        U(param1, wires=[i, i + 1])
    for i in range(1, 9, 2):
        U(param1, wires=[i, i + 1])

    for i in range(0, 10, 2):
        unitary.U_SO4(param4, wires=[i, i + 1])
    for i in range(1, 9, 2):
        unitary.U_SO4(param4, wires=[i, i + 1])

    U(param2, wires=[2, 3])
    U(param2, wires=[4, 5])
    U(param2, wires=[6, 7])
    U(param2, wires=[8, 9])
    U(param2, wires=[10, 11])

    unitary.U_SO4(param5, wires=[2, 3])
    unitary.U_SO4(param5, wires=[4, 5])
    unitary.U_SO4(param5, wires=[6, 7])
    unitary.U_SO4(param5, wires=[8, 9])
    unitary.U_SO4(param5, wires=[10, 11])


    U(param3, wires=[3, 4])
    unitary.U_SO4(param6, wires=[3, 4])

dev = qml.device('default.qubit', wires=16)
@qml.qnode(dev)
def QCNN(X, params, U, U_params, embedding_type='Amplitude', cost_fn='cross_entropy'):

    # Data Embedding
    embedding.data_embedding(X, embedding_type=embedding_type)

    # Quantum Convolutional Neural Network
    if U == 'U_TTN':
        QCNN_structure(unitary.U_TTN, params, U_params)
    elif U == 'U_5':
        QCNN_structure(unitary.U_5, params, U_params)
    elif U == 'U_6':
        QCNN_structure(unitary.U_6, params, U_params)
    elif U == 'U_9':
        QCNN_structure(unitary.U_9, params, U_params)
    elif U == 'U_13':
        QCNN_structure(unitary.U_13, params, U_params)
    elif U == 'U_14':
        QCNN_structure(unitary.U_14, params, U_params)
    elif U == 'U_15':
        QCNN_structure(unitary.U_15, params, U_params)
    elif U == 'U_SO4':
        QCNN_structure(unitary.U_SO4, params, U_params)
    elif U == 'U_SU4':
        QCNN_structure(unitary.U_SU4, params, U_params)
    elif U == 'U_SU4_no_pooling':
        QCNN_structure_without_pooling(unitary.U_SU4, params, U_params)
    elif U == 'U_SU4_1D':
        QCNN_1D_circuit(unitary.U_SU4, params, U_params)
    elif U == 'U_9_1D':
        QCNN_1D_circuit(unitary.U_9, params, U_params)
    elif U == 'U_SU4_1D_double':
        QCNN_1D_circuit_double(unitary.U_SU4, params, U_params)
    elif U == 'U_SU4_1D_tf':
        QCNN_1D_circuit_tf(unitary.U_SU4, params, U_params)
    else:
        print("Invalid Unitary Ansatze")
        return False

    if cost_fn == 'mse':
        result = qml.expval(qml.PauliZ(6))
    elif cost_fn == 'cross_entropy':
        result = qml.probs(wires=6)

    return result
