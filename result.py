import Benchmarking


"""
Here are possible combinations of benchmarking user could try.
Unitaries: ['U_TTN', 'U_5', 'U_6', 'U_9', 'U_13', 'U_14', 'U_15', 'U_SO4', 'U_SU4', 'U_SU4_no_pooling', 'U_SU4_1D', 'U_9_1D']
U_num_params: [2, 10, 10, 2, 6, 6, 4, 6, 15, 15, 15, 2]
Encodings: ['resize256', 'pca8', 'autoencoder8', 'pca16-compact', 'autoencoder16-compact', 'pca32-1', 'autoencoder32-1',
            'pca16-1', 'autoencoder16-1', 'pca30-1', 'autoencoder30-1', 'pca12-1', 'autoencoder12-1']
dataset: 'mnist' 
circuit: 'QCNN'
cost_fn: 'mse' or 'cross_entropy'
Note: when using 'mse' as cost_fn binary="True" is recommended, when using 'cross_entropy' as cost_fn must be binary="False".
"""

Unitaries = ['U_SU4_1D_tf']
U_num_params = [15, 15, 15]
Encodings = ['pca8']
dataset = 'brats2019'
classes = ['0','1']
binary = False
cost_fn = 'mse'

Benchmarking.Benchmarking(dataset, classes, Unitaries, U_num_params, Encodings, circuit='QCNN', cost_fn=cost_fn, binary=binary)
#U_9 autoencoder8 with cross_entropy
#U_SU4_no_pooling autoencoder8 mse
#U_SU4_no_pooling resize256
#U_9 autoencoder8  