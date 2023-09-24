import pennylane as qml
import embedding
import random
import data
import Benchmarking
''' to do here 
    1. make list with of qml.circuits 
    2. make dropout alo to drop only specific vals from that circuit
    3. 
'''
def gates_percent_drop(lst, percent, exceptions):
    num_elements_to_delete = int(len(lst) * (percent))
    indices_to_delete = random.sample(range(len(lst)), num_elements_to_delete)
    indices_to_delete.sort(reverse=True) # Sort in reverse order to preserve sequence during deletion
    
    for index in indices_to_delete:
        if lst[index] not in exceptions:
            print(lst[index])
            del lst[index]
    
    return lst

def QCNN_1D_circuit_U_SU4_1D(params, U_params, prcnt):
    
    
          
    #Convolutional Layer 1
    conv1 = ['qml.U3(param1[0], param1[1], param1[2], wires=0)','qml.U3(param1[3], param1[4], param1[5], wires=1)','qml.CNOT(wires=[0, 1])','qml.RY(param1[6], wires=0)','qml.RZ(param1[7], wires=1)','qml.CNOT(wires= [1,0])','qml.RY(param1[8], wires=0)','qml.CNOT(wires=[0, 1])','qml.U3(param1[9], param1[10], param1[11], wires=0)','qml.U3(param1[12], param1[13], param1[14], wires=1)',
             'qml.U3(param1[0], param1[1], param1[2], wires=1)','qml.U3(param1[3], param1[4], param1[5], wires=2)','qml.CNOT(wires=[1, 2])','qml.RY(param1[6], wires=1)','qml.RZ(param1[7], wires=2)','qml.CNOT(wires= [2,1])','qml.RY(param1[8], wires=1)','qml.CNOT(wires=[1, 2])','qml.U3(param1[9], param1[10], param1[11], wires=1)','qml.U3(param1[12], param1[13], param1[14], wires=2)',
             'qml.U3(param1[0], param1[1], param1[2], wires=2)','qml.U3(param1[3], param1[4], param1[5], wires=3)','qml.CNOT(wires=[2, 3])','qml.RY(param1[6], wires=2)','qml.RZ(param1[7], wires=3)','qml.CNOT(wires= [3,2])','qml.RY(param1[8], wires=2)','qml.CNOT(wires=[2, 3])','qml.U3(param1[9], param1[10], param1[11], wires=2)','qml.U3(param1[12], param1[13], param1[14], wires=3)',
             'qml.U3(param1[0], param1[1], param1[2], wires=3)','qml.U3(param1[3], param1[4], param1[5], wires=4)','qml.CNOT(wires=[3, 4])','qml.RY(param1[6], wires=3)','qml.RZ(param1[7], wires=4)','qml.CNOT(wires= [4,3])','qml.RY(param1[8], wires=3)','qml.CNOT(wires=[3, 4])','qml.U3(param1[9], param1[10], param1[11], wires=3)','qml.U3(param1[12], param1[13], param1[14], wires=4)',
             'qml.U3(param1[0], param1[1], param1[2], wires=4)','qml.U3(param1[3], param1[4], param1[5], wires=5)','qml.CNOT(wires=[4, 5])','qml.RY(param1[6], wires=4)','qml.RZ(param1[7], wires=5)','qml.CNOT(wires= [5,4])','qml.RY(param1[8], wires=4)','qml.CNOT(wires=[4, 5])','qml.U3(param1[9], param1[10], param1[11], wires=4)','qml.U3(param1[12], param1[13], param1[14], wires=5)',
             'qml.U3(param1[0], param1[1], param1[2], wires=5)','qml.U3(param1[3], param1[4], param1[5], wires=6)','qml.CNOT(wires=[5, 6])','qml.RY(param1[6], wires=5)','qml.RZ(param1[7], wires=6)','qml.CNOT(wires= [6,5])','qml.RY(param1[8], wires=5)','qml.CNOT(wires=[5, 6])','qml.U3(param1[9], param1[10], param1[11], wires=5)','qml.U3(param1[12], param1[13], param1[14], wires=6)',
             'qml.U3(param1[0], param1[1], param1[2], wires=6)','qml.U3(param1[3], param1[4], param1[5], wires=7)','qml.CNOT(wires=[6, 7])','qml.RY(param1[6], wires=6)','qml.RZ(param1[7], wires=7)','qml.CNOT(wires= [7,6])','qml.RY(param1[8], wires=6)','qml.CNOT(wires=[6, 7])','qml.U3(param1[9], param1[10], param1[11], wires=6)','qml.U3(param1[12], param1[13], param1[14], wires=7)',
             'qml.U3(param1[0], param1[1], param1[2], wires=7)','qml.U3(param1[3], param1[4], param1[5], wires=8)','qml.CNOT(wires=[7, 8])','qml.RY(param1[6], wires=7)','qml.RZ(param1[7], wires=8)','qml.CNOT(wires= [0,7])','qml.RY(param1[8], wires=7)','qml.CNOT(wires=[7, 0])','qml.U3(param1[9], param1[10], param1[11], wires=7)','qml.U3(param1[12], param1[13], param1[14], wires=0)',
             'qml.U3(param1[0], param1[1], param1[2], wires=8)','qml.U3(param1[3], param1[4], param1[5], wires=9)','qml.CNOT(wires=[8, 9])','qml.RY(param1[6], wires=8)','qml.RZ(param1[7], wires=9)','qml.CNOT(wires= [0,7])','qml.RY(param1[8], wires=7)','qml.CNOT(wires=[7, 0])','qml.U3(param1[9], param1[10], param1[11], wires=7)','qml.U3(param1[12], param1[13], param1[14], wires=0)',
             'qml.U3(param1[0], param1[1], param1[2], wires=9)','qml.U3(param1[3], param1[4], param1[5], wires=10)','qml.CNOT(wires=[9, 10])','qml.RY(param1[6], wires=9)','qml.RZ(param1[7], wires=10)','qml.CNOT(wires= [0,7])','qml.RY(param1[8], wires=7)','qml.CNOT(wires=[7, 0])','qml.U3(param1[9], param1[10], param1[11], wires=7)','qml.U3(param1[12], param1[13], param1[14], wires=0)',
             'qml.U3(param1[0], param1[1], param1[2], wires=10)','qml.U3(param1[3], param1[4], param1[5], wires=11)','qml.CNOT(wires=[10, 11])','qml.RY(param1[6], wires=10)','qml.RZ(param1[7], wires=11)','qml.CNOT(wires= [0,7])','qml.RY(param1[8], wires=7)','qml.CNOT(wires=[7, 0])','qml.U3(param1[9], param1[10], param1[11], wires=7)','qml.U3(param1[12], param1[13], param1[14], wires=0)',
             'qml.U3(param1[0], param1[1], param1[2], wires=11)','qml.U3(param1[3], param1[4], param1[5], wires=12)','qml.CNOT(wires=[11, 12])','qml.RY(param1[6], wires=11)','qml.RZ(param1[7], wires=12)','qml.CNOT(wires= [0,7])','qml.RY(param1[8], wires=7)','qml.CNOT(wires=[7, 0])','qml.U3(param1[9], param1[10], param1[11], wires=7)','qml.U3(param1[12], param1[13], param1[14], wires=0)',
             'qml.U3(param1[0], param1[1], param1[2], wires=12)','qml.U3(param1[3], param1[4], param1[5], wires=13)','qml.CNOT(wires=[12, 13])','qml.RY(param1[6], wires=12)','qml.RZ(param1[7], wires=13)','qml.CNOT(wires= [0,7])','qml.RY(param1[8], wires=7)','qml.CNOT(wires=[7, 0])','qml.U3(param1[9], param1[10], param1[11], wires=7)','qml.U3(param1[12], param1[13], param1[14], wires=0)',
             'qml.U3(param1[0], param1[1], param1[2], wires=13)','qml.U3(param1[3], param1[4], param1[5], wires=14)','qml.CNOT(wires=[13, 14])','qml.RY(param1[6], wires=13)','qml.RZ(param1[7], wires=14)','qml.CNOT(wires= [0,7])','qml.RY(param1[8], wires=7)','qml.CNOT(wires=[7, 0])','qml.U3(param1[9], param1[10], param1[11], wires=7)','qml.U3(param1[12], param1[13], param1[14], wires=0)',
             'qml.U3(param1[0], param1[1], param1[2], wires=14)','qml.U3(param1[3], param1[4], param1[5], wires=15)','qml.CNOT(wires=[14, 15])','qml.RY(param1[6], wires=14)','qml.RZ(param1[7], wires=15)','qml.CNOT(wires= [0,7])','qml.RY(param1[8], wires=7)','qml.CNOT(wires=[7, 0])','qml.U3(param1[9], param1[10], param1[11], wires=7)','qml.U3(param1[12], param1[13], param1[14], wires=0)',
             'qml.U3(param1[0], param1[1], param1[2], wires=15)','qml.U3(param1[3], param1[4], param1[5], wires=16)','qml.CNOT(wires=[15, 0])','qml.RY(param1[6], wires=15)','qml.RZ(param1[7], wires=16)','qml.CNOT(wires= [0,7])','qml.RY(param1[8], wires=7)','qml.CNOT(wires=[7, 0])','qml.U3(param1[9], param1[10], param1[11], wires=7)','qml.U3(param1[12], param1[13], param1[14], wires=0)',

             'qml.RY(param4[0], wires=0)','qml.RY(param4[1], wires=1)','qml.CNOT(wires=[0, 1])','qml.RY(param4[2], wires=0)','qml.RY(param4[3], wires=1)','qml.CNOT(wires=[0, 1])','qml.RY(param4[4], wires=0)','qml.RY(param4[5], wires=1)',
             'qml.RY(param4[0], wires=1)','qml.RY(param4[1], wires=2)','qml.CNOT(wires=[1, 2])','qml.RY(param4[2], wires=1)','qml.RY(param4[3], wires=2)','qml.CNOT(wires=[1, 2])','qml.RY(param4[4], wires=1)','qml.RY(param4[5], wires=2)',
             'qml.RY(param4[0], wires=2)','qml.RY(param4[1], wires=3)','qml.CNOT(wires=[2, 3])','qml.RY(param4[2], wires=2)','qml.RY(param4[3], wires=3)','qml.CNOT(wires=[2, 3])','qml.RY(param4[4], wires=2)','qml.RY(param4[5], wires=3)',
             'qml.RY(param4[0], wires=3)','qml.RY(param4[1], wires=4)','qml.CNOT(wires=[3, 4])','qml.RY(param4[2], wires=3)','qml.RY(param4[3], wires=4)','qml.CNOT(wires=[3, 4])','qml.RY(param4[4], wires=3)','qml.RY(param4[5], wires=4)',
             'qml.RY(param4[0], wires=4)','qml.RY(param4[1], wires=5)','qml.CNOT(wires=[4, 5])','qml.RY(param4[2], wires=4)','qml.RY(param4[3], wires=5)','qml.CNOT(wires=[4, 5])','qml.RY(param4[4], wires=4)','qml.RY(param4[5], wires=5)',
             'qml.RY(param4[0], wires=5)','qml.RY(param4[1], wires=6)','qml.CNOT(wires=[5, 6])','qml.RY(param4[2], wires=5)','qml.RY(param4[3], wires=6)','qml.CNOT(wires=[5, 6])','qml.RY(param4[4], wires=5)','qml.RY(param4[5], wires=6)',
             'qml.RY(param4[0], wires=6)','qml.RY(param4[1], wires=7)','qml.CNOT(wires=[6, 7])','qml.RY(param4[2], wires=6)','qml.RY(param4[3], wires=7)','qml.CNOT(wires=[6, 7])','qml.RY(param4[4], wires=6)','qml.RY(param4[5], wires=7)',
             'qml.RY(param4[0], wires=7)','qml.RY(param4[1], wires=8)','qml.CNOT(wires=[7, 8])','qml.RY(param4[2], wires=7)','qml.RY(param4[3], wires=8)','qml.CNOT(wires=[7, 8])','qml.RY(param4[4], wires=7)','qml.RY(param4[5], wires=0)',
             'qml.RY(param4[0], wires=8)','qml.RY(param4[1], wires=9)','qml.CNOT(wires=[8, 9])','qml.RY(param4[2], wires=8)','qml.RY(param4[3], wires=9)','qml.CNOT(wires=[8, 9])','qml.RY(param4[4], wires=9)','qml.RY(param4[5], wires=0)',
             'qml.RY(param4[0], wires=9)','qml.RY(param4[1], wires=10)','qml.CNOT(wires=[9, 10])','qml.RY(param4[2], wires=9)','qml.RY(param4[3], wires=10)','qml.CNOT(wires=[9, 10])','qml.RY(param4[4], wires=10)','qml.RY(param4[5], wires=0)',
             'qml.RY(param4[0], wires=10)','qml.RY(param4[1], wires=11)','qml.CNOT(wires=[10, 11])','qml.RY(param4[2], wires=10)','qml.RY(param4[3], wires=11)','qml.CNOT(wires=[10, 11])','qml.RY(param4[4], wires=11)','qml.RY(param4[5], wires=0)',
             'qml.RY(param4[0], wires=11)','qml.RY(param4[1], wires=12)','qml.CNOT(wires=[11, 12])','qml.RY(param4[2], wires=11)','qml.RY(param4[3], wires=12)','qml.CNOT(wires=[11, 12])','qml.RY(param4[4], wires=12)','qml.RY(param4[5], wires=0)',
             'qml.RY(param4[0], wires=12)','qml.RY(param4[1], wires=13)','qml.CNOT(wires=[12, 13])','qml.RY(param4[2], wires=12)','qml.RY(param4[3], wires=13)','qml.CNOT(wires=[12, 13])','qml.RY(param4[4], wires=13)','qml.RY(param4[5], wires=0)',
             'qml.RY(param4[0], wires=13)','qml.RY(param4[1], wires=14)','qml.CNOT(wires=[13, 14])','qml.RY(param4[2], wires=13)','qml.RY(param4[3], wires=14)','qml.CNOT(wires=[13, 14])','qml.RY(param4[4], wires=14)','qml.RY(param4[5], wires=0)',
             'qml.RY(param4[0], wires=14)','qml.RY(param4[1], wires=15)','qml.CNOT(wires=[14, 15])','qml.RY(param4[2], wires=14)','qml.RY(param4[3], wires=15)','qml.CNOT(wires=[14, 15])','qml.RY(param4[4], wires=15)','qml.RY(param4[5], wires=0)',
             'qml.RY(param4[0], wires=15)','qml.RY(param4[1], wires=16)','qml.CNOT(wires=[15, 0])','qml.RY(param4[2], wires=15)','qml.RY(param4[3], wires=16)','qml.CNOT(wires=[15, 16])','qml.RY(param4[4], wires=16)','qml.RY(param4[5], wires=0)',
        
             ]
    non_drop = [
                'qml.CNOT(wires= [1,2])','qml.CNOT(wires= [2,1])','qml.CNOT(wires=[1,2])',
                'qml.CNOT(wires= [2,3])','qml.CNOT(wires= [3,2])','qml.CNOT(wires=[2,3])',
                'qml.CNOT(wires= [3,4])','qml.CNOT(wires= [4,3])','qml.CNOT(wires=[3,4])',
                'qml.CNOT(wires= [4,5])','qml.CNOT(wires= [5,4])','qml.CNOT(wires=[4,5])',
                'qml.CNOT(wires= [5,6])','qml.CNOT(wires= [6,5])','qml.CNOT(wires=[5,6])',
                'qml.CNOT(wires= [6,7])','qml.CNOT(wires= [7,6])','qml.CNOT(wires=[6,7])',
                'qml.CNOT(wires= [7,8])','qml.CNOT(wires= [8,7])','qml.CNOT(wires=[7,8])',
                'qml.CNOT(wires= [8,9])','qml.CNOT(wires= [9,8])','qml.CNOT(wires=[8,9])',
                'qml.CNOT(wires= [9,10])','qml.CNOT(wires= [10,9])','qml.CNOT(wires=[9,10])',
                'qml.CNOT(wires= [10,11])','qml.CNOT(wires= [11,10])','qml.CNOT(wires=[10,11])',
                'qml.CNOT(wires= [11,12])','qml.CNOT(wires= [12,11])','qml.CNOT(wires=[11,12])',
                'qml.CNOT(wires= [12,13])','qml.CNOT(wires= [13,12])','qml.CNOT(wires=[12,13])',
                'qml.CNOT(wires= [13,14])','qml.CNOT(wires= [14,13])','qml.CNOT(wires=[13,14])',
                'qml.CNOT(wires= [14,15])','qml.CNOT(wires= [15,14])','qml.CNOT(wires=[14,15])',
                'qml.CNOT(wires= [15,16])','qml.CNOT(wires= [16,15])','qml.CNOT(wires=[15,16])',

                'qml.CNOT(wires=[0, 1])','qml.CNOT(wires=[0, 1])',
                'qml.CNOT(wires=[1, 2])','qml.CNOT(wires=[1, 2])',
                'qml.CNOT(wires=[2, 3])','qml.CNOT(wires=[2, 3)',
                'qml.CNOT(wires=[3, 4])','qml.CNOT(wires=[3, 4])',
                'qml.CNOT(wires=[4, 5])','qml.CNOT(wires=[4, 5])',
                'qml.CNOT(wires=[5, 6])','qml.CNOT(wires=[5, 6])',
                'qml.CNOT(wires=[6, 7])','qml.CNOT(wires=[6, 7])',
                'qml.CNOT(wires=[7, 8])','qml.CNOT(wires=[7, 8])',
                'qml.CNOT(wires=[8, 9])','qml.CNOT(wires=[8, 9])',
                'qml.CNOT(wires=[9, 10])','qml.CNOT(wires=[9, 10])',
                'qml.CNOT(wires=[10, 11])','qml.CNOT(wires=[10, 11])',
                'qml.CNOT(wires=[11, 12])','qml.CNOT(wires=[11, 12])',
                'qml.CNOT(wires=[12, 13])','qml.CNOT(wires=[12, 13])',
                'qml.CNOT(wires=[13, 14])','qml.CNOT(wires=[13, 14])',
                'qml.CNOT(wires=[14, 15])','qml.CNOT(wires=[14, 15])',
                ]
    print(len(conv1))
    drop_conv1 = gates_percent_drop(conv1, prcnt, non_drop)
    print(len(drop_conv1))
    #Convolutional Layer 2
    conv2 = ['qml.U3(param2[0], param2[1], param2[2], wires=2)','qml.U3(param2[3], param2[4], param2[5], wires=3)','qml.CNOT(wires=[2, 3])','qml.RY(param2[6], wires=2)','qml.RZ(param2[7], wires=3)','qml.CNOT(wires= [3,2])','qml.RY(param2[8], wires=2)','qml.CNOT(wires=[2, 3])','qml.U3(param2[9], param2[10], param2[11], wires=2)','qml.U3(param2[12], param2[13], param2[14], wires=3)',
             'qml.U3(param2[0], param2[1], param2[2], wires=4)','qml.U3(param2[3], param2[4], param2[5], wires=5)','qml.CNOT(wires=[4, 5])','qml.RY(param2[6], wires=4)','qml.RZ(param2[7], wires=5)','qml.CNOT(wires= [5,4])','qml.RY(param2[8], wires=4)','qml.CNOT(wires=[4, 5])','qml.U3(param2[9], param2[10], param2[11], wires=4)','qml.U3(param2[12], param2[13], param2[14], wires=5)',
             'qml.RY(param4[0], wires=2)','qml.RY(param4[1], wires=3)','qml.CNOT(wires=[2, 3])','qml.RY(param4[2], wires=2)','qml.RY(param4[3], wires=3)','qml.CNOT(wires=[2, 3])','qml.RY(param4[4], wires=2)','qml.RY(param4[5], wires=3)',
             'qml.RY(param4[0], wires=4)','qml.RY(param4[1], wires=5)','qml.CNOT(wires=[4, 5])','qml.RY(param4[2], wires=4)','qml.RY(param4[3], wires=5)','qml.CNOT(wires=[4, 5])','qml.RY(params[4], wires=4)','qml.RY(param4[5], wires=5)',
             ]
    non_drop = ['qml.CNOT(wires=[2, 3])','qml.CNOT(wires=[2, 3)',
                'qml.CNOT(wires=[4, 5])','qml.CNOT(wires=[4, 5])',]
    
    drop_conv2 = gates_percent_drop(conv1, prcnt, non_drop)
    
   
    #Convolutional Layer 3
    conv3 = ['qml.U3(param3[0], param3[1], param3[2], wires=3)','qml.U3(param3[3], param3[4], param3[5], wires=4)','qml.CNOT(wires= [3, 4])','qml.RY(param3[6], wires=3)','qml.RZ(param3[7], wires=4)','qml.CNOT(wires= [4,3])','qml.RY(param3[8], wires=3)','qml.CNOT(wires=[3,4])','qml.U3(param3[9], param3[10], param3[11], wires=3)','qml.U3(param3[12], param3[13], param3[14], wires=4)'
             ]
    non_drop = ['qml.CNOT(wires= [2,3])','qml.CNOT(wires= [3,2])','qml.CNOT(wires=[2,3])','qml.CNOT(wires=[3,4])','qml.CNOT(wires= [4,3])'
                ]
    
    drop_conv3 = gates_percent_drop(conv3, 0, non_drop)
    
    return drop_conv1,drop_conv2,drop_conv3
    
dev = qml.device('default.qubit', wires = 16)   #change qubits here
@qml.qnode(dev)
def circuit_eval(x, params, U_params, conv1, conv2, conv3, embedding_type, cost_fn):
    embedding.data_embedding(x, embedding_type=embedding_type)
    
    
    param1 = params[0:U_params]
    param2 = params[U_params: 2 * U_params]
    param3 = params[2 * U_params: 3 * U_params]
    param4 = params[U_params * 3 : U_params * 3 + 6]
    param5 = params[U_params * 3 + 6 : U_params * 3 + 12]
    param6 = params[U_params * 3 + 12 : U_params * 3 + 18]
    
    for i in conv1:
        eval(i)
    for i in conv2:
        eval(i)
    for i in conv3:
        eval(i)
    if cost_fn == 'mse':
        result = qml.expval(qml.PauliZ(4))
    elif cost_fn == 'cross_entropy':
        result = qml.probs(wires=4)
    return result

def MC_QCNN(X, params, U, U_params, embedding_type='Amplitude', cost_fn='mse', prcnt = 0.05):
    results = []

    conv1,conv2,conv3 = QCNN_1D_circuit_U_SU4_1D(params, U_params, prcnt)
   
    for x in X:
        
        result = circuit_eval(x, params, U_params, conv1, conv2, conv3, embedding_type, cost_fn)
        # Data Embedding
        results.append(result)
    return results

