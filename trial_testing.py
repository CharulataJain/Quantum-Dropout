import tensorflow as tf
import numpy as np
'''import os
import cv2
from tqdm import tqdm
from sklearn.model_selection import train_test_split


classes = {'AbdomenCT': 0, 'BreastMRI': 1, 'ChestCT': 2, 'CXR': 3, 'Hand': 4, 'HeadCT': 5}
IMAGE_WIDTH=64
IMAGE_HEIGHT=64
IMAGE_SIZE=(IMAGE_WIDTH, IMAGE_HEIGHT)
abdomen_DIR='C:/Users/maxaa/Monte-Carlo QML/MNIST_QCNN/DATA/AbdomenCT'
breastmri_DIR='C:/Users/maxaa/Monte-Carlo QML/MNIST_QCNN/DATA/BreastMRI'
chestct_DIR='C:/Users/maxaa/Monte-Carlo QML/MNIST_QCNN/DATA/ChestCT'
cxr_DIR='C:/Users/maxaa/Monte-Carlo QML/MNIST_QCNN/DATA/CXR'
hand_DIR='C:/Users/maxaa/Monte-Carlo QML/MNIST_QCNN/DATA/Hand'
headct = 'C:/Users/maxaa/Monte-Carlo QML/MNIST_QCNN/DATA/HeadCT'


def assign_label(img,class_type):
    return class_type

X = []
y = []

def make_train_data(class_type,DIR):
    for img in tqdm(os.listdir(DIR)):
        label=assign_label(img,class_type)
        path = os.path.join(DIR,img)
        img = cv2.imread(path,0)
        img = cv2.resize(img, (IMAGE_WIDTH,IMAGE_HEIGHT))
        
        X.append(np.array(img))
        y.append(str(label))

make_train_data(classes.get('AbdomenCT'), abdomen_DIR)
make_train_data(classes.get('BreastMRI'), breastmri_DIR)
make_train_data(classes.get('ChestCT'), chestct_DIR)
make_train_data(classes.get('CXR'), cxr_DIR)
make_train_data(classes.get('Hand'), hand_DIR)
make_train_data(classes.get('HeadCT'), headct)        


y = np.array(y)
X = np.array(X)
#X_train,X_test, y_train, y_test = train_test_split(X, y, shuffle=True, test_size=0.15, random_state=58954)
X_train, X_test, y_train, y_test = train_test_split(X, y, shuffle=True, test_size=0.25, random_state=58954)
X_test, X_val, y_test, y_val = train_test_split(X_test, y_test, shuffle=True, test_size=0.2, random_state=58954)
print(X_train.shape,X_test.shape,X_val.shape)
X_train, X_test = X_train/255, X_test/255
x_train_filter_01 = np.where((y_train == '0') | (y_train == '1'))
x_test_filter_01 = np.where((y_test == '0') | (y_test == '1'))

X_train, X_test = X_train[x_train_filter_01], X_test[x_test_filter_01]
Y_train, Y_test = y_train[x_train_filter_01], y_test[x_test_filter_01]
binary = False
if binary == False:
    Y_train = [1 if y == '0' else 0 for y in Y_train]
    Y_test = [1 if y == '0' else 0 for y in Y_test]
elif binary == True:
    Y_train = [1 if y == '0' else -1 for y in Y_train]
    Y_test = [1 if y == '0' else -1 for y in Y_test]
print(X_train.shape)


#fit for pca

x_train = np.concatenate((x_train, x_test))
y_train = np.concatenate((y_train, y_test))
'''

import random

def delete_percent_of_list(lst, percent, exceptions):
    """
    Deletes a certain percentage of elements from a list at random, except for specific items
    specified in the 'exceptions' list, while preserving the original sequence of the list.
    
    Args:
        lst (list): The original list.
        percent (float): The percentage of elements to be deleted, specified as a float between 0 and 100.
        exceptions (list): The list of items to be excluded from deletion.
    
    Returns:
        list: The modified list with the specified percentage of elements deleted.
    """
    num_elements_to_delete = int(len(lst) * (percent / 100))
    indices_to_delete = random.sample(range(len(lst)), num_elements_to_delete)
    indices_to_delete.sort(reverse=True) # Sort in reverse order to preserve sequence during deletion
    
    for index in indices_to_delete:
        if lst[index] not in exceptions:
            del lst[index]
    
    return lst
my_list = ['qml.U3(param1[0], param1[1], param1[2], wires=0)','qml.U3(param1[3], param1[4], param1[5], wires=1)','qml.CNOT(wires= [0,1])','qml.RY(param1[6], wires=0)','qml.RZ(param1[7], wires=1)','qml.CNOT(wires= [1,0])','qml.RY(param1[8], wires=0)','qml.CNOT(wires=[0,1])','qml.U3(param1[9], param1[10], param1[11], wires=0)','qml.U3(param1[12], param1[13], param1[14], wires=1)',
         'qml.U3(param1[0], param1[1], param1[2], wires=1)','qml.U3(param1[3], param1[4], param1[5], wires=2)','qml.CNOT(wires= [1,2])','qml.RY(param1[6], wires=1)','qml.RZ(param1[7], wires=2)','qml.CNOT(wires= [2,1])','qml.RY(param1[8], wires=1)','qml.CNOT(wires=[1,2])','qml.U3(param1[9], param1[10], param1[11], wires=1)','qml.U3(param1[12], param1[13], param1[14], wires=2)',
         'qml.U3(param1[0], param1[1], param1[2], wires=2)','qml.U3(param1[3], param1[4], param1[5], wires=3)','qml.CNOT(wires= [2,3])','qml.RY(param1[6], wires=2)','qml.RZ(param1[7], wires=3)','qml.CNOT(wires= [3,2])','qml.RY(param1[8], wires=2)','qml.CNOT(wires=[2,3])','qml.U3(param1[9], param1[10], param1[11], wires=2)','qml.U3(param1[12], param1[13], param1[14], wires=3)',
         'qml.U3(param1[0], param1[1], param1[2], wires=3)','qml.U3(param1[3], param1[4], param1[5], wires=4)','qml.CNOT(wires= [3,4])','qml.RY(param1[6], wires=3)','qml.RZ(param1[7], wires=4)','qml.CNOT(wires= [4,3])','qml.RY(param1[8], wires=3)','qml.CNOT(wires=[3,4])','qml.U3(param1[9], param1[10], param1[11], wires=3)','qml.U3(param1[12], param1[13], param1[14], wires=4)',
         'qml.U3(param1[0], param1[1], param1[2], wires=4)','qml.U3(param1[3], param1[4], param1[5], wires=5)','qml.CNOT(wires= [4,5])','qml.RY(param1[6], wires=4)','qml.RZ(param1[7], wires=5)','qml.CNOT(wires= [5,4])','qml.RY(param1[8], wires=4)','qml.CNOT(wires=[4,5])','qml.U3(param1[9], param1[10], param1[11], wires=4)','qml.U3(param1[12], param1[13], param1[14], wires=5)',
         'qml.U3(param1[0], param1[1], param1[2], wires=5)','qml.U3(param1[3], param1[4], param1[5], wires=6)','qml.CNOT(wires= [5,6])','qml.RY(param1[6], wires=5)','qml.RZ(param1[7], wires=6)','qml.CNOT(wires= [6,5])','qml.RY(param1[8], wires=5)','qml.CNOT(wires=[5,6])','qml.U3(param1[9], param1[10], param1[11], wires=5)','qml.U3(param1[12], param1[13], param1[14], wires=6)',
         'qml.U3(param1[0], param1[1], param1[2], wires=6)','qml.U3(param1[3], param1[4], param1[5], wires=7)','qml.CNOT(wires= [6,7])','qml.RY(param1[6], wires=6)','qml.RZ(param1[7], wires=7)','qml.CNOT(wires= [7,6])','qml.RY(param1[8], wires=6)','qml.CNOT(wires=[6,7])','qml.U3(param1[9], param1[10], param1[11], wires=6)','qml.U3(param1[12], param1[13], param1[14], wires=7)',
         'qml.U3(param1[0], param1[1], param1[2], wires=7)','qml.U3(param1[3], param1[4], param1[5], wires=0)','qml.CNOT(wires= [7,0])','qml.RY(param1[6], wires=7)','qml.RZ(param1[7], wires=8)','qml.CNOT(wires= [0,7])','qml.RY(param1[8], wires=7)','qml.CNOT(wires=[7,8])','qml.U3(param1[9], param1[10], param1[11], wires=7)','qml.U3(param1[12], param1[13], param1[14], wires=0)'
         ]
print(len(my_list))
percent_to_delete = 10
exceptions = ['qml.CNOT(wires= [0,1])','qml.CNOT(wires= [1,0])','qml.CNOT(wires=[0,1])',
            'qml.CNOT(wires= [1,2])','qml.CNOT(wires= [2,1])','qml.CNOT(wires=[1,2])',
            'qml.CNOT(wires= [2,3])','qml.CNOT(wires= [3,2])','qml.CNOT(wires=[2,3])',
            'qml.CNOT(wires= [3,4])','qml.CNOT(wires= [4,3])','qml.CNOT(wires=[3,4])',
            'qml.CNOT(wires= [4,5])','qml.CNOT(wires= [5,4])','qml.CNOT(wires=[4,5])',
            'qml.CNOT(wires= [5,6])','qml.CNOT(wires= [6,5])','qml.CNOT(wires=[5,6])',
            'qml.CNOT(wires= [6,7])','qml.CNOT(wires= [7,6])','qml.CNOT(wires=[6,7])',
            'qml.CNOT(wires= [7,0])''qml.CNOT(wires= [0,7])','qml.CNOT(wires=[7,8])']
print(len(exceptions))
new_list = delete_percent_of_list(my_list, percent_to_delete, exceptions)
print(len(new_list))

for i in range(0, 8, 2):
    print(i,i+1)
for i in range(1, 7, 2):
    print(i,i+1)