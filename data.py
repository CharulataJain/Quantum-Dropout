import numpy as np
import tensorflow as tf
from sklearn.decomposition import PCA
from tensorflow.keras.models import Model
from tensorflow.keras import layers, losses
import nibabel as nib
from glob import glob
import os
# import cv2
from tqdm import tqdm
from sklearn.model_selection import train_test_split
# imports
from skimage.transform import resize

pca32 = ['pca32-1', 'pca32-2', 'pca32-3', 'pca32-4']
autoencoder32 = ['autoencoder32-1', 'autoencoder32-2', 'autoencoder32-3', 'autoencoder32-4']
pca30 = ['pca30-1', 'pca30-2', 'pca30-3', 'pca30-4']
autoencoder30 = ['autoencoder30-1', 'autoencoder30-2', 'autoencoder30-3', 'autoencoder30-4']
pca16 = ['pca16-1', 'pca16-2', 'pca16-3', 'pca16-4', 'pca16-compact']
autoencoder16 = ['autoencoder16-1', 'autoencoder16-2', 'autoencoder16-3', 'autoencoder16-4', 'autoencoder16-compact']
pca12 = ['pca12-1', 'pca12-2', 'pca12-3', 'pca12-4']
autoencoder12 = ['autoencoder12-1', 'autoencoder12-2', 'autoencoder12-3', 'autoencoder12-4']

def data_load_and_process(dataset, classes=['0', '1'], feature_reduction='resize256', binary=True):

    # data_classes = {'AbdomenCT': 0, 'BreastMRI': 1, 'ChestCT': 2, 'CXR': 3, 'Hand': 4, 'HeadCT': 5}
    # IMAGE_WIDTH=64
    # IMAGE_HEIGHT=64
    # abdomen_DIR='C:/Users/maxaa/Monte-Carlo QML/MNIST_QCNN/DATA/AbdomenCT'
    # breastmri_DIR='C:/Users/maxaa/Monte-Carlo QML/MNIST_QCNN/DATA/BreastMRI'
    # chestct_DIR='C:/Users/maxaa/Monte-Carlo QML/MNIST_QCNN/DATA/ChestCT'
    # cxr_DIR='C:/Users/maxaa/Monte-Carlo QML/MNIST_QCNN/DATA/CXR'
    # hand_DIR='C:/Users/maxaa/Monte-Carlo QML/MNIST_QCNN/DATA/Hand'
    # headct = 'C:/Users/maxaa/Monte-Carlo QML/MNIST_QCNN/DATA/HeadCT'

    data_classes = {"HGG": 1, "LGG": 0}
    IMAGE_WIDTH=180
    IMAGE_HEIGHT=180
    abdomen_DIR=r"C:\Users\charu\Desktop\Projects\MNIST_QCNN\MNIST_QCNN\MICCAI_BraTS_2019_Data_Training\HGG"
    breastmri_DIR=r"C:\Users\charu\Desktop\Projects\MNIST_QCNN\MNIST_QCNN\MICCAI_BraTS_2019_Data_Training\LGG"

    def assign_label(img,class_type):
        return class_type
    # data lists
    X = []
    y = []

    # def make_train_data(class_type,DIR):
    #     label = class_type
    #     # we need to go into the directories, access flair of each patient, and read it slice by slice
    #     for patient in sorted(glob(os.path.join(DIR, "*"))):
    #         # patient = C:\Users\charu\Desktop\Projects\MNIST_QCNN\MNIST_QCNN\MICCAI_BraTS_2019_Data_Training\HGG\BraTS19_2013_3_1
    #         flair_path = patient + os.path.basename(patient) + "_flair.nii.gz"
    #         imgVol = nib.load(flair_path).get_fdata()
    #         for img in imgVol:
    #             X.append(np.array(img))
    #             y.append(str(label))

        # for img in tqdm(os.listdir(DIR)):
        #     label=assign_label(img,class_type)
        #     path = os.path.join(DIR,img)

            # img = cv2.imread(path,0)
            # img = cv2.resize(img, (IMAGE_WIDTH,IMAGE_HEIGHT))
    percent = 0.6 
    hgg_list = sorted(glob(os.path.join(abdomen_DIR, "*")))
    hgg_list = hgg_list[:int(percent*len(hgg_list))]
    lgg_list = sorted(glob(os.path.join(breastmri_DIR, "*")))
    lgg_list = lgg_list[:int(percent*len(lgg_list))]

    print(len(hgg_list), len(lgg_list))

    # X_train = np.empty((240, 240, 0))
    X_train = []
    y_train = []
    # X_test = np.empty((240,240,0))
    X_test = []
    y_test = []
    def make_train_data(class_type,DIR, num_samples, split , X, y):
        # print(num_samples)
        label = class_type
        if split == "train":
            patient_list = DIR[:num_samples]
        elif split == "test":
            patient_list = DIR[num_samples:]
        # we need to go into the directories, access flair of each patient, and read it slice by slice
        for patient in tqdm(patient_list):
            # patient = C:\Users\charu\Desktop\Projects\MNIST_QCNN\MNIST_QCNN\MICCAI_BraTS_2019_Data_Training\HGG\BraTS19_2013_3_1
            flair_path = patient + "\\" + os.path.basename(patient) + "_flair.nii.gz"
            imgVol = nib.load(flair_path).get_fdata()
            # print(imgVol.shape)

            for idx in range(imgVol.shape[-1]):
                img = np.array(imgVol[:,:,idx])
                img = resize(img, (IMAGE_WIDTH, IMAGE_HEIGHT))
                X.append(img)
                y.append(str(label))
        # print(np.array(X).shape, np.array(y).shape)
        return X, y

        # for img in tqdm(os.listdir(DIR)):
        #     label=assign_label(img,class_type)
        #     path = os.path.join(DIR,img)

            # img = cv2.imread(path,0)
            # img = cv2.resize(img, (IMAGE_WIDTH,IMAGE_HEIGHT))
            
    # calculating the number of patients needed for training (80%)
    len_HGG = len(hgg_list)
    num_train_HGG = int(0.8 * len_HGG)
    # num_test_HGG = len_HGG - num_train_HGG
    # print(num_train_HGG, num_test_HGG)
    len_LGG = len(lgg_list)
    num_train_LGG = int(0.8 * len_LGG)
    # num_test_LGG = len_LGG - num_train_LGG
    X_train,y_train = make_train_data(data_classes.get('HGG'), hgg_list, num_train_HGG, "train",X_train, y_train)
    X_train, y_train = make_train_data(data_classes.get('LGG'), lgg_list, num_train_LGG, "train",  X_train, y_train)
    X_test, y_test = make_train_data(data_classes.get('HGG'), hgg_list, num_train_HGG,"test", X_test, y_test)
    X_test, y_test = make_train_data(data_classes.get('LGG'), lgg_list, num_train_LGG,"test", X_test, y_test)
    X_train = np.array(X_train)
    # np.moveaxis(X_train,0,-1)
    y_train = np.array(y_train)
    X_test = np.array(X_test)
    # np.moveaxis(X_test, 0, -1)
    y_test = np.array(y_test)
                    
            
    # make_train_data(data_classes.get('HGG'), abdomen_DIR)
    # make_train_data(data_classes.get('LGG'), breastmri_DIR)
    # make_train_data(data_classes.get('AbdomenCT'), abdomen_DIR)
    # make_train_data(data_classes.get('BreastMRI'), breastmri_DIR)
    # make_train_data(data_classes.get('ChestCT'), chestct_DIR)
    # make_train_data(data_classes.get('CXR'), cxr_DIR)
    # make_train_data(data_classes.get('Hand'), hand_DIR)
    # make_train_data(data_classes.get('HeadCT'), headct)        


    # y = np.array(y)
    # X = np.array(X)
    # X_train, X_test, y_train, y_test = train_test_split(X, y, shuffle=True, test_size=0.20, random_state=58954)
    X_test, X_val, y_test, Y_val = train_test_split(X_test, y_test, shuffle=True, test_size=0.5, random_state=58954)
    X_train = np.concatenate((X_train, X_test))
    Y_train = np.concatenate((y_train, y_test))
    X_train, X_test, X_val = X_train[..., np.newaxis] / 255.0, X_test[..., np.newaxis] / 255.0, X_val[..., np.newaxis] / 255.0
    x_train_filter_01 = np.where((y_train == classes[0]) | (y_train == classes[1]))
    x_test_filter_01 = np.where((y_test == classes[0]) | (y_test == classes[1]))
    x_val_filter_01 = np.where((Y_val == classes[0]) | (Y_val == classes[1]))

    X_train, X_test, X_val = X_train[x_train_filter_01], X_test[x_test_filter_01], X_val[x_val_filter_01]
    Y_train, Y_test, Y_val = y_train[x_train_filter_01], y_test[x_test_filter_01], Y_val[x_val_filter_01]
    
    if binary == False:
        Y_train = [1 if y == classes[0] else 0 for y in Y_train]
        Y_test = [1 if y == classes[0] else 0 for y in Y_test]
        Y_val = [1 if y == classes[0] else 0 for y in Y_val]
    elif binary == True:
        Y_train = [1 if y == classes[0] else -1 for y in Y_train]
        Y_test = [1 if y == classes[0] else -1 for y in Y_test]
        Y_val = [1 if y == classes[0] else 0 for y in Y_val]
    
    #end    
    if feature_reduction == 'resize256':
        X_train = tf.image.resize(X_train[:], (256, 1)).numpy()
        X_test = tf.image.resize(X_test[:], (256, 1)).numpy()
        X_val = tf.image.resize(X_val[:], (256, 1)).numpy()
        X_train, X_test, X_val = tf.squeeze(X_train).numpy(), tf.squeeze(X_test).numpy(), tf.squeeze(X_val).numpy()
        return X_train, X_test, Y_train, Y_test, X_val, Y_val

    elif feature_reduction == 'pca8' or feature_reduction in pca32 \
            or feature_reduction in pca30 or feature_reduction in pca16 or feature_reduction in pca12:

        X_train = tf.image.resize(X_train[:], (46*46, 1)).numpy() #(784, 1)
        X_test = tf.image.resize(X_test[:], (46*46, 1)).numpy() #(784, 1)
        X_val = tf.image.resize(X_val[:], (46*46, 1)).numpy() #(784, 1)
        X_train, X_test, X_val = tf.squeeze(X_train), tf.squeeze(X_test), tf.squeeze(X_val)

        if feature_reduction == 'pca8':
            pca = PCA(8)
        elif feature_reduction in pca32:
            pca = PCA(32)
        elif feature_reduction in pca30:
            pca = PCA(30)
        elif feature_reduction in pca16:
            pca = PCA(16)
        elif feature_reduction in pca12:
            pca = PCA(12)


        X_train = pca.fit_transform(X_train)
        X_test = pca.transform(X_test)
        X_val = pca.transform(X_val)

        # Rescale for angle embedding
        if feature_reduction == 'pca8' or feature_reduction == 'pca16-compact' or feature_reduction in pca30 or feature_reduction in pca12:
            X_train, X_test, X_val = (X_train - X_train.min()) * (np.pi / (X_train.max() - X_train.min())), (X_test - X_test.min()) * (np.pi / (X_test.max() - X_test.min())),(X_val - X_val.min()) * (np.pi / (X_val.max() - X_val.min()))
        return X_train, X_test, Y_train, Y_test, X_val, Y_val

    elif feature_reduction == 'autoencoder8' or feature_reduction in autoencoder32 \
            or feature_reduction in autoencoder30 or feature_reduction in autoencoder16 or feature_reduction in autoencoder12:
        if feature_reduction == 'autoencoder8':
            latent_dim = 8
        elif feature_reduction in autoencoder32:
            latent_dim = 32
        elif feature_reduction in autoencoder30:
            latent_dim = 30
        elif feature_reduction in autoencoder16:
            latent_dim = 16
        elif feature_reduction in autoencoder12:
            latent_dim = 12



        class Autoencoder(Model):
            def __init__(self, latent_dim):
                super(Autoencoder, self).__init__()
                self.latent_dim = latent_dim
                self.encoder = tf.keras.Sequential([
                    layers.Flatten(),
                    layers.Dense(latent_dim, activation='relu'),
                ])
                self.decoder = tf.keras.Sequential([
                    layers.Dense(46*46, activation='sigmoid'),
                    layers.Reshape((46, 46))
                ])

            def call(self, x):
                encoded = self.encoder(x)
                decoded = self.decoder(encoded)
                return decoded

        autoencoder = Autoencoder(latent_dim)

        autoencoder.compile(optimizer='adam', loss=losses.MeanSquaredError())
        autoencoder.fit(X_train, X_train, X_val,
                        epochs=10,
                        shuffle=True,
                        validation_data=(X_test, X_test, X_val))

        X_train, X_test, X_val = autoencoder.encoder(X_train).numpy(), autoencoder.encoder(X_test).numpy(), autoencoder.encoder(X_val).numpy()

        # Rescale for Angle Embedding
        # Note this is not a rigorous rescaling method
        if feature_reduction == 'autoencoder8' or feature_reduction == 'autoencoder16-compact' or feature_reduction in autoencoder30 or feature_reduction in autoencoder12:
            X_train, X_test, X_val = (X_train - X_train.min()) * (np.pi / (X_train.max() - X_train.min())), (X_test - X_test.min()) * (np.pi / (X_test.max() - X_test.min())),(X_val - X_val.min()) * (np.pi / (X_val.max() - X_val.min()))
        return X_train, X_test, Y_train, Y_test, X_val, Y_val
