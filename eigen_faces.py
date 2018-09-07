import os
import numpy as np
import sys

from matplotlib import pyplot as plt
from scipy.spatial.distance import euclidean


class FaceRecognitionSystem:
    def __init__(self, CONFIG):
        """
            The init section of this calss is used to make sure that the config passed to the 
            class, contains all the necessary key value pairs and also that the values are vaild.

            In case, any of the directories are invaild, default values are assigned.
            By default, these are the directories:
            training images folder = './train/'
            testing images folder  = './test/'
            output folder          = './output/'

            In case the default directory is missing, then the program just informs the user that
            it cannot find the default directory and exits. 
        """
        self.CONFIG = CONFIG

        if 'T0' not in self.CONFIG:
            print('Threshold T0 is not specified. Exiting program now')
            input('Press "Enter" key to quit')
            sys.exit()

        if 'T1' not in self.CONFIG:
            print('Threshold T1 is not specified. Exiting program now')
            input('Press "Enter" key to quit')
            sys.exit()

        if not os.path.exists(self.CONFIG['train_dir']):
            print("Invaild directory of training images specified.\nReverting to default training directory './train/'")
            self.CONFIG['train_dir'] = './train/'
            
            if not os.path.exists('./train/'):
                print("default training directory does not exists. Exiting program now.")
                input('Press "Enter" key to quit')
                sys.exit()
        
        if not os.path.exists(self.CONFIG['test_dir']):
            print("Invaild directory of testing images specified.\nReverting to default testing directory './test/'")
            self.CONFIG['test_dir'] = './test/'
            
            if not os.path.exists('./test/'):
                print("default testing directory does not exists. Exiting program now.")
                input('Press "Enter" key to quit')
                sys.exit()

        if not os.path.exists(self.CONFIG['output_dir']):
            print("Invaild directory of output images specified.\nReverting to default output directory './output/'")
            os.mkdir('output')
            self.CONFIG['output_dir'] = './output/'

    def train(self):
        """
        This function performs the training component of the face recognition system.
        """
        # read list of files present in the training directory
        self.training_images_names = os.listdir(self.CONFIG['train_dir'])
        
        # load all the files found in the training directory as image objects and store them
        training_images = [plt.imread(self.CONFIG['train_dir'] + each_image) for each_image in self.training_images_names]
        
        # store the number of training images and thier dimensions for later
        self.training_images_count = len(training_images)
        self.image_shape = training_images[0].shape
        self.image_size = self.image_shape[0] * self.image_shape[1]
        
        # compute the mean face. Sum of all the training images / number of training images.
        mean_face = np.sum(training_images, axis=0)
        mean_face = mean_face / self.training_images_count

        # compute matrix A where each column contains the difference between a training image and the mean face
        A = np.transpose(np.vstack([each_image.flatten() - mean_face.flatten() for each_image in training_images]))
        L = np.dot(np.transpose(A), A)

        # compute the eigen values and eigen vectors
        eigen_values, eigen_vectors = np.linalg.eig(L)
        
        self.mean_face = mean_face
        self.U = np.dot(A, eigen_vectors)   # eigen space / face space / eigenfaces

        # projection of each training image onto the eigen face
        self.omega_i = [np.dot(np.transpose(self.U), A[:, index]) for index in range(self.training_images_count)]
        
        # code necessary for saving intermediate images like the eigen faces into image files for output purposes
        plt.imsave(self.CONFIG['output_dir'] + 'mean_face.jpg', mean_face, cmap='gray')
        for index in range(self.training_images_count):
            file_name = 'eigen_face - ' + self.training_images_names[index]
            plt.imsave(self.CONFIG['output_dir'] + file_name, np.reshape(self.U[:, index], self.image_shape), cmap='gray')
    
    def recognition(self, OUTPUT=False):
        """
        This function performs the recognition component of the face recognition system.
        """
        # read the list of files present in the testing directory 
        testing_images_names = os.listdir(self.CONFIG['test_dir'])

        # load all the files found in the testing directory as image objects and store them
        testing_images = [plt.imread(self.CONFIG['test_dir'] + each_image) for each_image in testing_images_names]

        # preparing a dictionary to represent the input face and the result of the recognition process
        output = {}

        for index, each in enumerate(testing_images):
            # subtracting mean face from the current testing image
            I_vector = np.transpose(each.flatten() - self.mean_face.flatten())

            # computing projection of above result onto the face space
            omega_I = np.dot(np.transpose(self.U), I_vector)

            # reconstruct input face image from eigenfaces
            Ir_vector = np.dot(self.U, omega_I)

            # for saving intermediate images for the report
            if OUTPUT:
                plt.imsave(self.CONFIG['output_dir'] + 'I_vector ' + testing_images_names[index], np.reshape(I_vector, self.image_shape), cmap='gray')
                plt.imsave(self.CONFIG['output_dir'] + 'Ir_vector ' + testing_images_names[index], np.reshape(Ir_vector, self.image_shape), cmap='gray')

            # compute distance between input face and it's reconstruction by using PCA
            d0 = np.round(euclidean(Ir_vector, I_vector))

            # if distance is less than threshold T0, then the input is classified as non-face
            if d0 > CONFIG['T0']:
                output[testing_images_names[index]] = 'Non-face'
                continue

            # compute the distance between each of the eigen faces and the projection of the current test image
            di = np.round(np.array([euclidean(omega_I, each) for each in self.omega_i]))

            # find the minimum of all the distances computed above.
            dj = np.min(di)

            # if the minimum distance is less than threashold T1, then the input is classified as the face which
            # when projected to the eigen space results in the smallest distance with the test image's projection
            if dj < CONFIG['T1']:
                output[testing_images_names[index]] = self.training_images_names[np.argmin(di)]

            # if distance is greater than T1, then the face is not recognized
            else:
                output[testing_images_names[index]] = 'Unknown Face'

        # print the results
        print("results:")
        for each in output:
        	print("{:<27} : {}".format(each, output[each]))

"""
    A dictionary object is used to store the configs for the face recognition system.
    Thus, if any of the configs need to be changed, it can be done easily by modifying
    the dictionary below.

    For example, if you wish to change the directory of the folder containing the training
    images, you would modify the 'train_dir' key's value in the dictionary below and change
    it from './train/' to './path/to/training/images/' (example)
"""
CONFIG = {
    'train_dir' : './train/',
    'test_dir'  : './test/',
    'output_dir': './output/',
    'T0'        : 6.5e+13,
    'T1'        : 8.65e+7
}

"""
    By coding the program in an object oriented fashion, we can train and detect multiple
    sets of facial images using the same script. It can even be imported into a larger
    project.
"""

FRS = FaceRecognitionSystem(CONFIG)
FRS.train()
FRS.recognition(OUTPUT=True)