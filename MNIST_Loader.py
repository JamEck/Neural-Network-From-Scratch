import numpy as np

def loadMNISTimages(fileName):
    with open(fileName, 'br') as file:
    
        file.seek(4)       # seek ahead to skip magic number
        buf = file.read(4) # read next 4 bytes into a buffer

        # convert those bytes into an integer
        # imgNum = number of images in file
        imgNum  = int.from_bytes(buf,"big") # interpret bytes as big-endian (MSB first)

        buf = file.read(4)
        imgHeight = int.from_bytes(buf,"big")

        buf = file.read(4)
        imgWidth  = int.from_bytes(buf,"big")

        imgs = [[]]*imgNum # list to hold all images

        # all images in the MNIST Data Set are 28x28 grayscale images
        # and are stored consecutively after the file header

        for i in range(imgNum):
            buf = file.read(imgWidth*imgHeight) # read in bytes for i'th image
            imgs[i] = np.asarray(list(buf)).reshape(28,28) # convert to 28x28 np array
    
    return np.array(imgs)


def loadMNISTlabels(fileName):
    with open(fileName, 'br') as file:

        file.seek(4) # skip magic number

        buf = file.read(4)
        labNum = int.from_bytes(buf,"big") # number of labels

        labels = np.zeros(labNum, dtype = np.uint8) # preallocate array to hold labels
        buf = file.read(labNum) # read in labels

        # convert each label from a byte to an integer and pass each to array
        for i, byte in enumerate(buf):
            labels[i] = int(byte)

        file.close()

    return labels

