import sys 
import cv2
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from board import *

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


def main():
    if len(sys.argv) != 2:
        print('Invalid arguments')
        exit(1)

    path = sys.argv[1]
    b = Board(path)
    b.find_board()
    # plt.imshow(b.img)
    # plt.show()
    b.predict(model_path='cnn/')


if __name__ == '__main__':
    main()

