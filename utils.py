import numpy as np
import tensorflow as tf
import tensorflow.keras.datasets as data
import matplotlib.pyplot as plt
from tensorflow.keras import Model, Input
from tensorflow.keras.optimizers import Adam, SGD
from tensorflow.keras.layers import Conv2D, MaxPool2D, Dropout, Dense, Flatten, BatchNormalization
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split


def make_cnn():
    inputs = Input(shape=(32, 32, 1))
    x = inputs

    x = Conv2D(16, 3, padding='same', activation='relu')(x)
    # x = BatchNormalization()(x)
    x = Conv2D(16, 3, padding='same', activation='relu')(x)
    # x = BatchNormalization()(x)
    x = MaxPool2D(pool_size=(2, 2))(x)
    # x = Dropout(0.2)(x)

    x = Conv2D(32, 3, padding='same', activation='relu')(x)
    # x = BatchNormalization()(x)
    x = Conv2D(32, 3, padding='same', activation='relu')(x)
    # x = BatchNormalization()(x)
    x = MaxPool2D(pool_size=(2, 2))(x)
    # x = Dropout(0.2)(x)

    x = Conv2D(64, 3, padding='same', activation='relu')(x)
    # x = BatchNormalization()(x)
    x = Conv2D(64, 3, padding='same', activation='relu')(x)
    # x = BatchNormalization()(x)
    x = MaxPool2D(pool_size=(2, 2))(x)
    # x = Dropout(0.2)(x)

    x = Flatten()(x)
    x = Dense(512, activation='relu')(x)
    x = Dropout(0.2)(x)
    x = Dense(96, activation='relu')(x)
    x = Dropout(0.2)(x)
    x = Dense(10, activation='softmax')(x)

    outputs = x
    cnn = Model(inputs=inputs, outputs=outputs)

    return cnn

def augment_data(X, y):

    # change 0's to empty cells
    for idx in range(len(y)):
        if y[idx] == 0:
            X[idx] = tf.zeros_like(X[idx])

    padding = tf.constant([[0, 0], [2, 2], [2, 2]]) # pad some images with border
    border = tf.pad(X, padding, constant_values=255) # white border
    no_border = tf.pad(X, padding, mode='SYMMETRIC') # no border
    X = tf.concat([border, no_border], 0)
    y = tf.concat([y, y], 0)

    X1, X2, y1, y2 = train_test_split(X.numpy(), y.numpy(), test_size=0.1, random_state=42)
    X = np.concatenate((X1, X2), axis=0)
    y = np.concatenate((y1, y2), axis=0)

    return X, y


def find_cell(board):
    n_r, n_c = board.shape
    for i in range(n_r):
        for j in range(n_c):
            if board[i][j] == 0:
                return (i, j)

    return None

def isvalid(board, num, pos):
    row, col = pos
    n_r, n_c = board.shape
    for i in range(n_c):
        if board[row][i] == num and col != i:
            return False

    for j in range(n_r):
        if board[j][col] == num and row != j:
            return False

    box_x = col // 3
    box_y = row // 3

    for i in range(box_y * 3, box_y * 3 + 3):
        for j in range(box_x * 3, box_x * 3 + 3):
            if board[i][j] == num and (i, j) != pos:
                return False

    return True

def solve_sudoku(board):
    cell = find_cell(board)
    if cell is None:
        return board, True

    for i in range(1, 10):
        if isvalid(board, i, cell):
            board[cell] = i

            b, solved = solve_sudoku(board)
            if solved:
                return b, solved

            board[cell] = 0

    return board, False


def print_board(board):
    for i in range(len(board)):
        if i % 3 == 0 and i != 0:
            print("- - - - - - - - - - - - - ")

        for j in range(len(board[0])):
            if j % 3 == 0 and j != 0:
                print(" | ", end="")

            if j == 8:
                print(board[i][j])
            else:
                print(str(board[i][j]) + " ", end="")
