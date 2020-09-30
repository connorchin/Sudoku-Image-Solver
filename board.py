import numpy as np
import pdb
import cv2
import tensorflow.keras as keras
from scipy.ndimage import gaussian_filter
from skimage.segmentation import clear_border
from cnn import *
from utils import *


class Board:
    def __init__(self, path):
        super().__init__()
        self.path = path
        self.orig = cv2.imread(path)
        self.img = None

    def find_board(self):
        orig_grey = cv2.cvtColor(self.orig, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(orig_grey, (5, 5), 1) 
        blur = cv2.fastNlMeansDenoising(blur)

        # apply adaptive threshold (for varied lighting)
        thresh = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)

        # denoise further
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        opening = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel) # erosion -> dilation to remove noise
        sure_bg = cv2.dilate(opening, kernel, iterations=3)
        denoise = cv2.subtract(sure_bg, thresh)

        # dilate edges
        dilate = cv2.dilate(denoise, kernel)

        # find outline of board
        contours, hierarchy = cv2.findContours(dilate.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours = sorted(contours, key=cv2.contourArea, reverse=True)
        puzzle = None
        puzzle_pts = None

        # find largest contour with 4 pts (rectangle)
        for cnt in contours:
            epsilon = 0.1 * cv2.arcLength(cnt, True)
            pts = cv2.approxPolyDP(cnt, epsilon, True)
            if len(pts) == 4:
                puzzle_pts = pts
                puzzle = cnt
                break

        if puzzle is None and puzzle_pts:
            print('Sudoku puzzle not found in image')
            exit()

        # rearrange points
        puzzle_pts = puzzle_pts.squeeze(axis=1)
        board = np.zeros_like(puzzle_pts)
        pt_sum = np.sum(puzzle_pts, axis=1)
        board[0] = puzzle_pts[np.argmin(pt_sum)] # top left
        board[2] = puzzle_pts[np.argmax(pt_sum)] # bottom right

        pt_diff = np.diff(puzzle_pts, axis=1)
        board[1] = puzzle_pts[np.argmin(pt_diff)] # top right
        board[3] = puzzle_pts[np.argmax(pt_diff)] # bottom left

        # find width, height of board from points
        board = board.astype('float32')
        tr, tl, bl, br = board
        w = max(np.linalg.norm(br - bl), np.linalg.norm(tr - tl))
        h = max(np.linalg.norm(tr - br), np.linalg.norm(tl - bl))

        # get birds-eye view of board
        dst = np.array([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]], dtype='float32')
        M = cv2.getPerspectiveTransform(board, dst)
        warp = cv2.warpPerspective(thresh.copy(), M, (w, h))

        self.img = warp


    def predict(self, invert=True, model_path=None):
        if self.img is None:
            print('Call find_board before attempting to solve')
            exit()

        if model_path is None:
            clf = CNN()
            clf.train()
            model = clf.model
        else:
            model = keras.models.load_model(model_path)

        img = np.invert(self.img)
        h, w = img.shape
        cw = int(w / 9)
        ch = int(h / 9)

        x, y = 0, 0
        b = list()
        plt_idx = 1
        while y + ch < h:
            x = 0
            b.append(list())
            while x + cw < w:
                cell = img[y:y+ch,x:x+cw]
                plt.subplot(9,9,plt_idx)
                # print(cell.shape, w, h, x, y, cw, ch, plt_idx)
                # pdb.set_trace()
                cell = clear_border(cell)
                cell = self._get_digit(cell)
                plt_idx += 1

                if cell is None:
                    pred = 0
                else:
                    cell = cv2.resize(cell, (32, 32))
                    cell = cell.astype('float32') / 255.
                    cell = np.expand_dims(cell, axis=0)
                    pred = np.argmax(model.predict(cell))

                b[-1].append(pred)
                x += cw
            y += ch

        b = np.array(b)
        print('Original board: ') 
        print_board(b)
        solution, solved = solve_sudoku(b)

        if solved:
            print('------------------------')
            print('Solution:')
            print_board(solution)
        else:
            print('Could not be solved')
        

    def _get_digit(self, cell, percent_fill=0.03):
        cell_cnts, hierarchy = cv2.findContours(cell, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
        if len(cell_cnts) == 0:
            return None

        num_idx = 0
        # argmax: key = contourArea
        for idx, cnt in enumerate(cell_cnts):
            if cv2.contourArea(cnt) > cv2.contourArea(cell_cnts[num_idx]):
                num_idx = idx
        mask = np.zeros_like(cell)

        num_cnts = [cell_cnts[num_idx]]
        hierarchy = hierarchy.squeeze(axis=0)
        child = hierarchy[num_idx][2]
        # include child contours in mask
        while child >= 0:
            num_cnts.append(cell_cnts[child])
            child = hierarchy[child][0]

        mask = cv2.fillPoly(mask.copy(), num_cnts, (255,0,0))

        # check if looking at noise or number
        if cv2.countNonZero(mask) / cell.size < percent_fill:
            return None

        filtered = cv2.bitwise_and(cell, cell, mask=mask) # apply mask

        return filtered 