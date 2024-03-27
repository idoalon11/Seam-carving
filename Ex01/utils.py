import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from numba import jit
from tqdm import tqdm
from abc import abstractmethod, abstractstaticmethod


class SeamImage:
    def __init__(self, img_path, vis_seams=True):
        """ SeamImage initialization.
        Parameters:
            img_path (str): image local path
            method (str) (a or b): a for Hard Vertical and b for the known Seam Carving algorithm
            vis_seams (bool): if true, another version of the original image shall be store, and removed seams should be marked on it
        """
        #################
        # Do not change #
        #################
        self.path = img_path

        self.gs_weights = np.array([[0.299, 0.587, 0.114]]).T

        self.rgb = self.load_image(img_path)
        self.resized_rgb = self.rgb.copy()

        self.vis_seams = vis_seams
        if vis_seams:
            self.seams_rgb = self.rgb.copy()

        self.h, self.w = self.rgb.shape[:2]

        try:
            self.gs = self.rgb_to_grayscale(self.rgb)
            self.resized_gs = self.gs.copy()
            self.cumm_mask = np.ones_like(self.gs, dtype=bool)
        except NotImplementedError as e:
            print(e)

        try:
            self.E = self.calc_gradient_magnitude()
        except NotImplementedError as e:
            print(e)
        #################

        # additional attributes you might find useful
        self.seam_history = []
        self.seam_balance = 0

        # This might serve you to keep tracking original pixel indices 
        self.idx_map_h, self.idx_map_v = np.meshgrid(range(self.w), range(self.h))

    def rgb_to_grayscale(self, np_img):
        """ Converts a np RGB image into grayscale (using self.gs_weights).
        Parameters
            np_img : ndarray (float32) of shape (h, w, 3) 
        Returns:
            grayscale image (float32) of shape (h, w, 1)

        Guidelines & hints:
            Use NumpyPy vectorized matrix multiplication for high performance.
            To prevent outlier values in the boundaries, we recommend to pad them with 0.5
        """
        gray_scale_image = np.dot(np_img, self.gs_weights)
        # padding
        gray_scale_image[0, :] = 0.5
        gray_scale_image[-1, :] = 0.5
        gray_scale_image[:, 0] = 0.5
        gray_scale_image[:, -1] = 0.5
        return gray_scale_image

    def calc_gradient_magnitude(self):
        """ Calculate gradient magnitude of a grayscale image

        Returns:
            A gradient magnitude image (float32) of shape (h, w)

        Guidelines & hints:
            In order to calculate a gradient of a pixel, only its neighborhood is required.
        """
        grad_x = np.roll(self.resized_gs, -1, axis=1) - self.resized_gs
        grad_y = np.roll(self.resized_gs, -1, axis=0) - self.resized_gs
        gradient_magnitude = np.sqrt(grad_x ** 2 + grad_y ** 2)
        gradient_magnitude[gradient_magnitude > 1] = 1
        return gradient_magnitude

    def calc_M(self):
        pass

    def seams_removal(self, num_remove):
        pass

    def seams_removal_horizontal(self, num_remove):
        pass

    def seams_removal_vertical(self, num_remove):
        pass

    def rotate_mats(self, clockwise):
        self.gs = np.rot90(self.gs, k=clockwise)
        self.resized_gs = np.rot90(self.resized_gs, k=clockwise)
        self.rgb = np.rot90(self.rgb, k=clockwise)
        self.resized_rgb = np.rot90(self.resized_rgb, k=clockwise)
        self.cumm_mask = np.rot90(self.cumm_mask, k=clockwise)
        self.seams_rgb = np.rot90(self.seams_rgb, k=clockwise)

    def init_mats(self):
        pass

    def update_ref_mat(self):
        pass

    def backtrack_seam(self):
        pass

    def remove_seam(self):
        pass

    def reinit(self):
        """ re-initiates instance
        """
        self.__init__(self.path)

    @staticmethod
    def load_image(img_path):
        return np.asarray(Image.open(img_path)).astype('float32') / 255.0


class ColumnSeamImage(SeamImage):
    """ Column SeamImage.
    This class stores and implements all required data and algorithmics from implementing the "column" version of the seam carving algorithm.
    """

    def __init__(self, *args, **kwargs):
        """ ColumnSeamImage initialization.
        """
        super().__init__(*args, **kwargs)
        try:
            self.M = self.calc_M()
        except NotImplementedError as e:
            print(e)

    def calc_M(self):
        """ Calculates the matrix M discussed in lecture, but with the additional constraint:
            - A seam must be a column. That is, the set of seams S is simply columns of M. 
            - implement forward-looking cost

        Returns:
            A "column" energy matrix M (float32) of shape (h, w)

        Guidelines & hints:
            As taught, the energy is calculated from top to bottom.
            The formula of calculation M is as taught, but with certain terms omitted.
            You might find the function 'np.roll' useful.
        """
        c_v = np.abs(np.roll(self.resized_gs, -1, axis=1) - np.roll(self.resized_gs, 1, axis=1))
        m = np.cumsum(self.E + c_v, axis=0)
        return m

    def seams_removal(self, num_remove: int):
        """ Iterates num_remove times and removes num_remove vertical seams

        Parameters:
            num_remove (int): number of vertical seam to be removed

        Guidelines & hints:
        As taught, the energy is calculated from top to bottom.
        You might find the function np.roll useful.

        This step can be divided into a couple of steps:
            i) init/update matrices (E, M, backtracking matrix, seam mask) where:
                - E is the gradient magnitude matrix
                - M is the cost matrix
                - backtracking matrix is an idx matrix used to track the minimum seam from bottom up
                - mask is a boolean matrix for removed seams
            ii) seam backtracking: calculates the actual indices of the seam
            iii) index update: when a seam is removed, index mapping should be updated in order to keep track indices for next iterations
            iv) seam removal: create the carved image with the reduced (and update seam visualization if desired)
            Note: the flow described below is a recommendation. You may implement seams_removal as you with, but it needs to support:
            - removing seams a couple of times (call the function more than once)
            - visualize the original image with removed seams marked (for comparison)
        """
        self.E = self.calc_gradient_magnitude()
        self.M = self.calc_M()

        for i in range(num_remove):
            inx_seam_to_remove = np.argmin(self.M[-1])
            self.idx_map_h[:, inx_seam_to_remove:] = np.roll(self.idx_map_h[:, inx_seam_to_remove:], -1, axis=1)
            self.update_E(inx_seam_to_remove)
            self.update_M(inx_seam_to_remove)
            self.cumm_mask[:, inx_seam_to_remove] = 0
            self.remove_seam()

    def update_E(self, seam_idx):
        self.E = np.delete(self.E, seam_idx, axis=1)
        # check if the seam's index is the last column
        # and calculate grad_x
        if seam_idx == self.resized_gs.shape[1] - 1:
            grad_x = self.resized_gs[:, seam_idx]
        else:
            grad_x = self.resized_gs[:, seam_idx + 1] - self.resized_gs[:, seam_idx - 1]

        # calculate grad_y
        grad_y = self.resized_gs.copy()[:, seam_idx - 1]
        grad_y[:-1, :] = np.diff(grad_y, axis=0)

        # update the column
        self.E[:, seam_idx - 1] = np.sqrt(grad_x ** 2 + grad_y ** 2)

        # normalize
        self.E[:, seam_idx > 1] = 1

    def update_M(self, seam_idx):
        self.M = np.delete(self.M, seam_idx, axis=1)
        if seam_idx != self.resized_gs.shape[1] - 1:
            c_v = np.abs(self.resized_gs[:, seam_idx - 2] - self.resized_gs[:, seam_idx + 1])
            self.M[:, seam_idx - 1] = np.cumsum(self.E[:, seam_idx - 1], axis=0) + c_v

    def seams_removal_horizontal(self, num_remove):
        """ Removes num_remove horizontal seams

        Parameters:
            num_remove (int): number of horizontal seam to be removed

        Guidelines & hints:
            You may find np.rot90 function useful

        """
        self.idx_map_h, self.idx_map_v = np.meshgrid(range(self.h), range(self.w))
        self.rotate_mats(-1)
        self.seams_removal(num_remove)
        self.rotate_mats(1)
        self.idx_map_h, self.idx_map_v = np.meshgrid(range(self.w), range(self.h))


    def seams_removal_vertical(self, num_remove):
        """ A wrapper for removing num_remove horizontal seams (just a recommendation)

        Parameters:
            num_remove (int): number of vertical seam to be removed
        """
        self.seams_removal(num_remove)

    def backtrack_seam(self):
        """ Backtracks a seam for Column Seam Carving method
        """
        raise NotImplementedError("TODO: Implement SeamImage.backtrack_seam")

    def remove_seam(self):
        """ Removes a seam for self.rgb (you may create a resized version, like self.resized_rgb)

        Guidelines & hints:
        In order to apply the removal, you might want to extend the seam mask to support 3 channels (rgb) using: 3d_mak = np.stack([1d_mask] * 3, axis=2), and then use it to create a resized version.
        """

        false_col_index = np.where(~self.cumm_mask)[1][0]
        self.seams_rgb[:, (self.idx_map_h[0, false_col_index])] = (255, 0, 0)
        self.cumm_mask = np.delete(self.cumm_mask, false_col_index, axis=1)
        self.resized_gs = np.delete(self.resized_gs, false_col_index, axis=1)
        self.resized_rgb = np.delete(self.resized_rgb, false_col_index, axis=1)


class VerticalSeamImage(SeamImage):
    def __init__(self, *args, **kwargs):
        """ VerticalSeamImage initialization.
        """
        super().__init__(*args, **kwargs)
        try:
            self.M = self.calc_M()
        except NotImplementedError as e:
            print(e)

    def calc_M(self):
        """ Calculates the matrix M discussed in lecture (with forward-looking cost)

        Returns:
            An energy matrix M (float32) of shape (h, w)

        Guidelines & hints:
            As taught, the energy is calculated from top to bottom.
            You might find the function 'np.roll' useful.
        """
        m = self.E.copy()

        for i in range(m.shape[0]):
            j = 0
            m[i, j] += min(m[i - 1, j] + self.calc_cv(i, j), m[i - 1, (j + 1) % self.E.shape[1]] + self.calc_cr(i, j))

            # edge case - the last row
            j = m.shape[1] - 1
            m[i, j] += min(m[i - 1, j - 1] + self.calc_cl(i, j), m[i - 1, j] + self.calc_cv(i, j))

            # other cases
            j_values = np.arange(1, m.shape[1] - 1)
            j_prev = j_values - 1
            j_next = (j_values + 1) % self.E.shape[1]
            j_curr = j_values

            m[i, j_values] += np.minimum.reduce([m[i - 1, j_prev] + self.calc_cl(i, j_curr),
                                                 m[i - 1, j_curr] + self.calc_cv(i, j_curr),
                                                 m[i - 1, j_next] + self.calc_cr(i, j_curr)])

        return m

    def calc_cv(self, i, j):
        return np.abs(self.gs[i, (j + 1) % self.E.shape[1]] - self.gs[i, j - 1])

    def calc_cr(self, i, j):
        return np.abs(self.gs[i, (j + 1) % self.E.shape[1]] - self.gs[i, j - 1]) + np.abs(self.gs[i, (j + 1) % self.E.shape[1]] - self.gs[i - 1, j])

    def calc_cl(self, i, j):
        return np.abs(self.gs[i, (j + 1) % self.E.shape[1]] - self.gs[i, j - 1]) + np.abs(self.gs[i - 1, j] - self.gs[i, j - 1])

    def seams_removal(self, num_remove: int):
        """ Iterates num_remove times and removes num_remove vertical seams
        
        Parameters:
            num_remove (int): number of vertical seam to be removed

        Guidelines & hints:
        As taught, the energy is calculated from top to bottom.
        You might find the function np.roll useful.

        This step can be divided into a couple of steps:
            i) init/update matrices (E, M, backtracking matrix, saem mask) where:
                - E is the gradient magnitude matrix
                - M is the cost matrix
                - backtracking matrix is an idx matrix used to track the minimum seam from bottom up
                - mask is a boolean matrix for removed seams
            ii) fill in the backtrack matrix corresponding to M
            iii) seam backtracking: calculates the actual indices of the seam
            iv) index update: when a seam is removed, index mapping should be updated in order to keep track indices for next iterations
            v) seam removal: create the carved image with the reduced (and update seam visualization if desired)
            Note: the flow described below is a recommendation. You may implement seams_removal as you with, but it needs to supprt:
            - removing seams a couple of times (call the function more than once)
            - visualize the original image with removed seams marked (for comparison)
        """
        for i in range(num_remove):
            self.E = self.calc_gradient_magnitude()
            self.M = self.calc_M()
            self.backtrack_seam()
            self.remove_seam()

    def seams_removal_horizontal(self, num_remove):
        """ Removes num_remove horizontal seams

        Parameters:
            num_remove (int): number of horizontal seam to be removed

        Guidelines & hints:
            You may find np.rot90 function useful

        """
        self.idx_map_h, self.idx_map_v = np.meshgrid(range(self.h), range(self.w))
        self.rotate_mats(-1)
        self.seams_removal(num_remove)
        self.rotate_mats(1)
        self.idx_map_h, self.idx_map_v = np.meshgrid(range(self.w), range(self.h))

    def seams_removal_vertical(self, num_remove):
        """ A wrapper for removing num_remove horizontal seams (just a recommendation)

        Parameters:
            num_remove (int): umber of vertical seam to be removed
        """

        self.seams_removal(num_remove)
        self.seam_balance = self.seam_balance + num_remove

    def backtrack_seam(self):
        """ Backtracks a seam for Seam Carving as taught in lecture
        """
        # find the minimum value in the last row and backtracking
        j = np.argmin(self.M[-1])

        for i in range(self.M.shape[0] - 1, 0, -1):
            if self.M[i, j] == self.E[i, j] + self.M[i - 1, j] + self.calc_cv(i, j):
                self.idx_map_h[i, j:] = np.roll(self.idx_map_h[i - 1, j:], -1)
                j = j
            elif self.M[i, j] == self.E[i, j] + self.M[i - 1, j - 1] + self.calc_cl(i, j):
                self.idx_map_h[i, j - 1:] = np.roll(self.idx_map_h[i - 1, j - 1:], -1)
                j = j - 1
            elif self.M[i, j] == self.E[i, j] + self.M[i - 1, (j + 1) % self.E.shape[1]] + self.calc_cr(i, j):
                self.idx_map_h[i, j + 1:] = np.roll(self.idx_map_h[i - 1, j + 1:], -1)
                j = j + 1
            else:
                j = self.findTheCloset(i, j)
                self.idx_map_h[i, j:] = np.roll(self.idx_map_h[i, j:], -1)
            self.seams_rgb[i + self.seam_balance, (self.idx_map_h[i, j])] = (255, 0, 0)
            self.cumm_mask[i, j] = False
        self.seams_rgb[0, (self.idx_map_h[0, j])] = (255, 0, 0)
        self.cumm_mask[0, j] = False

        if self.seam_balance > 0:
            for i in range(self.seam_balance, 0, -1):
                if self.M[i, j] == self.E[i, j] + self.M[i - 1, j] + self.calc_cv(i, j):
                    j = j
                elif self.M[i, j] == self.E[i, j] + self.M[i - 1, j - 1] + self.calc_cl(i, j):
                    j = j - 1
                elif self.M[i, j] == self.E[i, j] + self.M[i - 1, (j + 1) % self.M.shape[1]] + self.calc_cr(i, j):
                    j = (j + 1) % self.M.shape[1]
                else:
                    j = self.findTheCloset(i, j)
                self.seams_rgb[i, (self.idx_map_h[i, j])] = (255, 0, 0)
                self.idx_map_h[i, j:] = np.roll(self.idx_map_h[i, j:], -1)

    def findTheCloset(self, i, j):
        minDiff = np.abs(self.M[i, j] - (self.E[i, j] + self.M[i - 1, j] + self.calc_cv(i, j)))
        if np.abs(self.M[i, j] - (self.E[i, j] + self.M[i - 1, j - 1] + self.calc_cl(i, j))) < minDiff:
            minDiff = np.abs(self.M[i, j] - (self.E[i, j] + self.M[i - 1, j - 1] + self.calc_cl(i, j)))
            j = (j - 1)
        if np.abs(self.M[i, j] - (self.E[i, j] + self.M[i - 1, (j + 1) % self.M.shape[1]]
                                  + self.calc_cr(i, j))) < minDiff:
            j = (j + 1) % self.M.shape[1]
        return j

    def remove_seam(self):
        """ Removes a seam from self.rgb (you may create a resized version, like self.resized_rgb)

        Guidelines & hints:
        In order to apply the removal, you might want to extend the seam mask to support 3 channels (rgb) using: 3d_mak = np.stack([1d_mask] * 3, axis=2), and then use it to create a resized version.
        """
        self.E = np.reshape(self.E[self.cumm_mask], (self.E.shape[0], self.E.shape[1] - 1, 1))
        self.M = np.reshape(self.M[self.cumm_mask], (self.M.shape[0], self.M.shape[1] - 1, 1))
        self.resized_gs = np.reshape(self.resized_gs[self.cumm_mask], (self.resized_gs.shape[0],
                                                                       self.resized_gs.shape[1] - 1, 1))
        self.resized_rgb = np.reshape(self.resized_rgb[self.cumm_mask.squeeze()], (self.resized_rgb.shape[0],
                                                                                   self.resized_rgb.shape[1] - 1, 3))
        self.cumm_mask = np.reshape(self.cumm_mask[self.cumm_mask], (self.cumm_mask.shape[0],
                                                                     self.cumm_mask.shape[1] - 1, 1))

    def seams_addition(self, num_add: int):
        """ BONUS: adds num_add seamn to the image

            Parameters:
                num_add (int): number of horizontal seam to be removed

            Guidelines & hints:
            - This method should be similar to removal
            - You may use the wrapper functions below (to support both vertical and horizontal addition of seams)
            - Visualization: paint the added seams in green (0,255,0)

        """
        raise NotImplementedError("TODO: Implement SeamImage.seams_addition")

    def seams_addition_horizontal(self, num_add):
        """ A wrapper for removing num_add horizontal seams (just a recommendation)

        Parameters:
            num_remove (int): number of horizontal seam to be added

        Guidelines & hints:
            You may find np.rot90 function useful

        """
        raise NotImplementedError("TODO (Bonus): Implement SeamImage.seams_addition_horizontal")

    def seams_addition_vertical(self, num_add):
        """ A wrapper for removing num_add vertical seams (just a recommendation)

        Parameters:
            num_add (int): number of vertical seam to be added
        """

        raise NotImplementedError("TODO (Bonus): Implement SeamImage.seams_addition_vertical")

    @staticmethod
    @jit(nopython=True)
    def calc_bt_mat(M, backtrack_mat):
        """ Fills the BT back-tracking index matrix. This function is static in order to support Numba. To use it,
        uncomment the decorator above.
        
        Recommended parameters (member of the class, to be filled):
            M: np.ndarray (float32) of shape (h,w)
            backtrack_mat: np.ndarray (int32) of shape (h,w): to be filled here

        Guidelines & hints:
            np.ndarray is a reference type. changing it here may be affected outside.
        """
        raise NotImplementedError("TODO: Implement SeamImage.calc_bt_mat")


def scale_to_shape(orig_shape: np.ndarray, scale_factors: list):
    """ Converts scale into shape

    Parameters:
        orig_shape (np.ndarray): original shape [y,x]
        scale_factors (list): scale factors for y,x respectively

    Returns
        the new shape
    """
    return tuple(np.round(orig_shape * scale_factors).astype(int))


def resize_seam_carving(seam_img: SeamImage, shapes: np.ndarray):
    """ Resizes an image using Seam Carving algorithm

    Parameters:
        seam_img (SeamImage) The SeamImage instance to resize
        shapes (np.ndarray): desired shape (y,x)

    Returns
        the resized rgb image
    """
    seam_img.reinit()
    seam_img.seams_removal_vertical(shapes[0][1] - shapes[1][1])
    seam_img.seams_removal_horizontal(shapes[0][0] - shapes[1][0])
    return seam_img.resized_rgb


def bilinear(image, new_shape):
    """
    Resizes an image to new shape using bilinear interpolation method
    :param image: The original image
    :param new_shape: a (height, width) tuple which is the new shape
    :returns: the image resized to new_shape
    """
    in_height, in_width, _ = image.shape
    out_height, out_width = new_shape
    new_image = np.zeros(new_shape)

    ###Your code here###
    def get_scaled_param(org, size_in, size_out):
        scaled_org = (org * size_in) / size_out
        scaled_org = min(scaled_org, size_in - 1)
        return scaled_org

    scaled_x_grid = [get_scaled_param(x, in_width, out_width) for x in range(out_width)]
    scaled_y_grid = [get_scaled_param(y, in_height, out_height) for y in range(out_height)]
    x1s = np.array(scaled_x_grid, dtype=int)
    y1s = np.array(scaled_y_grid, dtype=int)
    x2s = np.array(scaled_x_grid, dtype=int) + 1
    x2s[x2s > in_width - 1] = in_width - 1
    y2s = np.array(scaled_y_grid, dtype=int) + 1
    y2s[y2s > in_height - 1] = in_height - 1
    dx = np.reshape(scaled_x_grid - x1s, (out_width, 1))
    dy = np.reshape(scaled_y_grid - y1s, (out_height, 1))
    c1 = np.reshape(image[y1s][:, x1s] * dx + (1 - dx) * image[y1s][:, x2s], (out_width, out_height, 3))
    c2 = np.reshape(image[y2s][:, x1s] * dx + (1 - dx) * image[y2s][:, x2s], (out_width, out_height, 3))
    new_image = np.reshape(c1 * dy + (1 - dy) * c2, (out_height, out_width, 3)).astype(int)
    return new_image