import time

from imutils.video import VideoStream
from joblib import load
#import tensorflow as tf
import imutils
import cv2 as cv
import numpy as np


CHECK_FRAME_FREQ = 1 # 32
CHARS = ['A', 'E', 'I', 'O', 'U']
A = cv.imread('images/a.png')
E = cv.imread('images/e.png')
I = None
O = cv.imread('images/o.png')
U = None
LETTER_IMAGES = [A, E, I, O, U]
CORNER_IDS = (923, 1001, 241, 1007)
CACHED_REF_POINTS = None
DIVIDER = np.zeros((480, 2, 3))
BLACKOUT = np.zeros((480, 319))
NUM_EIG_VECTORS = 100
EIGEN_SORTED_INDICES = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99]


def draw(img, corners, imgpts):
    corner = tuple(corners[0].ravel())
    img = cv.line(img, corner, tuple(imgpts[0].ravel()), (255,0,0), 5)
    img = cv.line(img, corner, tuple(imgpts[1].ravel()), (0,255,0), 5)
    img = cv.line(img, corner, tuple(imgpts[2].ravel()), (0,0,255), 5)
    return img


# need 5 points [bottom_left, top, bottom_right, mid_left, mid_right]
def draw_a(img, imgpts):
    img = cv.line(img, tuple(imgpts[0].ravel()), tuple(imgpts[1].ravel()), (255, 255, 0), 5)
    img = cv.line(img, tuple(imgpts[2].ravel()), tuple(imgpts[1].ravel()), (255, 255, 0), 5)
    img = cv.line(img, tuple(imgpts[3].ravel()), tuple(imgpts[4].ravel()), (255, 255, 0), 5)
    return img


# need 6 points [bottom_left, top_left, bottom_right, mid_left, mid_right, top_right]
def draw_e(img, imgpts):
    img = cv.line(img, tuple(imgpts[0].ravel()), tuple(imgpts[1].ravel()), (0, 255, 0), 5)
    img = cv.line(img, tuple(imgpts[0].ravel()), tuple(imgpts[2].ravel()), (0, 255, 0), 5)
    img = cv.line(img, tuple(imgpts[3].ravel()), tuple(imgpts[4].ravel()), (0, 255, 0), 5)
    img = cv.line(img, tuple(imgpts[1].ravel()), tuple(imgpts[5].ravel()), (0, 255, 0), 5)
    return img


# need 6 points [bottom_left, bottom_right, bottom_mid, top_mid, top_left, top_right]
def draw_i(img, imgpts):
    img = cv.line(img, tuple(imgpts[0].ravel()), tuple(imgpts[1].ravel()), (0, 0, 0), 5)
    img = cv.line(img, tuple(imgpts[2].ravel()), tuple(imgpts[3].ravel()), (0, 0, 0), 5)
    img = cv.line(img, tuple(imgpts[4].ravel()), tuple(imgpts[5].ravel()), (0, 0, 0), 5)
    return img    


# need 1 point [centre]
def draw_o(img, imgpts):
    img = cv.ellipse(img, tuple(imgpts[0].ravel()), (60, 80), 0, 0, 360, (255, 255, 255), 10)
    return img


# need 1 point [centre]
def draw_u(img, imgpts):
    img = cv.ellipse(img, tuple(imgpts[0].ravel()), (40, 120), 0, 0, 180, (0, 255, 255), 10)
    return img


def main():
    # model = tf.keras.models.load_model('isl_model_v3')
    print('loading eigenspace...')
    eigenspace = np.load('eigenspace.npy')
    print('loading classifier...')
    knn_classifier = load('knn_classifier.joblib')
    print('loading camera parameters...')
    # with open('params.yml') as cam_params_file:
    #     cam_params = yaml.load(cam_params_file)
    cam_params_file = cv.FileStorage('params.yml', cv.FileStorage_READ)
    # cam_params = cam_params_file.
    matrix = cam_params_file.getNode('camera_matrix').mat()#.reshape(3, 3)
    # print(matrix)
    distortion = cam_params_file.getNode('distortion_coefficients').mat()#.reshape(5, 1)
    criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    objp = np.zeros((6*7,3), np.float32)
    objp[:,:2] = np.mgrid[0:7,0:6].T.reshape(-1,2)
    axis = np.float32([[3,0,0], [0,3,0], [0,0,-3]]).reshape(-1,3)
    a_points = np.float32([[0, 0, 0], [3, 0, -4], [6, 0, 0], [1.5, 0, -2], [4.5, 0, -2]])
    e_points = np.float32([[0, 0, 0], [0, 0, -4], [3, 0, 0], [0, 0, -2], [3, 0, -2], [3, 0, -4]])
    i_points = np.float32([[0, 0, 0], [2, 0, 0], [1, 0, 0], [1, 0, -4], [0, 0, -4], [2, 0, -4]])
    o_points = np.float32([[2, 0, -2]])
    u_points = np.float32([[2, 0, -2]])

    arucoDict = cv.aruco.Dictionary_get(cv.aruco.DICT_ARUCO_ORIGINAL)
    arucoParams = cv.aruco.DetectorParameters_create()
    show_video = True
    print('starting camera...')
    video_stream = VideoStream(src=0, framerate=20).start()
    # load letter images
    letters_images = [A]
    time.sleep(2)
    print('loading screen...')
    frame_num = 0
    letter_seen = None
    while show_video:
        video_frame = video_stream.read()
        # new_frame = imutils.resize(frame, width=480, height=640)
        # TODO: run model in separate thread - if thread is running then ignore new frames
        if frame_num % CHECK_FRAME_FREQ == 0:
            img = video_frame.copy()
            # res = cv.bitwise_not(video_frame, video_frame, mask)
            grey = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
            # TODO: experiment with thresholding to make arm/hand clear against a black background
            # print(grey.shape)
            _, img = cv.threshold(grey, 110, 255, cv.THRESH_TOZERO_INV)
            #grey[:, 321:] = BLACKOUT
            img = img[:, :321]
            img = cv.resize(img, (120, 160))
            img = cv.flip(img, 1)
            img = cv.GaussianBlur(img, (15, 15), sigmaX=2.6, sigmaY=2.6)
            #img = img.transpose()
            img_vector = img.flatten()
            img_input = eigenspace @ img_vector
            img_input = img_input[EIGEN_SORTED_INDICES][:NUM_EIG_VECTORS].reshape(1, -1)
            prediction = knn_classifier.predict(img_input)
            pred_char = CHARS[prediction[0]]
            print(pred_char)
            ret, corners = cv.findChessboardCorners(grey, (7, 6), None)
            if ret:
                corners2 = cv.cornerSubPix(grey, corners, (11,11), (-1,-1), criteria)
                # Find the rotation and translation vectors.
                ret,rvecs, tvecs = cv.solvePnP(objp, corners2, matrix, distortion)
                # project 3D points to image plane
                # imgpts, jac = cv.projectPoints(axis, rvecs, tvecs, matrix, distortion)
                if pred_char == 'A':
                    imgpts, jac = cv.projectPoints(a_points, rvecs, tvecs, matrix, distortion)
                    video_frame = draw_a(video_frame, imgpts)
                elif pred_char == 'E':
                    imgpts, jac = cv.projectPoints(e_points, rvecs, tvecs, matrix, distortion)
                    video_frame = draw_e(video_frame, imgpts)
                elif pred_char == 'I':
                    imgpts, jac = cv.projectPoints(i_points, rvecs, tvecs, matrix, distortion)
                    video_frame = draw_i(video_frame, imgpts)
                elif pred_char == 'O':
                    imgpts, jac = cv.projectPoints(o_points, rvecs, tvecs, matrix, distortion)
                    video_frame = draw_o(video_frame, imgpts)
                elif pred_char == 'U':
                    imgpts, jac = cv.projectPoints(u_points, rvecs, tvecs, matrix, distortion)
                    video_frame = draw_u(video_frame, imgpts)
            # grey = np.stack((grey, grey, grey), axis=2)
            # grey = grey.transpose((1, 0, 2))
            # video_frame = img # grey
            # tf_frame = tf.expand_dims(grey, axis=0)
            # prediction = model(tf_frame)
            # prediction = tf.nn.sigmoid(prediction)
            # prediction_index = tf.argmax(prediction, axis=1)[0]
            # if prediction[0][prediction_index] > 0.95:
            #     letter_seen = LETTER_IMAGES[prediction_index]
            # else:
            #     letter_seen = None
            # print(f'{CHARS[prediction_index]}: {prediction[0]}')
        frame_num += 1
        frame_num = frame_num % CHECK_FRAME_FREQ
        # video_frame[:, 319:321, :] = DIVIDER
        cv.imshow('Sign with your right hand - hold the markers with your left hand', video_frame) # put instructions in title
        cv.imshow('What the classifier sees', img)
        key = cv.waitKey(1) & 0xff
        if key == ord('q'):
            show_video = False

    cv.destroyAllWindows()
    video_stream.stop()


if __name__ == '__main__':
    main()
