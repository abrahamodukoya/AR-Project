import time

import numpy as np
from imutils.video import VideoStream
from joblib import load
from PIL import Image
from OpenGL.GLUT import glutPostRedisplay
# import imutils
import cv2 as cv
from constants import *
import opengl_lib as ogl

criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)
objp = np.zeros((6*7,3), np.float32)
objp[:,:2] = np.mgrid[0:7,0:6].T.reshape(-1,2)

a_points = np.float32([[0, 0, 0], [3, 0, -4], [6, 0, 0], [1.5, 0, -2], [4.5, 0, -2]])
e_points = np.float32([[0, 0, 0], [0, 0, -4], [3, 0, 0], [0, 0, -2], [3, 0, -2], [3, 0, -4]])
i_points = np.float32([[0, 0, 0], [2, 0, 0], [1, 0, 0], [1, 0, -4], [0, 0, -4], [2, 0, -4]])
o_points = np.float32([[2, 0, -2]])
u_points = np.float32([[2, 0, -2]])

curr_pred_char = 'A'
curr_video_frame = np.zeros((640, 480, 3))
matrix = None
distortion = None
video_stream = VideoStream(src=0, framerate=20).start()

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


def draw_letter(letter, video_frame, rvecs, tvecs, matrix, distortion):
    if letter == 'A':
        imgpts, jac = cv.projectPoints(a_points, rvecs, tvecs, matrix, distortion)
        video_frame = draw_a(video_frame, imgpts)
    elif letter == 'E':
        imgpts, jac = cv.projectPoints(e_points, rvecs, tvecs, matrix, distortion)
        video_frame = draw_e(video_frame, imgpts)
    elif letter == 'I':
        imgpts, jac = cv.projectPoints(i_points, rvecs, tvecs, matrix, distortion)
        video_frame = draw_i(video_frame, imgpts)
    elif letter == 'O':
        imgpts, jac = cv.projectPoints(o_points, rvecs, tvecs, matrix, distortion)
        video_frame = draw_o(video_frame, imgpts)
    elif letter == 'U':
        imgpts, jac = cv.projectPoints(u_points, rvecs, tvecs, matrix, distortion)
        video_frame = draw_u(video_frame, imgpts)

    return video_frame


def process_image_for_classification(img, threshold):
    grey = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    # TODO: experiment with thresholding to make arm/hand clear against a black background
    # print(grey.shape)
    _, img = cv.threshold(grey, threshold, 255, cv.THRESH_TOZERO_INV)
    #grey[:, 321:] = BLACKOUT
    img = img[:, :321]
    img = cv.resize(img, (120, 160))
    img = cv.flip(img, 1)
    img = cv.GaussianBlur(img, (15, 15), sigmaX=2.6, sigmaY=2.6)
    return img, grey


def create_image_vector(img, eigenspace):
    img_vector = img.flatten()
    img_input = eigenspace @ img_vector
    img_input = img_input[EIGEN_SORTED_INDICES][:NUM_EIG_VECTORS].reshape(1, -1)
    return img_input


def predict_letter(img_input, knn_classifier):
    prediction = knn_classifier.predict(img_input)
    return CHARS[prediction[0]]


def display_result(pred_char, video_frame, grey, matrix, distortion):
    ret, corners = cv.findChessboardCorners(grey, (7, 6), None)
    if ret:
        corners2 = cv.cornerSubPix(grey, corners, (11,11), (-1,-1), criteria)
        # Find the rotation and translation vectors.
        ret, rvecs, tvecs = cv.solvePnP(objp, corners2, matrix, distortion)
        rmat, _ = cv.Rodrigues(rvecs)
        print(f'rotation matrix: {rmat}')
        print(f'translation vectors: {tvecs}')
        # TODO: add rmat and tvecs to model matrix for the letter model
        # project 3D points to image plane
        video_frame = draw_letter(pred_char, video_frame, rvecs, tvecs, matrix, distortion)
    return video_frame


def update_scene():
    # TODO: run prediction here
    # TODO: figure out how to use perspective/projection matrix
    # global curr_model, curr_img
    # print('updating scene')
    video_frame = video_stream.read()
    ogl.curr_img.texture_image = Image.fromarray(video_frame).tobytes('raw', 'RGB', 0, -1)
    grey = cv.cvtColor(video_frame, cv.COLOR_BGR2GRAY)
    ret, corners = cv.findChessboardCorners(grey, (7, 6), None)
    if ret:
        ogl.curr_model = ogl.char_models[CHARS.index(curr_pred_char)]
        corners2 = cv.cornerSubPix(grey, corners, (11,11), (-1,-1), criteria)
        ret, rvecs, tvecs = cv.solvePnP(objp, corners2, matrix, distortion)
        rmat = np.identity(4)
        rodrigues_mat, _ = cv.Rodrigues(rvecs)
        rmat[0:3, 0:3] = rodrigues_mat
        rmat[0, 1] = -rmat[0, 1]
        rmat[1, 0] = -rmat[1, 0]
        rmat = np.transpose(rmat)
        ogl.curr_camera_model = np.identity(4)
        tvecs = tvecs.flatten()
        # swap y and z
        # tmp = tvecs[1]
        # tvecs[1] = tvecs[2]
        # tvecs[2] = tmp
        tvecs[2] = -tvecs[2]
        tvecs[1] = -tvecs[1]
        tvecs[0] = -tvecs[0]
        tmat = np.identity(4)
        # trying column-major order
        tmat[0:3, 3] = 0.05 * tvecs
        tmat = np.transpose(tmat)
        scale_mat = np.identity(4)
        scale_mat[0, 0] = 0.05
        scale_mat[1, 1] = 0.05
        scale_mat[2, 2] = 0.05
        # 90 degrees about the x-axis
        rot_mat = np.identity(4)  # translates to OpenGL coord system
        rot_mat[1, 1] = 0
        rot_mat[1, 2] = 1
        rot_mat[2, 1] = -1
        rot_mat[2, 2] = 0
        # change sign of y (may need to do the same for z)
        change_y_mat = np.identity(4)
        change_y_mat[1, 1] = -change_y_mat[1, 1]
        inverse_matrix = np.transpose(np.array([[ 1.0, 1.0, 1.0, 1.0],
                               [-1.0,-1.0,-1.0,-1.0],
                               [-1.0,-1.0,-1.0,-1.0],
                               [ 1.0, 1.0, 1.0, 1.0]]))
        # ogl.curr_camera_model = tmat @ rmat @ scale_mat @ rot_mat #@ inverse_matrix
        # ogl.curr_camera_model = tmat @ scale_mat #@ ogl.curr_camera_model #@ inverse_matrix
        # tmat first because we're using column major order
        ogl.curr_camera_model = rmat @ scale_mat @ change_y_mat @ tmat# np.transpose(ogl.curr_camera_model)
    else:
        ogl.curr_model = None
    glutPostRedisplay()
    


def main():
    global curr_pred_char, curr_video_frame, matrix, distortion, video_stream
    # print('loading eigenspace...')
    # eigenspace = np.load('eigenspace.npy')
    print('loading classifier...')
    knn_classifier = load('knn_classifier.joblib')
    print('loading camera parameters...')
    cam_params_file = cv.FileStorage('params.yml', cv.FileStorage_READ)
    matrix = cam_params_file.getNode('camera_matrix').mat()
    distortion = cam_params_file.getNode('distortion_coefficients').mat()
    print('setting up OpenGL...')
    ogl.launch(update_scene)

    show_video = True
    print('starting camera...')
    # video_stream = VideoStream(src=0, framerate=20).start()
    # time.sleep(2)
    print('loading screen...')

    frame_num = 0
    threshold = 127
    while show_video:
        video_frame = video_stream.read()
        # curr_video_frame = video_frame.copy()
        # TODO: run model in separate thread - if thread is running then ignore new frames
        if frame_num % CHECK_FRAME_FREQ == 0:
            img = video_frame.copy()
            img, grey = process_image_for_classification(img, threshold)
            img_input = create_image_vector(img, eigenspace)
            pred_char = predict_letter(img_input, knn_classifier)
            curr_pred_char = pred_char
            print(pred_char)
            video_frame = display_result(pred_char, video_frame, grey, matrix, distortion)
                
        frame_num += 1
        frame_num = frame_num % CHECK_FRAME_FREQ
        # video_frame[:, 319:321, :] = DIVIDER
        cv.imshow('Sign with your right hand - hold the markers with your left hand', video_frame) # put instructions in title
        cv.imshow('What the classifier sees', img)
        key = cv.waitKey(1) & 0xff
        if key == ord('q'):
            show_video = False
        elif key == ord('-') and threshold > 0:
            threshold -= 1
        elif key == ord('+') and threshold < 255:
            threshold += 1

    cv.destroyAllWindows()
    video_stream.stop()


if __name__ == '__main__':
    main()
