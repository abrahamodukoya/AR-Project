import numpy as np
from imutils.video import VideoStream
from joblib import load
from PIL import Image
from OpenGL.GLUT import glutPostRedisplay
import cv2 as cv
from constants import *
import opengl_lib as ogl

criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)
objp = np.zeros((6*7,3), np.float32)
objp[:,:2] = np.mgrid[0:7,0:6].T.reshape(-1,2)
matrix = None
distortion = None
video_stream = VideoStream(src=0, framerate=20).start()
threshold = 127
eigenspace = None
knn_classifier = None
classifier_img = None
show_classifier_img_freq = 0


def process_image_for_classification(img, threshold):
    grey = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    _, img = cv.threshold(grey, threshold, 255, cv.THRESH_TOZERO_INV)
    img = img[:, :321]
    img = cv.resize(img, (120, 160))
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


def update_scene():
    global classifier_img
    video_frame = video_stream.read()
    img = video_frame.copy()
    print(f'Greyscale thresholding value (0 <= value <= 255): {threshold}')
    img, grey = process_image_for_classification(img, threshold)
    classifier_img = img
    img_input = create_image_vector(img, eigenspace)
    pred_char = predict_letter(img_input, knn_classifier)
    print(f'Predicted letter: {pred_char}')
    ogl.curr_img.texture_image = Image.fromarray(video_frame).tobytes('raw', 'RGB', 0, -1)
    grey = cv.cvtColor(video_frame, cv.COLOR_BGR2GRAY)
    ret, corners = cv.findChessboardCorners(grey, (7, 6), None)
    if ret:
        ogl.curr_model = ogl.char_models[CHARS.index(pred_char)]
        corners2 = cv.cornerSubPix(grey, corners, (11,11), (-1,-1), criteria)
        ret, rvecs, tvecs = cv.solvePnP(objp, corners2, matrix, distortion)
        rmat = np.identity(4)
        rodrigues_mat, _ = cv.Rodrigues(rvecs)
        rmat[0:3, 0:3] = rodrigues_mat
        rmat = np.transpose(rmat)
        ogl.curr_camera_model = np.identity(4)
        tvecs = tvecs.flatten()
        tvecs[2] = -tvecs[2] if -tvecs[2] > 0 else 0
        tvecs[1] = -tvecs[1]
        tvecs[0] = -tvecs[0]
        tmat = np.identity(4)
        # column-major order
        tmat[0:3, 3] = 0.05 * tvecs
        tmat = np.transpose(tmat)
        scale_mat = np.identity(4)
        scale_mat[0, 0] = 0.05
        scale_mat[1, 1] = 0.05
        scale_mat[2, 2] = 0.05
        # change sign of x, y
        change_mat = np.identity(4)
        change_mat[1, 1] = -change_mat[1, 1]
        change_mat[0, 0] = -change_mat[0, 0]

        # tmat first because we're using column major order
        ogl.curr_camera_model = rmat @ scale_mat @ change_mat @ tmat
    else:
        ogl.curr_model = None
    glutPostRedisplay()


def main():
    global matrix, distortion, eigenspace, knn_classifier, threshold
    print('loading eigenspace...')
    eigenspace = np.load('blurred_eigenspace.npy')
    print('loading classifier...')
    knn_classifier = load('blurred_knn_classifier.joblib')
    print('loading camera parameters...')
    cam_params_file = cv.FileStorage('params.yml', cv.FileStorage_READ)
    matrix = cam_params_file.getNode('camera_matrix').mat()
    distortion = cam_params_file.getNode('distortion_coefficients').mat()
    print('setting up OpenGL...')
    ogl.launch(update_scene)
    print('setting up classifier image view')
    show_video = True
    while show_video:
        if classifier_img is not None:
            cv.imshow('What the classifier sees', classifier_img)
            key = cv.waitKey(1) & 0xff
            if key == ord('q'):
                show_video = False
            elif key == ord('-') and threshold > 0:
                threshold -= 1
            elif key == ord('+') and threshold < 255:
                threshold += 1

    cv.destroyAllWindows()


if __name__ == '__main__':
    main()
