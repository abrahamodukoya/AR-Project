# CS7GV4 Project

Before running, use OpenCV to calibrate your webcam, saving the result to params.yml.  
To run:
1. Upload PCA\_of\_ISL.ipynb to Google Colab and run. Download the kNN classifier (or use the one in the repository) and eigenspace matrix and save them in the same directory as video.py and opengl_lib.py.
2. Install the requirements in requirements.txt.
3. Print of an A6 sized 9x6 chessboard from OpenCV.
4. Run video.py. Note that the eigenspace matrix uses a lot of memory, so it may be best to close other memory intensive applications.
5. Hold the chessboard in your left hand, ensuring that the chessboard is clearly visible in the camera frame, while signing an ISL vowel with your right hand, ensuring that your right hand is by itself in the right-hand side of the camera frame.

A smaller window showing the image seen by the classifier will load after a few seconds. If your hand does not show clearly in this video stream, use the + and - keys to change the pixel thresholding level. This only works when the smaller window has focus.
