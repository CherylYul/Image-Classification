# Image Classification: Face Detection

## Data
Data collect from google images and project study from Dhavel Patel teachers.

## Method
Detect a person from image using face detection, try to detect 2 eyes. If can find 2 eyes clearly we will keep the image otherwise we will discarded
 * Cleaning Images: cropped faces with 2 eyes using haar cascade in open cv (80-90% cleaning), Pictures without 2 eyes clearly or just one side face will be discarded, then run a process of manual verification to make sure that classifiers accuracy stay high
 * Feature Engineering: using wavelet transform to extract the facial features such as eyes nose and lips in white and black color
 * Build Model: using GridSearchCv to hyper parameter tune the logistic Regression, Randrom Forest, Support Vector Machine model to figure out the best score one.
 * Save Model: in pickle file