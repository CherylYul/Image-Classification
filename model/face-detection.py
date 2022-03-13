# import
import os
import shutil
import joblib
import json

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import cv2
import pywt

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.pipeline import make_pipeline
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

# load image
img = cv2.imread("./test_images/messi.jpg")
convert_color = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
print(convert_color.shape)
plt.imshow(convert_color, cmap="gray")
plt.savefig("messi_converted_to_gray.png")
plt.show()

# using haarcascade from OpenCV to detect features
face_cascade = cv2.CascadeClassifier(
    "./haarcascades/haarcascade_frontalface_default.xml"
)
eye_cascade = cv2.CascadeClassifier("./haarcascades/haarcascade_eye.xml")
faces = face_cascade.detectMultiScale(convert_color, 1.3, 5)
print(faces)

# draw the rectangle detect the face and the eye
(x, y, w, h) = faces[0]
face_img = cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
gray_region = convert_color[y : y + h, x : x + w]
color_region = face_img[y : y + h, x : x + w]
eyes = eye_cascade.detectMultiScale(gray_region)
for (ex, ey, ew, eh) in eyes:
    cv2.rectangle(color_region, (ex, ey), (ex + ew, ey + eh), (0, 255, 0), 2)
plt.figure()
plt.imshow(face_img, cmap="gray")
plt.savefig("rectangle_detected1.png")
plt.show()
plt.imshow(color_region, cmap="gray")
plt.savefig("rectangle_detected2.png")
plt.show()

# create function that crop the face location
def get_face_location(image_path):
    img = cv2.imread(image_path)
    convert_color = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(convert_color, 1.3, 5)
    for (x, y, w, h) in faces:
        face_img = cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
        gray_region = convert_color[y : y + h, x : x + w]
        color_region = img[y : y + h, x : x + w]
        eyes = eye_cascade.detectMultiScale(gray_region)
        if len(eyes) >= 2:
            return color_region


cropped_image = get_face_location("./test_images/messi.jpg")
plt.imshow(cropped_image)
plt.savefig("cropped_image.png")
plt.show()

# get cropped image
data_path = "./dataset/"
cropped_path = "./dataset/cropped/"
img_dirs = []
for i in os.scandir(data_path):
    if i.is_dir():
        img_dirs.append(i.path)

if os.path.exists(cropped_path):
    shutil.rmtree(cropped_path)
os.mkdir(cropped_path)

# image directories
cropped_image_dirs = []
file_dict = {}
for img_dir in img_dirs:
    count = 1
    name = img_dir.split("/")[-1]
    print(name)
    file_dict[name] = []
    for i in os.scandir(img_dir):
        t = get_face_location(i.path)
        if t is not None:
            cropped_folder = cropped_path + name
            if not os.path.exists(cropped_folder):
                os.makedirs(cropped_folder)
                cropped_image_dirs.append(cropped_folder)
                print("Generating: ", cropped_folder)
            cropped_file_name = name + str(count) + ".png"
            cropped_file_path = cropped_folder + "/" + cropped_file_name
            cv2.imwrite(cropped_file_path, t)
            file_dict[name].append(cropped_file_path)
            count += 1

# feature engineering using wavelet transform
def w2d(img, mode="haar", level=1):
    img_convert = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    img_convert = np.float32(img_convert)
    img_convert /= 255
    coeffs = pywt.wavedec2(img_convert, mode, level=level)
    coeffs = list(coeffs)
    coeffs[0] *= 0
    img_convert = pywt.waverec2(coeffs, mode)
    img_convert *= 255
    img_convert = np.uint8(img_convert)
    return img_convert


im_har = w2d(cropped_image, "db1", 5)
plt.imshow(im_har, cmap="gray")
plt.savefig("feature_image.png")
plt.show()

# create dict again
file_dict = {}
for img_dir in cropped_image_dirs:
    name = img_dir.split("/")[-1]
    file_list = []
    for i in os.scandir(img_dir):
        file_list.append(i.path)
    file_dict[name] = file_list

class_dict = {
    "elon_musk": 0,
    "lionel_messi": 1,
    "serena_williams": 2,
    "taylor_swift": 3,
    "virat_kohli": 4,
}

x, y = [], []
for name, training_files in file_dict.items():
    for training_image in training_files:
        img = cv2.imread(training_image)
        scaled_raw_img = cv2.resize(img, (32, 32))
        img_har = w2d(img, "db1", 5)
        scaled_img_har = cv2.resize(img_har, (32, 32))
        combined_img = np.vstack(
            (scaled_raw_img.reshape(32 * 32 * 3, 1), scaled_img_har.reshape(32 * 32, 1))
        )
        x.append(combined_img)
        y.append(name)
print(len(x))
print(len(x[0]))
x = np.array(x).reshape(len(x), 4096).astype(float)
print(x.shape)

# training the model
x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=0)
pipe = Pipeline([("scaler", StandardScaler()), ("svc", SVC(kernel="rbf", C=10))])
pipe.fit(x_train, y_train)
pipe.score(x_test, y_test)

print(classification_report(y_test, pipe.predict(x_test)))

# using GridSearchCV in hyper parameter tunning
model_params = {
    "svm": {
        "model": SVC(gamma="auto", probability=True),
        "params": {"svc_C": [1, 10, 100, 1000], "svc_kernel": ["rbf", "linear"]},
    },
    "random_forest": {
        "model": RandomForestClassifier(),
        "params": {"randomforest_n_estimators": [1, 5, 10]},
    },
    "logistic_regression": {
        "model": LogisticRegression(solver="liblinear", multi_class="auto"),
        "params": {"logisticregression_C": [1, 5, 10]},
    },
}

scores = []
best_estimators = {}
for algo, mp in model_params.items():
    pipe = make_pipeline(StandardScaler(), mp["model"])
    clf = GridSearchCV(pipe, mp["params"], cv=5, return_train_score=False)
    clf.fit(x_train, y_train)
    scores.append(
        {"model": algo, "best_score": clf.best_score_, "best_params": clf.best_params_}
    )
    best_estimators[algo] = clf.best_estimator_
df = pd.DataFrame(scores, columns=["model", "best_score", "best_params"])
print(df)
print(best_estimators["svm"].score(x_test, y_test))

# plot confusion matrix
best_clf = best_estimators["svm"]
cm = confusion_matrix(y_test, best_clf.predict(x_test))
plt.figure(figsize=(10, 7))
sns.heatmap(cm, annot=True)
plt.xlabel("Predicted")
plt.ylabel("Truth")
plt.savefig("confusion_matrix.png")
plt.show()

# Save model
joblib.dump(best_clf, "best_model.pkl")
with open("class_dictionary.json", "w") as f:
    f.write(json.dump(class_dict))
