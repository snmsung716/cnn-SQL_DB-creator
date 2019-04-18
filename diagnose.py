import cv2
import numpy as np
from keras.preprocessing.image import ImageDataGenerator
from keras import optimizers
from keras.models import Sequential, load_model
from keras.layers import Dropout, Flatten, Dense, Activation
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras import callbacks

import sqlite3
import time
import datetime
import random
import numpy as np

def main():
    model_path = './models/model.h5'
    model_weights_path = './models/weights.h5'
    test_path = 'data/alien_test'

    #Load the pre-trained models
    model = load_model(model_path)
    model.load_weights(model_weights_path)

    conn = sqlite3.connect("info_detected.db")
    c= conn.cursor()

    def create_table():
        c.execute("create table if not exists\
        stuffToPlot(unix INT, name TEXT, axis_x TEXT,axis_y TEXT)")


    def dynamic_data_entry(disease_name, x_axis, y_axis):

        unix = 0
        name = disease_name
        axis_x = x_axis
        axis_y = y_axis

        c.execute("INSERT INTO stuffToPlot (unix,name, axis_x,axis_y) VALUES(?,?,?,?)",
                 (unix,name,axis_x,axis_y))
        conn.commit()

    def close():
        c.close()
        conn.close()

    def get_disease(y_pred):
        for number, per in enumerate(y_pred[0]):
            if per != 0:
                final_number = str(int(number))
                per = round((per * 100), 2)
                return final_number, per


    video = cv2.VideoCapture(0)
    count = 0
    disease_name =[]
    x_axis = []
    y_axis = []
    if(video.isOpened()):
        while True:
            count += 1
            if count % 10 == 0:
                print(count)

            
            check, img = video.read()
            cv2.imshow("Frame", img)

            #Display purpose
            img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
            img_gau = cv2.GaussianBlur(img_gray, (5, 5), 0)
            ret, thresh = cv2.threshold(img_gau, 80, 255, cv2.THRESH_BINARY_INV)
            # cv2.imshow("Frame thersh",thresh)

            kernel = np.ones((5, 5), np.uint8)
            dilation = cv2.dilate(thresh, kernel, iterations=1)
            # cv2.imshow("Frame dilation", thresh)

            edged = cv2.Canny(dilation, 50, 250)
            # cv2.imshow("Frame edged", thresh)

            key = cv2.waitKey(1)

            if key == 27:
                break
            elif key & 0xFF == ord('c'):
                cv2.imwrite('output/capture.jpg',img)
                capture_img = cv2.imread('output/capture.jpg')

                img2 = capture_img.copy()
                img_gray = cv2.cvtColor(capture_img, cv2.COLOR_RGB2GRAY)
                img_gau = cv2.GaussianBlur(img_gray, (5, 5), 0)
                ret, thresh = cv2.threshold(img_gau, 80, 255, cv2.THRESH_BINARY_INV)

                kernel = np.ones((5, 5), np.uint8)
                dilation = cv2.dilate(thresh, kernel, iterations=1)

                edged = cv2.Canny(dilation, 50, 250)

                _, contours, hierachy = cv2.findContours(edged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

                num_str = ''
                per = ''

                num_list = []
                if len(contours) > 0:
                    for c in contours:
                        if cv2.contourArea(c) > 2500:
                            x, y, w, h = cv2.boundingRect(c)
                            cv2.rectangle(img2, (x, y), (x + w, y + h), (255, 0, 255), 3)
                            append.x_axis(x)
                            append.y_axis(y)

                            new_img = thresh[y:y + h, x:x + w]
                            new_img2 = cv2.resize(new_img, (28, 28))
                            im2arr = np.array(new_img2)
                            im2arr = im2arr.reshape(1, 28, 28, 1)
                            y_pred = model.predict(im2arr)

                            num, per = get_disease(y_pred)
                            num_list.append(str(int(num)))
                            num_str = '[' + str(str(int(num))) + ']'
                            append.disease_name(num_str)
                            cv2.putText(img2, num_str, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 0, 0), 2, cv2.LINE_AA)

                str1 = ' '.join(num_list)
                if (str1 != ''):
                    y_p = str('Predicted Value is ' + str(str1))
                    print(y_p)
                    cv2.putText(img2, y_p, (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
                cv2.imshow("Capture Frame", img2)
                cv2.imshow("Contours Frame", thresh)
                
            if count > 10:
                x_axis = np.array(x_axis).reshape(-1)
                x_axis = x_axis.flatten().tolist()
                y_axis = np.array(y_axis).reshape(-1)
                y_axis = y_axis.flatten().tolist()
                print("name:", disease_name, "\nx,y:", x_axis, y_axis)
                try:
                    create_table()
                    for i in range(10):
                        dynamic_data_entry(disease_name=disease_name, x_axis=x_axis, y_axis=y_axis)
                        time.sleep(0.01)
                    close()
                except:
                    pass
                
                video.release()
                cv2.destroyAllWindows()
                return  disease_name, x_axis, y_axis




    video.release()
    cv2.destroyAllWindows()

if __name__=="__main__":
    main()