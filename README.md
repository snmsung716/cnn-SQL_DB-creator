# cnn-SQL_DB-creator
How to extract the data of object on DB_SQL based on opencv + make use of this DB on Raspberry Pi.

The name of object and the location of object can be saved on Database.


![OJB_wH-7](https://user-images.githubusercontent.com/42028366/56334231-77a77380-61d2-11e9-96ed-e08e3b819700.jpg)


And then download package.

`pip install -r requirement.txt`

you might train your images which you put in the data. this below code would be run.

`python3 obj_train.py`

if you would like to have test images for the prediction, put your image on alien test on data and run this code.

`python3 obj_test.py`

and to use opencv with the model which you made, this below code can be used

`python3 obj_recog.py`

since running this code you might create the file, info_detected.db.

This code has the x and y axis of coordinate and the name of object. If you add some more information you can create the table and add what you want.

For the next step, automatically you can send this file to Raspberry Pi and You can use this one on the another project, Queen Ant, which I uploaded on Github.


![maxresdefault](https://user-images.githubusercontent.com/42028366/56333545-8fc9c380-61cf-11e9-8737-2f888ade751b.jpg)

