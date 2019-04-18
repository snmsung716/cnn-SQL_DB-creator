
import sqlite3
import time
import datetime
import random
import numpy as np

conn = sqlite3.connect("object_info.db")
c= conn.cursor()

def create_table():
    c.execute("create table if not exists\
    stuffToPlot(unix REAL, name TEXT, axis_x TEXT,axis_y TEXT)")

 
def dynamic_data_entry(disease_name, x_axis, y_axis):

    unix = Mac
    name = disease_name
    axis_x = x_axis
    axis_y = y_axis
    
    c.execute("INSERT INTO stuffToPlot (unix,name, axis_x,axis_y) VALUES(?,?,?,?)",
             (unix,name,axis_x,axis_y))
    conn.commit()
    
def close():
    c.close()
    conn.close()
    

    
create_table()
#data_entry()
for i in range(10):
    dynamic_data_entry()
    time.sleep(0.01)

