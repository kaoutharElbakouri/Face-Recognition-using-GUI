import cv2
import tkinter as tk
from tkinter import filedialog
from tkinter.filedialog import askopenfile
from PIL import Image, ImageTk
import cv2
import face_recognition
import os
import numpy as np



KNOWN_FACES_DIR = "known_faces"
known_faces = []
known_names = []
TOLERANCE = 0.6

for filename in os.listdir(KNOWN_FACES_DIR):
    image = face_recognition.load_image_file(f"{KNOWN_FACES_DIR}/{filename}")
    encoding = face_recognition.face_encodings(image)[0]
    known_faces.append(encoding)
    known_names.append(filename)

my_w = tk.Tk()
my_w.geometry("700x700")  # Size of the window 
my_w.title('Face Recognition')
my_font1=('times', 18, 'bold')
l1 = tk.Label(my_w,text='Add new Image',width=30,font=my_font1)  
l1.grid(row=1,column=1)
b1 = tk.Button(my_w, text='Upload File', 
   width=20,command = lambda:upload_file())
b1.grid(row=2,column=1) 

def upload_file():
    global img
    f_types = [('Jpg Files', '*.jpg')]
    filename = filedialog.askopenfilename(filetypes=f_types)
    frame = face_recognition.load_image_file(filename)
    face_locations = face_recognition.face_locations(frame)
    unknown_face_encoding = face_recognition.face_encodings(frame,face_locations)
    for face_encoding, face_loc in zip(unknown_face_encoding, face_locations):
        results = face_recognition.compare_faces(known_faces, face_encoding,TOLERANCE)
        top_left = (face_loc[1], face_loc[0])
        bottom_right = (face_loc[3], face_loc[2])
        color = (0, 0, 200)
        cv2.rectangle(frame, top_left, bottom_right, color , 4)
        match = None
        if True in np.array(results):
                match = known_names[results.index(True)]
                font = cv2.FONT_HERSHEY_COMPLEX
                cv2.putText(frame,match[:len(match)-4],(face_loc[1]-150, face_loc[0]),cv2.FONT_HERSHEY_SIMPLEX , 1 , (255,255,255) ,2)
        else:
                font = cv2.FONT_HERSHEY_COMPLEX
                cv2.putText(frame,"unknown",(face_loc[1]-150, face_loc[0]),cv2.FONT_HERSHEY_SIMPLEX , 1 , (255,255,255) ,2 )
  
    #how to display an image from a numpy array in tkinter
    image=img = Image.fromarray(frame)
    image = image.resize((700, 600), Image.ANTIALIAS)
    img = ImageTk.PhotoImage(image)
    b2 =tk.Button(my_w,image=img) 
    b2.grid(row=3,column=1)
my_w.mainloop()
