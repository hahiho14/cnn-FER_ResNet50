import os
from tkinter import *
from glob import glob
from pymediainfo import MediaInfo

from keras.models import load_model, model_from_json
from pandastable import Table

import cv2
import math
import statistics
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#//from keras.preprocessing.image import img_to_array

class FER_Prototype:
    def __init__(self, master):
        self.master = master
        
        master.title("Prototype Face Expression Recognition")
        
        self.listbox = Listbox(master, width=60, height=20)
        self.label = Label(root, text = "Video list:") 
        self.video_label = Label(root, text = "Select a file")
        #self.table = Frame(root)
         
        self.t = Toplevel(master)
        self.t.wm_title("Output Table")
        self.l = Label(self.t)  

        #self.image_list()
        self.video_list()
        self.load_env()

        # initialize button
        self.close_button = Button(master, text="Close", command=master.quit)
        #self.predict_button = Button(master, text="Predict", command=self.predict_image, state=DISABLED)
        self.predict_button = Button(master, text="Predict", command=self.predict_video, state=DISABLED)

        # grid display
        self.label.grid(row=0, column=0, columnspan=2, sticky=W+E)
        self.listbox.grid(row=1, column=0, columnspan=2, sticky=W+E)
        self.video_label.grid(row=2, column=0, columnspan=2, sticky=W+E)
        self.close_button.grid(row=3, column=0)
        self.predict_button.grid(row=3, column=1)

        self.listbox.bind("<Double-Button>", lambda x: self.play())
        #self.listbox.bind("<<ListboxSelect>>", self.properties())
        self.listbox.bind("<<ListboxSelect>>", self.selected)

    def load_env(self):
        #/self.model = load_model('e:/fer/model/ResNet-50.hdf')
        #//self.model = model_from_json(open("e:/fer/model/face_model.json", "r").read())
        #//self.model = load_model('e:/fer/model/face_model.h5')
        self.model = load_model('e:/fer/model/my_model.h5')
        self.face_cascade = cv2.CascadeClassifier('e:/fer/haarcascade/haarcascade_frontalface_default.xml')    
        #/self.input_shape = (197, 197)
        #//self.input_shape = (48, 48)
        self.input_shape = (48, 48)
        self.emotions = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']
        self.satisfactions = ["neutral", "negative", "positive"]

    def video_list(self):
        fl = glob("E:\\fer\\data\\video\\*.mp4")
        # image path
        #fl = glob("E:\\fer\\data\\extracted-video\\1\\*.jpg")
        for f in fl:
            self.listbox.insert(0,f)

    def fname(self):
        return self.listbox.get(self.listbox.curselection())

    def play(self):
        os.startfile(self.fname())

    #   video detail
    def properties(self):
        path = self.fname()
        self.path_tail = os.path.split(path)[-1]
        self.path_tail = self.path_tail.replace('.mp4', '')
        #print(self.path_tail)
        #print(type(self.path_tail))
        size = os.path.getsize(path) // 1000000
        duration = MediaInfo.parse(path)
        ms = int(duration.tracks[0].duration / 3600) *4
        filedata = f"File size: {size:n} MB\n Duration: {ms} seconds"
        self.video_label['text'] = filedata
        

    def selected(self, event):
        self.properties()
        widget = event.widget
        selection=widget.curselection()
        self.value = widget.get(selection[0])
        #print ("selection", selection, ": '{0}'" .format(self.value))
        self.predict_button.configure(state=NORMAL)

    def loop(self):
        # reading all the frames from temp folder
        #images = []
        #images.append("E:\\fer\\data\\extracted-video\\{0}\\*.jpg" .format(self.path_tail))
        
        images = glob("E:\\fer\\data\\extracted-video\\{0}\\*.jpg" .format(self.path_tail))
        images.sort(key=lambda f: int(re.sub('\D', '', f)))
        #print(images)
        for filename in images:
            # update img path
            self.img = cv2.imread(filename)
            self.img_filename = os.path.split(filename)[-1]
            self.img_filename = self.img_filename.replace('.jpg', '')
            self.predict_image()  

    def predict_image(self):
        # update img path
        #self.img = cv2.imread('%s'% self.value)

        gray = cv2.cvtColor(self.img, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray, 1.3, 5)
        
        for (x,y,w,h) in faces:
            roi_color = self.img[y:y+h, x:x+w]
            #//roi_gray = gray[y:y+h, x:x+w] #give border
            #//detected_face = roi_gray #get detected face
            detected_face = roi_color #get detected face
            detected_face = cv2.resize(detected_face, self.input_shape) #resize to 48x48
            #//detected_face = img_to_array(detected_face)
            detected_face = np.expand_dims(detected_face, axis=0) #add 1 dimension in the front of array

            predictions = self.model.predict(detected_face)
                
            #find max indexed array
            max_index = np.argmax(predictions[-1])
                
            #emotions = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']
            #satisfactions = ["neutral", "negative", "positive"]
            emotion = self.emotions[max_index]
            if max_index == 3 or max_index == 5:
                satisfy = self.satisfactions[2]
            elif max_index == 6:
                satisfy = self.satisfactions[0]
            else:
                satisfy = self.satisfactions[1]      
            
            self.img = cv2.putText(self.img, emotion, (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2) #white text
            #cv2.putText(img, emotion, (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX, 1, (105,105,105), 2) #black text
            self.img = cv2.rectangle(self.img,(x,y),(x+w,y+h),(255,0,0),2)
            #cv2.imshow('img',self.img)
            self.img_arr.append(self.img)
            self.frame_arr.append(self.img_filename)
            self.emotion_idx.append(max_index)
            self.emotion_arr.append(emotion)
            self.satisfy_arr.append(satisfy)
            #if cv2.waitKey(0) & 0xFF == ord('q'): #press q to quit
            #    break
            

    def predict_video(self):
        self.img_arr, self.frame_arr, self.emotion_idx, self.emotion_arr, self.satisfy_arr, count_emotion_idx = [], [], [], [], [], []
        count_positive, count_neutral, count_negative = 0, 0, 0

        # update vid path
        self.vid = cv2.VideoCapture('{0}' .format(self.value))
        self.extract_video()
        self.loop()
        #print(self.frame_arr)
        #print(self.emotion_arr)
        
        for idx in self.emotion_idx:
            if idx == 3 or idx == 5:
                count_positive+=1
            elif idx == 6:
                count_neutral+=1
            else:
                count_negative+=1
        count_emotion_idx.append(count_neutral)
        count_emotion_idx.append(count_negative)
        count_emotion_idx.append(count_positive)
        # get the most satisfaction
        most_emotion = max(count_emotion_idx)
        satisfy_idx = count_emotion_idx.index(most_emotion)
        satisfy = self.satisfactions[satisfy_idx]
        # get the most emotion
        avg_emotion = statistics.mode(self.emotion_idx)
        avg_emotion = self.emotions[avg_emotion]
        
        message1 = "Emosi yang paling sering muncul adalah {0}\n Kepuasan customer terhadap website {1}\n\n Output tabel dan video akan disimpan pada path berikut: ".format(avg_emotion, satisfy)
        message2 = "E:/fer/data/output-table/ dan E:/fer/data/output-video/"

        self.image_to_video()
        self.plot_table()
        self.popup("Report", message1, message2)
        #return self.vid
    
    def extract_video(self):
        count = 0
        cap = self.vid
        frameRate = cap.get(5) #get frame rate
        x = 1
        while(cap.isOpened()):
            frameId = cap.get(1) #current frame number
            ret, frame = cap.read()
            if(ret != True):
                break
            if(frameId % math.floor(frameRate) == 0):
                filename = "E:\\fer\\data\\extracted-video\\{0}\\{1}.jpg" .format(self.path_tail, count);count+=1
                cv2.imwrite(filename, frame)
        cap.release()

    def plot_table(self):
        output = np.vstack((self.frame_arr, self.emotion_arr, self.satisfy_arr)).T
        df_output = pd.DataFrame({'Frame': output[:, 0], 'Expression': output[:, 1], 'Satisfation': output[:, 2]})
        # create excel writer object
        writer = pd.ExcelWriter("E:\\fer\data\output-table\{0}.xlsx" .format(self.path_tail))
        # write dataframe to excel
        df_output.to_excel(writer)
        # save the excel
        writer.save()
        #print(df_output)
        #l.pack(side="top", fill="both", expand=True, padx=100, pady=100)
        pt = Table(self.l, dataframe=df_output, showtoolbar=True, showstatusbar=True)
        pt.show()
        self.l.pack()

    def popup(self, title, message1, message2):
        root = Tk()
        root.title(title)
        w = 400     # popup window width
        h = 200     # popup window height
        sw = root.winfo_screenwidth()
        sh = root.winfo_screenheight()
        x = (sw - w)/2
        y = (sh - h)/2
        root.geometry('%dx%d+%d+%d' % (w, h, x, y))
        m = message1
        m += '\n'
        m += message2
        w = Label(root, text=m, width=120, height=10)
        w.pack()
        b = Button(root, text="OK", command=root.destroy, width=10)
        b.pack()
        mainloop()  
   
    def image_to_video(self):
        size=(1280, 720)
        filepath = "E:\\fer\\data\\output-video\\{0}.avi" .format(self.path_tail)
        out = cv2.VideoWriter(filepath, cv2.VideoWriter_fourcc(*'DIVX'), 1, size)    
        for i in range(len(self.img_arr)):
            # writing to a image array
            out.write(self.img_arr[i])
            #print(i)
        out.release()
        
root = Tk()
my_gui = FER_Prototype(root)
root.mainloop()