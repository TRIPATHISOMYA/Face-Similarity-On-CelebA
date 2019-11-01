import cv2
import os
import pandas as pd
import numpy as np


def loadImages():
    final = pd.read_csv('identity_CelebA.txt',sep=" ",names = ['image_id','labels'])
    face_detector = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    final = final[final['labels'].isin(list(final.labels.value_counts()[:600].index))].reset_index(drop=True)
    final['labels'] = final.labels.apply(lambda x: str(x))
    new_d = {}
    j = 0
    for i in list(final.labels.unique()):
        new_d[i]=j
        j+=1
    final['labels'] = final['labels'].apply(lambda x : new_d[x])


    imgs_array = []
    filenames=[]
    labels = []
    for img_file in list(final.image_id):
        try:
            img = cv2.imread('./data/celebA/{}'.format(img_file))
            label=int(final[final['image_id']==img_file.split('/')[-1]].to_dict(orient='record')[0]['labels'])   
            (x,y,w,h) = face_detector.detectMultiScale(img, 1.1, minNeighbors=8)[0]
            face = img[y:y+h, x:x+w]
            face = cv2.resize(face,(100,100))
            imgs_array.append(face.flatten())
            filenames.append(img_file)
            labels.append(label)
        except:
            print(img_file)
    return imgs_array, filenames, labels

def generate_data():
    imgs_array, filenames, labels = loadImages()
    final_data = {'img_array':imgs_array,'file_name':filenames,'label':labels}
    np.save('final_emb',final_data)
    
    
if __name__ == "__main__":
    generate_data()
