import cv2
import numpy as np
import easyocr

def processing(img, reader):
    img = cv2.imread(img)
    cascade = cv2.CascadeClassifier("ocr\\cascade.xml")
    gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    nplate=cascade.detectMultiScale(gray,1.1,4)
    plate = [None]
    result = ""
    for (x,y,w,h) in nplate:
        wT,hT,cT=img.shape
        a,b=(int(0.02*wT),int(0.02*hT))
        plate=img[y+a:y+h-a,x+b:x+w-b,:]
        kernel=np.ones((1,1),np.uint8)
        plate=cv2.dilate(plate,kernel,iterations=1)
        plate=cv2.erode(plate,kernel,iterations=1)
        plate_gray=cv2.cvtColor(plate,cv2.COLOR_BGR2GRAY)
        (thresh,plate)=cv2.threshold(plate_gray,127,255,cv2.THRESH_BINARY)
        prediction = reader.readtext(plate)
        for i in prediction:
            result += i[1] + " "

        result = result[:-1]
        cv2.rectangle(img,(x,y),(x+w,y+h),(51,51,255),2)
        cv2.rectangle(img,(x-1,y-40),(x+w+1,y),(51,51,255),-1)
        cv2.putText(img,result,(x,y-10),cv2.FONT_HERSHEY_SIMPLEX,0.9,(255,255,255),2)
    
    return plate, img, result

def detect(image_path):
    reader = easyocr.Reader(['en'])
    plate, img, result = processing(image_path, reader)
    if len(plate) < 2:
        prediction = reader.readtext(img)
        for i in prediction:
            result += i[1] + " "
        x, y = ((int) (img.shape[1]/2 - 268/2), (int) (img.shape[0]/2 - 36/2))
        cv2.rectangle(img,(x-1,y-40),(x+170+1,y),(51,51,255),-1)
        cv2.putText(img,result,(x,y-10),cv2.FONT_HERSHEY_SIMPLEX,0.9,(255, 255, 255),2)
        img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        
        return  img, result
        
    else:
        img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        return  img, result