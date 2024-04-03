from __future__ import division
import numpy as np
import cv2
import math
from PIL import Image as im
vid= cv2.VideoCapture('TownCentreXVID.avi')

#i'm trying to divide my image into foreground and background (2 classes)
mean1= 170
var1=2

mean2= 50
var2=5

w1=0.5 #weight of class 1, their summation has to be 1
w2=0.5 #weight of class 2


t=100
alpha=1/(t+0.00001)
print(alpha)


count =0
fps= int(vid.get(cv2.CAP_PROP_FPS))
print(fps)
width= int(vid.get(cv2.CAP_PROP_FRAME_WIDTH))
height= int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT))
fourcc = cv2.VideoWriter_fourcc(*'XVID')
# GMM foregroundvideo
out2= cv2.VideoWriter('foreground.avi', cv2.VideoWriter_fourcc(*'DIVX'), 30, (width,height),0)


ret, frame = vid.read()

grayscaled = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)

gaussian1 = np.zeros((height,width))

w1_mat = np.full((height,width), w1)
w2_mat = np.full((height,width), w2)

var1_mat = np.full((height,width), var1)
var2_mat = np.full((height,width), var2)

mean1_mat = np.full((height,width), mean1)
mean2_mat = np.full((height,width), mean2)

oA=np.zeros((height,width))
oB=np.zeros((height,width))
beta1 = 0
beta2 = 0
while(vid.isOpened()):
    ret, frame = vid.read()
    grayscaled = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)  

    std1=var1_mat**(1/2)
    a= (1/(np.sqrt(2*np.pi)*std1)) * (np.exp( (-1/(2*(std1**2)) ) * ((grayscaled-mean1_mat)**2) ) )
    
    std2= var2_mat**(1/2)
    b= (1/(np.sqrt(2*np.pi)*std2)) * ( np.exp( (-1/(2*(std2**2)) ) * ((grayscaled-mean2_mat)**2) ) )

    tempa= a*w1_mat
    tempb= b*w2_mat
    
    gaussian1[ tempa > tempb] = 255
    gaussian1[ tempa <= tempb] = 0

    oA[ tempa > tempb] = 1
    oA[ tempa <= tempb] = 0

    oB[ tempb > tempa] = 1
    oB[ tempb <= tempa] = 0

    beta1= (grayscaled[ tempa > tempb] -mean1)
    beta2= (grayscaled[ tempa <= tempb]-mean2)

    b1= (grayscaled -mean1)
    b2= (grayscaled -mean2)


    w1_mat = w1_mat +  alpha * (oA - w1_mat)  #* (len(beta1)) )
    mean1_mat = mean1_mat + ( (alpha / (w1_mat + 0.00001) ) * (oA * b1) )
    var1_mat = var1_mat + oA * ( (alpha / (w1_mat + 0.00001) ) * ( ( ( (oA * b1)**2 ) - (var1_mat) ) ) )
        
    w2_mat = w2_mat +  alpha * (oB - w2_mat)
    mean2_mat = mean2_mat + ( (alpha / (w2_mat + 0.00001) ) * (oB * b2) )
    var2_mat = var2_mat + oB * ( (alpha / (w2_mat + 0.00001) ) * ( ( ( (oB * b2)**2 ) - (var2_mat) ) ) )


    ww1 = w1_mat / (w1_mat + w2_mat)
    ww2 = w2_mat / (w1_mat + w2_mat)

    w1_mat = ww1
    w2_mat = ww2

    out2.write(gaussian1.astype("uint8"))
    #cv2.imshow("aaa",gaussian1) 
    cv2.waitKey(1)    

    
    if ret==False:
        break     
    if cv2.waitKey(fps) & 0xFF == ord('q'): #to quit the video
        cv2.destroyAllWindows()
        break     

vid.release()
out2.release()
cv2.destroyAllWindows()
