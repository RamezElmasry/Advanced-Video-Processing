import numpy as np
import cv2
import math

img=cv2.imread('Lenna.png')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
dimensions=gray.shape
height=gray.shape[0]
width=gray.shape[1]
c1=[]
c2=[]
m1=200
m2=15
s1=2
s2=5
w1=0.5
w2=0.5
t=0.1
img1=np.zeros_like(gray)
img2=np.zeros_like(gray)
b=True 
N = 0
while b:
    print(N)
    img1=np.zeros_like(gray)
    img2=np.zeros_like(gray) 
    for i in range(height):
        for j in range(width):
            d1=(1/( ((2*math.pi)**0.5)*s1 )) * math.exp( (-1/(2*s1**2)) * (gray[i,j]-m1)**2)
            d2=(1/( ((2*math.pi)**0.5)*s2 )) * math.exp( (-1/(2*s2**2)) * (gray[i,j]-m2)**2)
            r1=d1*w1/((d1+d2)+0.0001)
            r2=d2*w2/((d1+d2)+0.0001)
            if(r1>r2):
                c1.append(gray[i,j])
                img1[i,j]=255
            if(r2>r1):
                c2.append(gray[i,j])
                img2[i,j]=255            
    M1=(sum(c1)/ (len(c1)+0.00001))
    M2=(sum(c2)/ (len(c2)+0.00001)) 
    S1=((((sum(((np.array(c1)-(sum(c1)/ (len(c1)+0.00001)))**2)))/(len(c1)+0.001))))**0.5
    S2=((((sum(((np.array(c1)-(sum(c1)/ (len(c1)+0.00001)))**2)))/(len(c2)+0.001))))**0.5   
    W1=len(c1)/(width*height)
    W2=len(c2)/(width*height)
    if ((abs(M1-m1)<t) and (abs(M2-m2)<t) and ((S1-s1)<t) and ((S2-s2)<t)):
        print(M1-m1, M2-m2, S1-s1, S2-s2)
        b=False
    m1=M1
    m2=M2
    s1=S1
    s2=S2
    w1=W1
    w2=W2
    N+=1        
for i in range(height):
    for j in range(width):
        d1=(1/( ((2*math.pi)**0.5)*s1 )) * math.exp( (-1/(2*s1**2)) * (gray[i,j]-m1)**2)
        d2=(1/( ((2*math.pi)**0.5)*s2 )) * math.exp( (-1/(2*s2**2)) * (gray[i,j]-m2)**2)
        r1=d1*w1/((d1+d2)+0.0001)
        r2=d2*w2/((d1+d2)+0.0001)
        if(r1>r2):
            c1.append(gray[i,j])
            img1[i,j]=255
        if(r2>r1):
            c2.append(gray[i,j])
            img2[i,j]=255            
     
cv2.imshow('image1', img1)
cv2.imwrite('image1.jpg',img1)
cv2.imshow('image2', img2) 
cv2.imwrite('image2.jpg',img2)   
cv2.waitKey(0)
cv2.destroyAllWindows()
