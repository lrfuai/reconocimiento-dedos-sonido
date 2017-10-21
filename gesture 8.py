#https://github.com/vipul-sharma20/gesture-opencv/blob/master/gesture.py
#nestor balich 15-10--2017 Ok
import cv2
import numpy as np
import math
import subprocess
import time


valSet = 90  #  BINARY threshold
cap_region_x_begin=0.5  # start point/total width
cap_region_y_end=0.8  # start point/total width

proc = subprocess.Popen(['python', 'speech_to_text.py', ''], stdout=subprocess.PIPE, stderr=subprocess.PIPE)

def printValSet(valor):
    print("! se Cambio valo de seteo de prueba "+str(valor))

def play_pyttsx(sTexto):
    global proc
    retcode = proc.poll()
        
    if retcode is not None: # Process finished.
        proc = subprocess.Popen(['python', 'speech_to_text.py', sTexto], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        print ("Ejecutando speech asincronico")
   
def calculateFingers(res,drawing):  # -> finished bool, cnt: finger count
    #  convexity defect
    hull = cv2.convexHull(res, returnPoints=False)
    if len(hull) > 2:
        defects = cv2.convexityDefects(res, hull)
        if type(defects) != type(None):  # avoid crashing.   (BUG not found)
            
            cnt = 0
            for i in range(defects.shape[0]):  # calculate the angle
                s, e, f, d = defects[i][0]
                start = tuple(res[s][0])
                end = tuple(res[e][0])
                far = tuple(res[f][0])
                a = math.sqrt((end[0] - start[0]) ** 2 + (end[1] - start[1]) ** 2)
                b = math.sqrt((far[0] - start[0]) ** 2 + (far[1] - start[1]) ** 2)
                c = math.sqrt((end[0] - far[0]) ** 2 + (end[1] - far[1]) ** 2)
                angle = math.acos((b ** 2 + c ** 2 - a ** 2) / (2 * b * c))  # cosine theorem
                             
                if  (angle <= math.pi / 2 ):  # math.pi / 2 (angle >= 0.7) and (angle <= 1.5): angle less than 90 degree, treat as fingers
                    cnt += 1
                    #dibuja circulo en el punto de defecto de convergencia interno
                    cv2.circle(drawing, far, 8, [211, 84, 0], -1) 
                    # dibuja una linea que une los puntos externo OK
                    #cv2.line(drawing,start,end,(0,255,0),4)
                    #dibuja un ciculo en el punto externo de la mano
                    cv2.circle(crop_img, end, 8, [100, 255, 255], -1)
                    cv2.circle(drawing, end, 8, [100, 255, 255], -1)
                    #escribe coordenadas sobre los dedos
                    sAux = "["+str(end[0])+","+str(end[1])+"]"
                    cv2.putText(drawing, sAux, (end[0],end[1]-15), cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.65, (255,255,255),1)
                   

                # elif  (angle > 2 ) and (angle <  2 ) :    
                #     cv2.circle(drawing, far, 8, [211, 211, 0], -1)
                #     cv2.line(drawing,start,end,(255,255,0),4)
                #print( "cnt:",cnt," p1:" , start ," p2:" , end , " far:" , far ,"Angle:",angle)
                #print( start[0] - far[0])

            return True, cnt
    return False, 0

bGuardar = True
cap = cv2.VideoCapture(0)
cap.set(10,200)

# cv2.namedWindow('valor seteo prueba')
# cv2.createTrackbar('valor', 'valor seteo prueba', valSet, 360, printValSet)

ret, img = cap.read()
img = cv2.flip(img, 1)
cv2.rectangle(img, (0,int(img.shape[1]*0.3) ), (int(img.shape[0]*0.5),img.shape[1]), (255,0,0),2)
aBor = 0 #ajusta el borde del contorno
crop_img = img[int(img.shape[1]*0.3)+aBor:img.shape[0]-aBor, aBor:int(img.shape[0]*0.5)-aBor]

drawing = crop_img

# variables para el control de permanencia en estado
iPaso = 0
sMgs =  ""
paso = [0, 0, 0, 0, 0, 0]

while cap.isOpened():
    ret, img = cap.read()
    img = cv2.flip(img, 1)
    # valSet = cv2.getTrackbarPos('valor', 'valor seteo prueba')

    # metodo 1 de definicion de zona de captura
    #cv2.rectangle(img, (int(cap_region_x_begin * img.shape[1]), 0),
    #             (img.shape[1], int(cap_region_y_end* img.shape[0])), (255, 0, 0), 2)
                 
    #crop_img = img[0:int(cap_region_y_end * img.shape[0]),
    #                int(cap_region_x_begin * img.shape[1]):img.shape[1]]  # clip the ROI

    # metodo 2 de definicion de zona de captura
    cv2.rectangle(img, (0,int(img.shape[1]*0.3) ), (int(img.shape[0]*0.5),img.shape[1]), (255,0,0),2)
    crop_img = img[int(img.shape[1]*0.3)+aBor:img.shape[0]-aBor, aBor:int(img.shape[0]*0.5)-aBor]

    if (bGuardar):
        #guarda la imagen final
        cv2.imwrite( 'fondo_box1.png',crop_img)
        bGuardar=False

    #lee el fondo de la imagen
    fondo_box1 = cv2.imread('fondo_box1.png') 
    #Calculamos la diferencia absoluta de las dos imagenes
    resta = cv2.subtract(fondo_box1,crop_img)
    #resta = cv2.bitwise_not(resta)
    
    # filtro de desenfoque
    resta = cv2.blur(resta,(5,5)) 


    # convert to grayscale
    grey = cv2.cvtColor(resta, cv2.COLOR_BGR2GRAY)

    # applying gaussian blur
    value = (15, 15)
    blurred = cv2.GaussianBlur(grey, value, 0)

    # thresholdin: Otsu's Binarization method
    _, thresh1 = cv2.threshold(blurred, 35, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    #_, thresh1 = cv2.threshold(blurred, valSet, 255, cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)

    



    # check OpenCV version to avoid unpacking error
    (version, _, _) = cv2.__version__.split('.')

    if version == '3':
        _, contours, hierarchy = cv2.findContours(thresh1.copy(), \
               cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)    
    elif version == '2':
        contours, hierarchy = cv2.findContours(thresh1.copy(),cv2.RETR_TREE, \
               cv2.CHAIN_APPROX_NONE)

    # find contour with max area metodo 1
    #cnt = max(contours, key = lambda x: cv2.contourArea(x))

    
     # find contour with max area metodo 2
    length = len(contours)
    maxArea = -1
    count_defects = 0
     
    if length > 0:
        ci=0 
        for i in range(length):  # find the biggest contour (according to area)
            temp = contours[i]
            area = cv2.contourArea(temp)
            if area < 34000 and area > 20000 :
                if area > maxArea :
                    maxArea = area
                    ci = i
            #        print (ci)
            #print (area)

            res = contours[ci]

    # print (cv2.contourArea(cnt))
    # create bounding rectangle around the contour (can skip below two lines)
    # x, y, w, h = cv2.boundingRect(cnt)
    # cv2.rectangle(crop_img, (x, y), (x+w, y+h), (0, 0, 255), 0)

             # finding convex hull
            hull = cv2.convexHull(res)

        # dibuja los conteornos
            drawing = np.zeros(crop_img.shape,np.uint8)
            cv2.drawContours(drawing, [res], 0, (0, 255, 0), 0)
            cv2.drawContours(drawing, [hull], 0,(0, 0, 255), 4)
   
            isFinishCal,count_defects = calculateFingers(res,drawing)
            
            #    drawing = np.zeros(crop_img.shape,np.uint8)



    #define los puntos de superosicion
    # rX = 200
    # rY = 250
    # y =[0,9,15,16,15,8]
    # x =[9,0,4,9,13,18] 

    # for num in range(0,6):
    #      pCenter = ( rX+  x[num] * 10,rY - y[num] * 10)
    #      cv2.circle(img, pCenter, 15, [0,0,255], 3)

    # print (count_defects)
    # define actions required


    if count_defects == 1:
            iPaso = 1
            sMgs =  "Abre tu mano"
            sVos = ""

    elif count_defects == 2:
            iPaso = 2
            sMgs =  "dos"
            sVos = sMgs
      
    elif count_defects == 3:
            iPaso = 3
            sMgs =  "tres"
            sVos = sMgs

    elif count_defects == 4:
            iPaso = 4
            sMgs =  "cuatro"
            sVos = sMgs

    elif count_defects == 5:
            iPaso = 5
            sMgs =  "cinco"
            sVos = sMgs

    else:
        paso = [0, 0, 0, 0, 0, 0] 
    
    if paso[iPaso] > 50:
        paso[iPaso] = 0

    elif paso[iPaso] > 25:
        if ( len(sVos) > 0 ): play_pyttsx(sMgs)
        cv2.circle(img, (58, 40), 30, [0,255,0],-1) 
        cv2.putText(img, str(iPaso), (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1,  (255,255,255),2,cv2.LINE_AA) 
        cv2.putText(img, sMgs, (100, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255),2,cv2.LINE_AA)
        paso[iPaso] = paso[iPaso] + 1

    elif paso[iPaso] > 0:      
        cv2.circle(img, (58, 40), 30, [0,0,255],-1)    
        cv2.putText(img, str(iPaso), (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1,  (255,255,255),2,cv2.LINE_AA) 
        cv2.putText(img, "? chequeando...", (100, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255),2,cv2.LINE_AA)           
        paso[iPaso] = paso[iPaso] + 1

    elif paso[iPaso] == 0: 
        paso = [0, 0, 0, 0, 0, 0] 
        paso[iPaso] = 1


    # show appropriate images in windows
    cv2.imshow('Gesture', img)
    all_img = np.hstack((drawing, crop_img))
    cv2.imshow('Contours', all_img)
   
    cv2.imshow('Thresholded', thresh1)
    #cv2.imshow('Resta', resta)
    #cv2.imshow('drawing', drawing)
    #cv2.imshow('crop_img', crop_img)
 

    k = cv2.waitKey(10)
    if k == 27:
        break
    elif k == ord('b'):
        bGuardar = True