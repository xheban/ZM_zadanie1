# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import numpy as np # matematicke nastroje
import cv2 as cv # nastroje na pracu s obrazkami / installed package is "opencv" in Anaconda environment
import matplotlib.pyplot as plt # nastroje pre vystup na konzolu
from skimage import measure
from imutils import contours
import imutils
    
img = cv.imread('lenna.png',1) #1 - farebná , 0- čiernobiela
imgwork5 = img.copy()

# menu pre prácu

def menu():
    print("--------------------------------------")
    print("\n Vytvorenie farebnej kocky/obdlžníka: 1")
    print("\n Histogram jasu fotografie: 2")
    print("\n Zmena jasu a kontrastu: 3")
    print("\n Zmena jasu ručne: 4")
    print("\n Zmena jasu a kontrastu interaktívne: 5")
    print("\n Nájdenie najsvetlejších a najtmavších miest 6")
    print("\n Ukončenie aplikácie: 9")
    choose_task()

# funkciana výber úlohy

def choose_task():
    task = input("\nZvoľte si úlohu:")
    
    if task == '1' :
        task1()
    elif task == '2' :
        task2()
    elif task == '3' :
        task3()
    elif task == '4' :
        task4()
    elif task == '5' :
        task5()
    elif task == '6' :
        task6()
    elif task == '8' :
        task_test()
    elif task == '9' :
        exit()
    else:
        print("\nNeplatný vstup")
        choose_task()

#úloha 1 vytvoreniefarebného obdlznika/kocky
def task1():
    colors = [[255,0,0],[0,0,255],[0,255,0], [255,255,255],[0,0,0]]
   
    imgwork = img.copy()
    x_size = imgwork.shape[0]
    y_size = imgwork.shape[1]
    print("Veľkosť obrázku je",x_size," x ", y_size)
    x_start = input_handler(1)
    y_start = input_handler(2)
    n = input_handler(3)
    m = input_handler(4)
    color = input_handler(5)  
    end_x = x_start + n
    end_y = y_start + m
    if end_x > imgwork.shape[0]:
        end_x = imgwork.shape[0]
    if end_y > imgwork.shape[1]:
        end_y = imgwork.shape[1]
    
    for x in range (x_start,end_x):
        for y in range (y_start,end_y):
            imgwork[y,x] = colors[color]
        
    window_name = 'kocky'
    cv.imshow(window_name, imgwork)
    cv.waitKey(0)
    menu()

#úloha 2 histogram jasu
def task2():
    imgwork2 = img.copy()
    imgwork2_HSV = cv.cvtColor(imgwork2, cv.COLOR_BGR2HSV)
    histogram = cv.calcHist([imgwork2_HSV],[2],None,[256],[0,256])
    plt.plot(histogram)
    plt.xlim([0, 256])
    plt.title('Histogram jasu')
    plt.show()
    menu()

#úloha 3
def task3():
    imgwork3 = img.copy()
    
    alpha = input_handler(8)
    beta = input_handler(7)

    imgwork3_changed = cv.convertScaleAbs(imgwork3, alpha=alpha, beta=beta)
    
    window_name = 'Zmeneny_jas_a_kontrast'
    cv.imshow(window_name, imgwork3_changed)
    cv.waitKey(0)
    menu()
    
#úloha 4
def task4():
    imgwork4 = img.copy()
    imgwork4_HSV = cv.cvtColor(imgwork4, cv.COLOR_BGR2HSV)
    value_v = input_handler(6)
    for x in range(0, len(imgwork4_HSV)):
        for y in range(0, len(imgwork4_HSV[0])):
            if imgwork4_HSV[x, y][2] +  value_v > 255:
                imgwork4_HSV[x, y][2] = 255
            elif imgwork4_HSV[x, y][2] +  value_v < 0 :
                imgwork4_HSV[x, y][2] = 0
            else:
                imgwork4_HSV[x, y][2] += value_v
    

    imgwork4 = cv.cvtColor(imgwork4_HSV, cv.COLOR_HSV2BGR)
    window_name = 'Zmeneny_jas'
    cv.imshow(window_name, imgwork4)
    cv.waitKey(0)
    menu()
 
#úloha5
def task5():
    cv.namedWindow('Interaktivne_jas_kontrast',1)
    bright = 255
    contrast = 127
    #Brightness value range -255 to 255
    #Contrast value range -127 to 127
    cv.createTrackbar('bright', 'Interaktivne_jas_kontrast', bright, 2*255, funcBrightContrast)
    cv.createTrackbar('contrast', 'Interaktivne_jas_kontrast', contrast, 2*127, funcBrightContrast)
    funcBrightContrast(0)
    cv.imshow('Interaktivne_jas_kontrast', img)
    cv.waitKey(0)
    menu()
    
#ostatné funkcie pre úlohu 5
def funcBrightContrast(bright=0):
    bright = cv.getTrackbarPos('bright', 'Interaktivne_jas_kontrast')
    contrast = cv.getTrackbarPos('contrast', 'Interaktivne_jas_kontrast')
    effect = apply_brightness_contrast(imgwork5,bright,contrast)
    cv.imshow('Zmeny', effect)
def apply_brightness_contrast(input_img, brightness = 255, contrast = 127):
    brightness = map(brightness, 0, 510, -255, 255)
    contrast = map(contrast, 0, 254, -127, 127)
    if brightness != 0:
        if brightness > 0:
            shadow = brightness
            highlight = 255
        else:
            shadow = 0
            highlight = 255 + brightness
        alpha_b = (highlight - shadow)/255
        gamma_b = shadow
        buf = cv.addWeighted(input_img, alpha_b, input_img, 0, gamma_b)
    else:
        buf = input_img.copy()
    if contrast != 0:
        f = float(131 * (contrast + 127)) / (127 * (131 - contrast))
        alpha_c = f
        gamma_c = 127*(1-f)
        buf = cv.addWeighted(buf, alpha_c, buf, 0, gamma_c)
    cv.putText(buf,'B:{},C:{}'.format(brightness,contrast),(10, 30), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    return buf
def map(x, in_min, in_max, out_min, out_max):
    return int((x-in_min) * (out_max-out_min) / (in_max-in_min) + out_min)
   
#úloha 6  
def task6():
    imgToShow = img.copy()
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    imgwork6 = cv.GaussianBlur(gray, (11, 11), 0)
    thresholdB = cv.threshold(imgwork6, 200, 255, cv.THRESH_BINARY)[1]
    thresholdB = cv.erode(thresholdB, None, iterations=2)
    thresholdB = cv.dilate(thresholdB, None, iterations=4)
    labels = measure.label(thresholdB, connectivity=2, background=0)
    mask = np.zeros(thresholdB.shape, dtype="uint8")
    for label in np.unique(labels):
    	if label == 0:
    		continue

    	labelMask = np.zeros(thresholdB.shape, dtype="uint8")
    	labelMask[labels == label] = 255
    	numPixels = cv.countNonZero(labelMask)
    	if numPixels > 300:
    		mask = cv.add(mask, labelMask)


    cnts = cv.findContours(mask.copy(), cv.RETR_EXTERNAL,
    	cv.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    cnts = contours.sort_contours(cnts)[0]
    for (i, c) in enumerate(cnts):
    	(x, y, w, h) = cv.boundingRect(c)
    	((cX, cY), radius) = cv.minEnclosingCircle(c)
    	cv.circle(imgToShow, (int(cX), int(cY)), int(radius),
    		(0, 0, 255), 3)
        
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    imgwork6 = cv.GaussianBlur(gray, (11, 11), 0)
    thresholdD = cv.threshold(imgwork6, 55, 255, cv.THRESH_BINARY)[1]
    thresholdD = cv.erode(thresholdD, None, iterations=2)
    thresholdD = cv.dilate(thresholdD, None, iterations=4)
    
    for x in range(0, len(thresholdD)):
        for y in range(0, len(thresholdD)):
            if thresholdD[x][y] == 0:
                thresholdD[x][y] = 255
            else:
                thresholdD[x][y] = 0
    labels = measure.label(thresholdD, connectivity=2, background=0)
    mask = np.zeros(thresholdD.shape, dtype="uint8")
    for label in np.unique(labels):
    	if label == 0:
    		continue
    	labelMask = np.zeros(thresholdD.shape, dtype="uint8")
    	labelMask[labels == label] = 255
    	numPixels = cv.countNonZero(labelMask)
    	if numPixels > 300:
    		mask = cv.add(mask, labelMask)

    cnts = cv.findContours(mask.copy(), cv.RETR_EXTERNAL,
    	cv.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    cnts = contours.sort_contours(cnts)[0]
    
    for (i, c) in enumerate(cnts):
    	(x, y, w, h) = cv.boundingRect(c)
    	((cX, cY), radius) = cv.minEnclosingCircle(c)
    	cv.circle(imgToShow, (int(cX), int(cY)), int(radius),
    		(255, 0, 0), 3)
    
    cv.imshow("Svetle_tmave_miesta", imgToShow)
    cv.waitKey(0)
    menu()
    
#test task
def task_test():
    print("test")
   
#handling inputov
def input_handler(inp):
    if inp == 1:
        try:
            x_start = int(input("Zvoľte štartovný pixel pre širku: "))
            if x_start > img.shape[0]:
                print("Neplatný vstup")
                input_handler(inp)
            else:
                return x_start
        except:
            print("Neplatný vstup")
            input_handler(inp)
    elif inp == 2:
         try:
             y_start = int(input("Zvoľte štartovný pixel pre výšku: "))
             if y_start > img.shape[1]:
                print("Neplatný vstup")
                input_handler(inp)
             return y_start
         except:
            print("Neplatný vstup")
            input_handler(inp)
    elif inp == 3:
        try:
            n = int(input("Zvoľte počet pixelov na šírku: "))
            return n
        except:
            print("Neplatný vstup")
            input_handler(inp)
    elif inp == 4:
        try:
            m = int(input("Zvoľte počet pixelov na výšku: "))
            return m
        except:
            print("Neplatný vstup")
            input_handler(inp)
    elif inp == 5:
        try:
            color = int(input("Zvoľte farbu (0-modrá,1-červená,2-zelená,3-biela,4-čierna) :"))
            if color > 4 or color < 0:
                print("Neplatný vstup")
                input_handler(inp)
            else:
                return color
        except:
            print("Neplatný vstup")
            input_handler(inp)
            
    elif inp == 6:
        try:
            value = int(input("Zmena jasu (od -255 do 255): "))
            if value > 255 or  value < -255:
                print("Neplatný vstup")
                input_handler(inp)
            else:
                return value
        except:
            print("Neplatný vstup")
            input_handler(inp)
    elif inp == 7:
        try:
            value = float(input("Zmena jasu beta (od 0 do 100): "))
            if value > 100 or  value < 0:
                print("Neplatný vstup")
                input_handler(inp)
            else:
                return value
        except:
            print("Neplatný vstup")
            input_handler(inp)

    elif inp == 8:
        try:
            value = float(input("Zmena contrastu alpha (od 1 do 3): "))
            if value > 3 or  value < 1:
                print("Neplatný vstup")
                input_handler(inp)
            else:
                return value
        except:
            print("Neplatný vstup")
            input_handler(inp)



#START
menu()