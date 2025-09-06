import numpy as np 
import pandas as pd 
import cv2 
import matplotlib.pyplot as plt 

# Extract panels from image 

''''   Take image , turn to grayscale ( reduc colors' noise , and find rectangular contours )'''


def divide_panels ( path_img ) :

    # load image 
    img  = cv2.imread(  path_img)

    img = cv2.resize( img , ( 640 , 640   ))

    gray_img = cv2.cvtColor( img , cv2.COLOR_BGR2GRAY )

    # APply Gaussian blur on img 
    filter_img = cv2.GaussianBlur ( gray_img , (5 , 5) , 0 ) # kernel size and std dev of colors 

    # USe OStu's Binarization ( to convert img to Pure Balck & White ) - takes median value for threhold ( bw_img is second return value)
    _ , bw_img  = cv2.threshold ( filter_img , 0 , 255 , cv2.THRESH_BINARY + cv2.THRESH_OTSU )

    # find countors ( ONLY RECTANGULAR IN IMG )
    contours , heirarchy  = cv2.findContours ( bw_img , mode = cv2.RETR_TREE , method =cv2.CHAIN_APPROX_NONE)

    # ignore alrget contour , whole window
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[1:]  

    # Sort contours group ( rectnagulstrips ) inreading ordr ( y-corodinate first , x-cordiante next )

    sorted_contours = sorted(contours, key=lambda c: (cv2.boundingRect(c)[1], cv2.boundingRect(c)[0]))


    # TO EXTRACCT ONLY RECTANGUALR CONTOURS , WE USE R-D-P ALGO ( THAT APPROXIAMTES ANY ABNORMAL CONTOUR TO SMOOTHER PATH )
    # by reducng no of turns (vertices ) betwene specific poitns 

    # img cpy 
    img_dup = img.copy()

    index = 0 

    for contour in  (sorted_contours) :
            
        # find approxima vetices of detect contours
        approx_vertices = cv2.approxPolyDP ( contour , 0.02 * ( cv2.arcLength( contour , True ) ) , True )
        
        # ignore 4 sided contour formed by letters 
        area = cv2.contourArea( contour)

        if area < 500 or area>150000 :
            continue
        
        # Prevent weird shapes with 4 sides 
        x,y , w, h = cv2.boundingRect ( approx_vertices  )
        aspect_ratio = ( w/ float(h) )

        if aspect_ratio<0.3 or aspect_ratio>3.5 :
            continue 

        # check if its recnahel(vetices ==4 )
        if len( approx_vertices)==4 :

            #Rectangular contour detected 

            # crop region grom boundign ret over current contour groups 
            panel = img[ y : y+ h , x : x+ w]

            # -1 ot ONLY draw all contour poitn on given img  ( DISPALY FINAL DRAWN IMG OUTSIDE LOOp)
            cv2.drawContours ( img_dup , [approx_vertices]  , -1 , ( 0 , 255 , 0 ))

            cv2.imwrite( f'panel_{index}.jpg' , panel )
            index+=1 

    return index 
        
# cv2.imshow ( 'img' , img_dup )

if __name__ == "__main__":
    num_img = divide_panels("basic_comic.jpg")