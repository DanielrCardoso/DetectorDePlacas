import math
import cv2
from cv2 import THRESH_OTSU
import numpy as np

def imshow_components(labels):
    label_hue = np.uint8(179*labels/np.max(labels))
    blank_ch = 255*np.ones_like(label_hue)
    labeled_img = cv2.merge([label_hue, blank_ch, blank_ch])

    labeled_img = cv2.cvtColor(labeled_img, cv2.COLOR_HSV2BGR)
    labeled_img[label_hue==0] = 0

    cv2.imshow('labeled.png', labeled_img)

def imshow_segments(numLabels,labels,stats,centroids,imageOriginal):
    final = cv2.cvtColor(imageOriginal,cv2.COLOR_GRAY2BGR)
    for i in range(0, numLabels):
        x = stats[i, cv2.CC_STAT_LEFT]
        y = stats[i, cv2.CC_STAT_TOP]
        w = stats[i, cv2.CC_STAT_WIDTH]
        h = stats[i, cv2.CC_STAT_HEIGHT]
        area = stats[i, cv2.CC_STAT_AREA]
        (cX, cY) = centroids[i]

        output = cv2.cvtColor(imageOriginal,cv2.COLOR_GRAY2BGR)

        cv2.rectangle(output, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.circle(output, (int(cX), int(cY)), 4, (0,0,255), -1)

        if(x>0 and y>0):
            final = cv2.rectangle(final, (x, y), (x + w, y + h), (0, 255, 0), 1)

        componentMask = (labels == i).astype("uint8") * 255

        cv2.imshow("6 - Elemento Conectado", output)
        cv2.imshow("6 - Elemento Conectado marcação", componentMask)
        cv2.waitKey(0)
    cv2.imshow("7 - marcacao",final)
    cv2.waitKey(0)

def get_placa(imageori,ori):
    imageori2 = imageori.copy()
    imageori2 = cv2.cvtColor(imageori2,cv2.COLOR_GRAY2BGR)

    ori = cv2.cvtColor(ori,cv2.COLOR_GRAY2BGR)
    contornos,hierarchy= cv2.findContours(imageori,cv2.RETR_TREE,cv2.CHAIN_APPROX_NONE)

    # print(contornos)
    for c in contornos:
        # print(c)
        perimetro = cv2.arcLength(c,True)
        # print(perimetro)
        # if ()
        if perimetro > 150:
            aprox = cv2.approxPolyDP(c,0.03 * perimetro,True)
            if len(aprox) == 4:
                (x,y,alt,lar) = cv2.boundingRect(c)
                cv2.rectangle(imageori2,(x,y),(x+alt,y+lar),(0,255,0),2)
                cv2.rectangle(ori,(x,y),(x+alt,y+lar),(0,255,0),2)
                roi = ori[y:y + lar, x:x + alt]
                # cv2.imwrite("roi.jpg",roi)
    cv2.imshow("roi",roi)
    cv2.imshow("fima",ori)
    cv2.imshow("fim",imageori2)
    cv2.waitKey()
    return roi

def opsPlacaCinza():
    print("placa cinza")

def opsPlacaMercosul():
    print("placa mercosul")

def getOpcao():
    print("menu")
    
def lerPlaca(image):
    cv2.imshow("0 - Placa original",image)
    image2 = image.copy()

    #blur
    image = cv2.GaussianBlur(image,(3,3),3)

    #fechamento
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(3,3))
    closing = cv2.morphologyEx(image,cv2.MORPH_CLOSE,kernel,iterations=2)
    # cv2.imshow("1 - Aplicando fechamento",closing)
    
    #adicao
    image = cv2.add(image,closing)
    # cv2.imshow("2 - soma",image)
    
    #binarizacao
    _, mask = cv2.threshold(image,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
    # cv2.imshow('3 - Binarização',mask)
    
    #erosao
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(7,7))
    erosao = cv2.erode(mask,kernel)
    # cv2.imshow('4 - erosao',erosao)
    roi = get_placa(erosao,image2)
    #dilatacao
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(6,6)) 
    dilate = cv2.dilate(erosao,kernel)
    # cv2.imshow('5 - Dilatação', dilate)
    
    #extracao elementos conectados
    connectivity = 4
    (numLabels, labels, stats, centroids) = cv2.connectedComponentsWithStats(dilate , connectivity , cv2.CV_32S)
    # imshow_components(labels)
    # imshow_segments(numLabels,labels,stats,centroids,image2)
    cv2.waitKey(0)

image = cv2.imread("carro4.jpg",0)
lerPlaca(image)