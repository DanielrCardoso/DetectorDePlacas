import cv2
import numpy as np
from matplotlib import pyplot as plt
import imutils
import easyocr

def colorir_regioes(labels):
    label_hue = np.uint8(179*labels/np.max(labels))
    blank_ch = 255*np.ones_like(label_hue)
    imgColorida = cv2.merge([label_hue, blank_ch, blank_ch])
    imgColorida = cv2.cvtColor(imgColorida, cv2.COLOR_HSV2BGR)
    imgColorida[label_hue==0] = 0
    return imgColorida

def mostrar_segmentos(numLabels,labels,stats,centroids,imageOriginal):
    final = cv2.cvtColor(imageOriginal,cv2.COLOR_GRAY2BGR)
    for i in range(0, numLabels):
        x = stats[i, cv2.CC_STAT_LEFT]
        y = stats[i, cv2.CC_STAT_TOP]
        w = stats[i, cv2.CC_STAT_WIDTH]
        h = stats[i, cv2.CC_STAT_HEIGHT]
        area = stats[i, cv2.CC_STAT_AREA]
        (cX, cY) = centroids[i]

        seg = cv2.cvtColor(imageOriginal,cv2.COLOR_GRAY2BGR)
        cv2.rectangle(seg, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.circle(seg, (int(cX), int(cY)), 4, (0,0,255), -1)

        if(x>0 and y>0):
            final = cv2.rectangle(final, (x, y), (x + w, y + h), (0, 255, 0), 2)

        segmentoMarcado = (labels == i).astype("uint8") * 255
        segmento = imageOriginal[y:y + h, x:x + w]

        cv2.imshow("Segmento Isolado",segmento)
        cv2.imshow("7.1 - Elemento Conectado", seg)
        cv2.imshow("7.2 - Elemento Conectado marcacao", segmentoMarcado)
        cv2.waitKey(0)

    show_N_imgs("8 - marcacao",[final])

def img2text(image):
    reader = easyocr.Reader(['en'])
    result = reader.readtext(image)
    text = result[0][-2]
    return text

def applyTextInImg(image):
    font = cv2.FONT_HERSHEY_PLAIN
    res = cv2.putText(image,text=text,org=(aprox[0][0][0], aprox[1][0][1]+60),fontFace=font,fontScale=1,color=(255,0,0),thickness=1,lineType=cv2.LINE_AA)
    (x,y,alt,lar) = cv2.boundingRect(aprox)
    cv2.rectangle(image,(x,y),(x+alt,y+lar),(0,0,255),2)
    return res

def get_placa(image):
    bfilter = cv2.bilateralFilter(image,11,17,17) #reducao de ruido
    bordas = cv2.Canny(bfilter,30,200) #detccao de bordas

    pontos = cv2.findContours(bordas.copy(),cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    contornos = imutils.grab_contours(pontos)
    contornos = sorted(contornos,key=cv2.contourArea,reverse=True)[:10]

    posicao = None
    marcacaoPlaca = cv2.cvtColor(image,cv2.COLOR_GRAY2BGR)
    for contour in contornos:
        aprox = cv2.approxPolyDP(contour,10,True)
        if len(aprox) == 4:
            (x,y,alt,lar) = cv2.boundingRect(aprox)
            cv2.rectangle(marcacaoPlaca,(x,y),(x+alt,y+lar),(0,255,0),2)
            placa = image[y:y + lar, x:x + alt]
            posicao = aprox
            break
    
    mask = np.zeros(image.shape,np.uint8)
    somentePlaca = cv2.drawContours(mask,[posicao],0,255,-1)
    somentePlaca = cv2.bitwise_and(image,image,mask=mask)
    
    return [bordas,marcacaoPlaca,somentePlaca],placa,aprox

def show_N_imgs(label,vetImages):
    for img in vetImages:
        cv2.imshow(label,img)
        cv2.waitKey(0)
    
def showMsgImg(text,label):
    font = cv2.FONT_HERSHEY_SIMPLEX
    img = cv2.putText(255 - np.zeros(image.shape,np.uint8),text=text,org=(round(image.shape[0]/2)-60,round(image.shape[1]/2)-60),fontFace=font,fontScale=1,color=(0,0,0),thickness=1,lineType=cv2.LINE_AA)
    cv2.imshow(label,img)
    cv2.waitKey(500)

def getOpcao():
    print("menu")

def maintain_aspect_ratio_resize(image, width=None, height=None, inter=cv2.INTER_AREA):
    dim = None
    (h, w) = image.shape[:2]
    if width is None and height is None:
        return image
    if width is None:
        r = height / float(h)
        dim = (int(w * r), height)
    else:
        r = width / float(w)
        dim = (width, int(h * r))
    return cv2.resize(image, dim, interpolation=inter)

def operacoesPlacaMercosul(image):
    print("placa mercosul")

def operacoesPlacaCinza(image):
    show_N_imgs("0 - Placa original",[image])
    imageCopy = image.copy()

    #blur
    image = cv2.GaussianBlur(image,(3,3),3)
    show_N_imgs("1 - Aplicando Gaussian Blur",[image])

    #fechamento
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(3,3))
    closing = cv2.morphologyEx(image,cv2.MORPH_CLOSE,kernel,iterations=2)
    show_N_imgs("2 - Aplicando fechamento",[closing])
    
    #adicao
    image = cv2.add(image,closing)
    show_N_imgs("3 - soma",[image])
    
    #binarizacao
    _, mask = cv2.threshold(image,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
    show_N_imgs('4 - Binarização',[mask])
    
    #erosao
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(7,7))
    erosao = cv2.erode(mask,kernel)
    show_N_imgs('5 - erosao',[erosao])

    #dilatacao
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(6,6)) 
    dilate = cv2.dilate(erosao,kernel)
    show_N_imgs('6 - Dilatação', [dilate])
    
    #extracao elementos conectados
    connectivity = 4
    (numLabels, labels, stats, centroids) = cv2.connectedComponentsWithStats(dilate , connectivity , cv2.CV_32S)
    
    mostrar_segmentos(numLabels,labels,stats,centroids,imageCopy)
    regioesColoridas = colorir_regioes(labels)
    show_N_imgs('labeled.png', [regioesColoridas])

    return regioesColoridas

##############################################

#processo de leitura
image = cv2.imread("carro4.jpg")
show_N_imgs("placa",[image])
imageGray = cv2.cvtColor(image,cv2.COLOR_RGB2GRAY)
show_N_imgs("placa",[imageGray])

#caso seja necessario extrair placa de imagem
passosOperacoes,placa,aprox = get_placa(imageGray)
show_N_imgs("placa",passosOperacoes)
show_N_imgs("somente placa",[placa])

#preparar placa para tamanho padrao
placa = maintain_aspect_ratio_resize(placa, width=300)

#caso placa cinza
placaPreparada = operacoesPlacaCinza(placa)

#caso placa mercosul
#placaPreparada = operacoesPlacaCinza(placa)

#processo de extrair texto e aplicar na imagem
showMsgImg("OCR is running...","placa")
text = img2text(placaPreparada)
print(text)
res = applyTextInImg(image)
show_N_imgs("placa",[res])
