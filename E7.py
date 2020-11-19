import numpy as np
import cv2
import matplotlib.pyplot as plt

## EJERCICIO 7 N. Jofré y A. Veas

###################################################################################################
### IMPORTANTE: Para visualizar todas las imágenes, cerrar cada una para poder ver la siguiente ###
###################################################################################################

def mostrar(I, O):
    ## Muestra en un output dos imagenes simultaneamente
    ## I: imagen 1 ó input; O: imagen 2 ó output
    plt.figure()
    plt.subplot(1, 2, 1)
    plt.imshow(I, cmap = "gray")
    plt.title("Imagen Original")
    plt.subplot(1, 2, 2)
    plt.imshow(O, cmap = "gray")
    plt.title("Imagen de salida")
    plt.show()
    
def segmentar(I, l):
    ## Permite obtener la segmentación de una imagen en escala de grises para un rango de valores entre 0 y 255
    ## I: imagen en escala de grises; l = intervalo de segmentación de tipo [v_min, v_max]
    N, M = I.shape
    J = np.zeros((N,M))
    for i in range(N):
        for j in range(M):
            if I[i,j] >= l[0] and I[i,j]<=l[-1]:
                J[i,j] = 1
    return J   

def pintar(img, seg):
    ## Genera bordes de rojo en una imagen RGB a partir de una segmentación 
    ## img: imagen input; seg: segmentación de referencia
    I = img
    kernel = cv2.getStructuringElement(cv2.MORPH_CROSS,(5,5))
    borde = cv2.dilate(seg,kernel,iterations=2) - seg
    N, M = borde.shape
    for i in range(N):
        for j in range(M):
            if borde[i,j]==1:
                I[i,j] = [255,0,0]
    return I
    
## PREGUNTA 1

atar = cv2.imread('atardecer.png')
IA = cv2.cvtColor(atar, cv2.COLOR_BGR2GRAY)
med_flt = cv2.medianBlur(IA, 11)
kernel = np.ones((5,5), np.uint8)
erosion = cv2.erode(med_flt, kernel, iterations = 8)

IA_flt = IA - erosion

Sol = segmentar(IA_flt, [150, 195])
Gaviota = segmentar(IA_flt, [215, 244])
atar_rojo = pintar(atar,Sol+Gaviota)

mostrar(cv2.imread('atardecer.png'), atar_rojo)

## PREGUNTA 2

tazm = cv2.imread('tazmania.png')
IB = cv2.cvtColor(tazm, cv2.COLOR_BGR2GRAY)
kernel_1 = np.array([-1, 0, 1], dtype = 'uint8')
Gx = cv2.morphologyEx(IB, cv2.MORPH_GRADIENT, kernel_1)
kernel_2 = np.array([-1, 0, 1], dtype = 'uint8').T
Gy = cv2.morphologyEx(IB, cv2.MORPH_GRADIENT, kernel_2)
G = np.sqrt(Gx.astype('float')**2 + Gy.astype('float')**2)

mostrar(tazm, G)


## PREGUNTA 3
gotas = cv2.imread("gotas.png")
I3 = cv2.cvtColor(gotas, cv2.COLOR_BGR2GRAY)
kernel3a = cv2.getStructuringElement(cv2.MORPH_RECT, (50,50))
kernel3b = cv2.getStructuringElement(cv2.MORPH_RECT, (11,11))
I3th = cv2.morphologyEx(I3, cv2.MORPH_TOPHAT, kernel3a)
I3s = segmentar(I3th, [65,255])
I3sf = cv2.morphologyEx(I3s, cv2.MORPH_OPEN, kernel3b)
kernel3c = cv2.getStructuringElement(cv2.MORPH_CROSS, (8,8))
dilatacion = cv2.dilate(I3sf, kernel3c)

mostrar(gotas, pintar(cv2.imread("gotas.png"),dilatacion))

