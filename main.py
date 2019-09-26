# -*- coding: utf-8 -*-
"""
Created on Sun Dec 23 11:37:38 2018

@author: OSCAR
"""
  
import numpy as np
import cv2
from matplotlib import pyplot as plt
import math as mth
from scipy.spatial import distance as dist
from os import walk
from time import time

#Constantes para definir la bandera DE GRIS y COLOR
GRIS = 0
COLOR = 1
CVALUE = 10000
MILLION = 1000001

#DEFINICIÓN DE FUNCIONES DE UTILIDAD
# Definimos la fucnión para leer la imagen en
# color o escala de grises indicados como parámetros.
def leeimagen(filename,flagColor):
    img = cv2.imread(filename,flagColor)
    return img

#Definimos una función para mostrar una imagen en una ventana
#indicando el título de la misma por parámetro.
def pintaim(img,titulo_Ventana = "Titulo de Ventana por Defecto"):
    cv2.namedWindow(titulo_Ventana, cv2.WINDOW_NORMAL)
    cv2.imshow(titulo_Ventana,img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


#Para poder mostrar varias imágenes en una única ventana hemos
#definido dos funciones distintas, ya que con la primera forma
#de implementación se muestran en el terminal de spyder, y con
#la segunda forma podemos mostrarla en una ventana aparte y podremos
#observar las diferencias de los resultados obtenidos con mayor
#facilidad.
    
#Con el primer método hacemos uso de las funciones de matplotlib,
#indicando por parámtros los títulos de cada una de las imágenes,
#el de la ventana, y la distribución en filas y columnas.
def plotImagenes(vim,titulos,titulo_Ventana,fil,col):
    #Indicamos el título de la ventana
    plt.figure(titulo_Ventana)
    i = 1
    for im in vim:
        plt.subplot(fil,col,i)
        #Si la imágen es en color, tendremos un vector de 3 
        #características por lo que trataremos la imagen
        #para cada caso respectivo
        if len(np.shape(im)) == 3:
            #Debemos trabajar con BGR por lo que hacemos la conversión
            im = cv2.cvtColor(np.array(im), cv2.COLOR_RGB2BGR)
            plt.imshow(im)
        else:
            plt.imshow(im,'gray')
        #Ponemos cada uno de los títulos a cada imágen
        plt.title(titulos[i-1])
        i = i+1
    plt.show()
    
#Definimos una función para mostrar varias imágenes concatenadas
#pudiendo ser de diferentes tamaños, indicando el título de la
#ventana y el eje en el que queremos mostrar la concatenación
#de las imágenes.
def mostrarImagenes(vim,titulo_Ventana="Titulo de Ventana por Defecto", eje=1):
    tam = vim[0].shape[eje]
    
    for i in range(len(vim)):
        #Para cada imágen, completamos con ceros en 
        #el eje que hemos indicado para hacer las 
        #imágenes del mismo tamaño.
        if len(vim[i]) < tam:
            while vim[i].shape[eje] < tam:
                vim[i] = np.insert(vim[i], 0, 0, eje)
    #Concatenamos las imágenes ya igualadas al mismo tamaño.
    vim = np.concatenate(vim, (eje+1)%2)    
    #Llamamos a la función arriba definida para mostrarlas
    #en una única ventana.
    pintaim(vim,titulo_Ventana)
    
    
#Función para leer las imagenes de la ruta dada
def readImgs(pathroot):
    print("Iniciando lectura de imagenes...")
    imgs=[]
    for (path, ficheros, archivos) in walk (pathroot):
        for img in archivos:
            imgs.append(cv2.imread(pathroot+str(img),COLOR))
    print("Lectura de imágenes finalizada.")
    return imgs
  
#Función para obtener las correspondencias dados los dos descriptores
#con crossCheck por defecto empleando fuerza bruta.
def getMatchesFB(d1,d2,cross=True):
    bf = cv2.BFMatcher(crossCheck=cross)
    #Hallamos las correspondencias con los descriptores de ambas imágenes halladas
    return bf.match(d1,d2)

#Función para obtener las correspondencias dados los dos descriptores
#empleando KNN = 2 por defecto, devolviendo aquellas cuya distancia es menor
def getMatchesNN(d1,d2,kNN = 2, LA = 0.7):
    kbf = cv2.BFMatcher()
    #Hallamos las correspondencias con los descriptores de ambas imágenes halladas
    matches = kbf.knnMatch(d1,d2,k=kNN)
    return LAverage(matches)

#Función para seleccionar los mejores matches atendiendo a un threshold
def LAverage(matches, LA = 0.7):
    best_matches = []
    for m in matches:
        if m[0].distance < m[1].distance * LA:
            best_matches.append(m)
    return best_matches

#Función para crear una lista vacia de MILLION filas
def getMatrix():
    iv = []
    for i in range(MILLION):
        iv.append([])
        
    return iv


#Función para realizar la consulta de una imagen empleada como query sobre
#el resto de las imágenes.
def queryImage(modelH, imgQuery, dictionary, mode=2, bword=0):
    #Calculamos el histograma de la imagen query
    print("Iniciando query...")
    histogramq = (getHistogram(imgQuery, dictionary,[],mode, bword))[0]
    similarity = []
    #Hallamos la similitud empleando la función Sim que implementa
    #la fórmula dada en las diapositivas de teoría para ésta
    for imgh in range(len(modelH)):
        #Los valores dados se encuentran normalizados entre 0 y 1, donde 1 vendrá
        #dado para la query consigo misma
        s = Sim(histogramq,modelH[imgh])
        pair = [s,imgh]
        #Añadimos el valor de similaridad junto con el índice de la imagen
        similarity.append(pair) 
    #Ordenamos en orden decreciente de similaridad para obtener las que más
    #se asemejen a la dada.
    similarity_ordered = sorted(similarity,key=lambda x: x[0])
    similarity_ordered.reverse()
    similarity=[]
    print("Query terminada.")
    return similarity_ordered, similarity

#Función para otener la bolsa de palabras para una imagen dada
#a partir del vocabulario dado.
def getBagWord(img,dictionary,descriptors=[],desc=False, mode = 2):
    des = []
    #Si ya tenemos los descriptores calculados nos saltamos su cálculo
    if len(descriptors)>0:
        des = descriptors
    else:
        #Calculamos los descriptores de la imagen dada como parámetro para
        #la obtención de la bolsa de palabras
        sift = cv2.xfeatures2d.SIFT_create()
        kp, des = sift.detectAndCompute(img,mask=None)
    
    #La forma indicada mediante el cálculo de la distancia euclidea resulta
    #muy lenta, por lo que otra forma que intenté fue obtener las correspondencias más
    #cercanas de mis descriptores con las palabras del diccionario, tal y como 
    #lo hemos hehco cuando hemos tratado de obtener las correspondencias entre
    #dos imágenes, ya que en teoría es eso lo que intentamos obtener
    bwords = []
    #Modo rápido mediante el método de matching por fuerza bruta
    if mode == 1:
        bf = cv2.BFMatcher(cv2.NORM_L2)
        #Hallamos las correspondencias con los descriptores de ambas imágenes halladas
        matches = bf.match(des,dictionary)
        for match in matches:
            bwords.append(match.trainIdx)
    else:
        #Modo más lento que calcula la distancia euclidea del descriptor con cada
        #una de las palabras del vocabulario y se queda con la más cercana para ese
        #descriptor
        word = dictionary[0]
        #Para cada uno de los descriptores de la imagen:
        for d in des:
            dst = CVALUE
            #Nos quedamos con aquella palabra que más cercana a nuestro descriptor
            #se encuentre empleando la norma euclídea como métrica y añadimos dicha
            #palabra más cercana del vocabulario a la bolsa de palabras para esa 
            #imagen
            for w in range(len(dictionary)):
                dst_w = dist.euclidean(d,dictionary[w])
                if (dst_w < dst):
                    word = w
                    dst = dst_w
            bwords.append(word)
    #Devolvemos la bolsa de palabras calculada
    return bwords

#Función para obtención de descriptores de un vector imagenes dadas
def getDescriptors(imgs):
    print("Calculando descriptores...")
    sift = cv2.xfeatures2d.SIFT_create()
    descriptors = []
    #Calculamos descriptores de cada una de las imagenes dadas por parámetro
    for img in imgs:
        kp, des  = sift.detectAndCompute(img,mask=None)
        descriptors.append(des)
    
    print("Fin del cálculo de descriptores.")
    #Devolvemos el conjunto de descriptores calculados para cada imagen
    return descriptors
    

#Función para obtener el histograma de una imagen
def getHistogram(img, dictionary, des=[], mode = 2, bwords = 0):
    #Obtenemos la bolsa de palabras de dicha imagen
    if (bwords == 1):
        words = getBagWord(img,dictionary,des,False, mode)
    else:
        words = dictionary
        
    histogram = np.zeros(MILLION)
    #Incrementamos en una unidad el valor de la palabra dada por el indice w
    #que indica el indice de la palabra en el vocabulario dado
    for w in words:
        histogram[int(w)] = histogram[int(w)] + 1
    #Devolvemos el histograma y la bolsa de palabras para la imagen dada
    return histogram, words

#Función para realizar el cálculo de la similitud
def Sim(h1, h2):
    s = (np.sum(np.dot(h1,h2)) / (mth.sqrt(np.sum(np.power(h1,2))) * mth.sqrt(np.sum(np.power(h2,2)))))
    
    return s

#Función para calcular el histograma de cada una de las imágenes indicadas en el 
#parámetro imgs
def getHistDictionary(imgs, dictionary, mode = 2, bwords = 0):
    hist = []
    print("Inicio de calculo de histogramas...")
    for img in range(len(imgs)):
        #Llamamos a la función arriba definida para obtener el histograma de dicha
        #imagen y lo añadimos al vector de histpgramas calculados
        if(bwords == 0):
            h = (getHistogram(imgs[img],dictionary[img],[], mode))[0]
        else:
            #1: para indicar bf.match
            #1: para indicar que calcule la bolsa de palaras
            h = (getHistogram(imgs[img],dictionary,[],mode,1))[0]
        hist.append(h)
    print("Histogramas calculados.")
    #Devolvemos el vector de histogramas de las imagenes
    return hist

#Función para realizar la lectura de palabras asociadas a cada imagen dada una
#ruta.
def readWords(pathW):
    print("Inicio de lectura de palabras de imagenes...")
    words=[]
    #Recorremos cada uno de los ficheros
    for (path, ficheros, archivos) in walk (pathW):
        for img in archivos:
            words_img = []
            f = open(pathW+str(img))
            #Atendiendo a la estructura del fichero, nos saltamos las dos
            #primeras líneas
            f.readline()
            f.readline()
            for line in f:
                #Parseamos cada línea y nos quedamos con la primera palabra,
                #correspondiente al índice de la palabra dentro del 
                #vocabulario de 1M de palabras
                words_img.append((line.split())[0])
            #Cerramos el fichero
            f.close()
            #Cada índice se corresponde con las palabras de la imagen 
            #correspondiente a dicho índice
            words.append(words_img)
    print("Fin de lectura de palabras de imagenes.")
    return words

def getDictionary(descriptors, K=300):
    print("Inicio de clustering...")
    descriptors = np.asarray(descriptors)
    descriptors = np.concatenate(descriptors,axis=0)  
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    centers = (cv2.kmeans(descriptors,K,None,criteria,10,cv2.KMEANS_RANDOM_CENTERS))[2]
    print("Fin de clustering.")
    return centers


#Función para ejecutar una consulta de recuperación de una instancia sobre
#un modelo de índice invertido
def queryII(iv, qw, num_imgs):
    print("Iniciando query...")
    #Inicializamos a cero el contador de ocurrencias para cada imagen
    ocurrencias = np.zeros(num_imgs)
    #Para cada palabra de la instancia a recuperar
    for w in qw:
        #Obtenemos las instancias asociadas a esa palabra
        imgs = iv[int(w)]
        #Para cada instancia asociada a la palabra
        for img in imgs:
            #Incrementamos el contador de ocurrencias de esa imagen para 
            #la palabra que se esta tratando
            ocurrencias[img] = ocurrencias[img] + 1
    
    #Creamos un vector de pares, donde contendremos los índices de las imágenes
    #asociados a las ocurrencias contabiliazadas en el paso anterior
    instancias = []   
    for i in range(num_imgs):  
        pair=[ocurrencias[i],i]
        instancias.append(pair)
    
    #Ordenamos en orden decreciente de ocurrencias
    instancias_ordered = sorted(instancias,key=lambda x: x[0])
    instancias_ordered.reverse()
    print("Query terminada.")
    #Devolvemos vector de instancias ordenado
    return instancias_ordered



#Función para la construcción del modelo de índice invertido con bolsa de 
#palabras.
def getInvertedIndex(imgs, words):
    #Construimos el modelo donde cada uno de los índices del vector se corresponde
    #con el índice de la palabra del diccionario y para cada una de las palabras
    #calculadas para cada imagen introducimos el índice de dicha imagen en el 
    #modelo
    print("Iniciando construcción de índice invertido...")
    iv = getMatrix()
    for img in range(len(imgs)):
        img_words = words[img]
        for w in img_words:
            iv[int(w)].append(img)
    print("Indice invertido finalizado.")
    #Devolvemos el modelo calculado
    return iv


#Función definida para la ejecución de un experimento, donde podremos indicar
#el tamaño del vocabulario a generar (k), el índice de la imagen a emplear como
#consulta (idxquery), y el número de instancias a recuperar (n).
#Los dos primeros argumentos se corresponden con el conjunto de imágenes y
#descriptores asociados a cada una.
#Concretamente este experimento tiene como objetivo comprobar el tiempo de 
#obtención del vocabulario, y la influencia del número de palabras en la
#obtención de buenos resultados a la hora de recuperar las instancias, a la vez
#que medimos el tiempo de recuperación.
def experimento1(imgs, des, k = 100, idxquery = 34, n=10):
    print("\nEXPERIMENTO TIPO 1: ")
    print("Número de imágenes: "+str(len(imgs)))
    #Iniciamos la toma de tiempo para la creación del vocabulario
    t_init = time()
    #Obtenemos el vocabulario a partir de los descriptores asociados a las
    #imágenes
    dictionary = getDictionary(des,k)
    #Volvemos a obtener el tiempo e imprimimos la diferencia respecto al tiempo
    #inicial para obtener el tiempo de ejecución
    t_end = time()
    print("Tiempo de construccion del vocabulario: "+ str(t_end-t_init))
    #Obtenemos el histograma para cada una de las imagenes con el 
    #vocabulario calculado
    histograms = getHistDictionary(imgs,dictionary,1,1)
    #Seleccionamos una imagen de consulta para la recuperación de instancias
    img_query = imgs[idxquery]
    #Realizamos el mismo proceso que antes para tomar el tiempo de la query
    t_init = time()
    #Obtenemos un vector con las instancias recuperadas
    q = (queryImage(histograms,img_query,dictionary,1,1))[0]
    t_end = time()
    print("Tiempo de query: "+ str(t_end-t_init))
    #Imprimimos las n 'mejores' recuperadas (con respecto a la similitud de 
    #los histogramas)
    #Nos aseguramos que no salimos del vector
    if(n>len(q)): n = len(q)
    for i in range(n):
        pintaim(imgs[q[i][1]])
        

#Función definida para la ejecución de un experimento, donde podremos indicar
#el path donde se encuentre las palabras, el índice de la imagen a emplear como
#consulta (idxquery), y el número de instancias a recuperar (n).
#El primer argumento se corresponde con el conjunto de imagenes sobre las que
#ejecutar la consulta.
#Con este experimento comprobaremos como los resultados mejoran notablemente al
#emplear el vocabulario de 1M de palabras proporcionado a la vez
#que medimos el tiempo de recuperación.
def experimento2(imgs, idxquery = 34, pathW = 'words/exp2/', n = 10):
    print("\nEXPERIMENTO TIPO 2: ")
    print("Número de imágenes: "+str(len(imgs)))
    #Lectura de las palabras asociadas a cada imagen (bolsa de palabras)
    wordsImgs = readWords(pathW)
    #Obtención de los histogramas asociados a cada imagen
    h = getHistDictionary(imgs,wordsImgs)
    #Seleccionamos una imagen de consulta
    w = wordsImgs[idxquery]
    wordsImgs = []
    #Realizamos el mismo proceso que antes para tomar el tiempo de la query
    t_init = time()
    #Ejecutamos la query sobre el conjunto de imágenes dado
    q = (queryImage(h,imgs[idxquery],w))[0]
    t_end = time()
    print("Tiempo de query: "+ str(t_end-t_init))
    #Nos aseguramos que no salimos del vector
    if(n>len(q)): n = len(q)
    #Mostramos las n 'mejores' instancias recuperadas
    for i in range(n):
        pintaim(imgs[q[i][1]])
        

#Función definida para la ejecución de un experimento, donde podremos indicar
#el path donde se encuentre las palabras, el índice de la imagen a emplear como
#consulta (idxquery), y el número de instancias a recuperar (n).
#El primer argumento se corresponde con el conjunto de imagenes sobre las que
#ejecutar la consulta.     
#Con este experimento pretendemos comprobar si los resultados
#obtenidos para un modelo de índice invertido con bolsa de 
#palabras en el proceso de recuperación mejora respecto de los anteriores.
def experimento3(imgs, idxquery = 34, pathW = 'words/exp2/', n = 5):
    print("\nEXPERIMENTO TIPO 3: ")
    print("Número de imágenes: "+str(len(imgs)))
    #Lectura de las palabras asociadas a cada imagen (bolsa de palabras)
    wordsImgs = readWords(pathW)
    #Creamos el índice invertido
    t_init = time()
    iv = getInvertedIndex(imgs,wordsImgs)
    t_end = time()
    print("Tiempo de construcción de indice invertido: "+ str(t_end-t_init))
    #Seleccionamos una imagen de consulta
    w = wordsImgs[idxquery]
    wordsImgs = []
    #Realizamos el mismo proceso que antes para tomar el tiempo de la query
    t_init = time()
    #Ejecutamos la query sobre el conjunto de imágenes dado
    q = queryII(iv,w, len(imgs))
    t_end = time()
    print("Tiempo de query: "+ str(t_end-t_init))
    #Nos aseguramos que no salimos del vector
    if(n>len(q)): n = len(q)
    #Mostramos las n 'mejores' instancias recuperadas
    for i in range(n):
        print(q[i])
        pintaim(imgs[q[i][1]],"Query 34 - Indice invertido y bolsa de palabras")



#Definición del programa principal para la llamada a 
#las funciones implementadas y obtención de resultados.
def main():
    #ELIMINAR ASHMOLEAN_214
    
    #Procedemos a la lectura de imagenes 
    imgs1 = readImgs('img_experimentos/exp1/')
    imgs2 = readImgs('img_experimentos/exp2/')
    
    #Realizamos el cálculo de los descriptores SIFT fuera del tipo de 
    #experimento tipo 1 para ahorrar el cálculo innecesario cada vez que la
    #función sea llamada para definir un nuevo experimento
    des = getDescriptors(imgs1)
    
    #EXPERIMENTO tipo 1
    #Tamaño de vocabulario por defeto => 100 palabras
    #Imagen de consulta por defecto => indice 34
    #Numero de instancias recuperadas por defecto => 10 instancias
    experimento1(imgs1,des)
    
    #EXPERIMENTO tipo 1
    #Tamaño de vocabulario => 1000 palabras
    #Imagen de consulta por defecto => indice 34
    #Numero de instancias recuperadas por defecto => 10 instancias
    experimento1(imgs1,des,1000) 
    
    #EXPERIMENTO tipo 2
    #Path de palabras  => 'words/exp1/'
    #Imagen de consulta por defecto => indice 34
    #Numero de instancias recuperadas por defecto => 10 instancias
    experimento2(imgs1,34,'words/exp1/') 
    
    #EXPERIMENTO tipo 2
    #Path de palabras por defecto => 'words/exp2/'
    #Imagen de consulta por defecto => indice 34
    #Numero de instancias recuperadas por defecto => 10 instancias
    #Esta vez el número de imágenes es superior que en los anteriores casos
    experimento2(imgs2) 
    
    #EXPERIMENTO tipo 3
    #Path de palabras por defecto => 'words/exp2/'
    #Imagen de consulta por defecto => indice 34
    #Numero de instancias recuperadas por defecto => 10 instancias
    #Mismo número de imágenes que el caso anterior
    experimento3(imgs2)
    
   
#Llamada al Programa Principal
if __name__ == "__main__":
    main()
