'''
LABORATORIO 2: CONVOLUCION 2D

Estudiante: Francisco Rousseau
Ayudante: Nicole Reyes
Profesor: Carlos González

'''

######################## LIBRERIAS ########################
import numpy as np
import matplotlib.image as mpi
import matplotlib.pyplot as mpp

######################## FUNCIONES ########################

# Retorna la imagen que se encuentra en la ruta especificada (en escala de grises!!).
#
# Entrada: 
#   ruta            - string correspondiente a la ruta de la imagen
#
# Salida:
#   img             - arreglo que contiene los datos de la imagen normalizada

def leerImagen(ruta):
    img = mpi.imread(ruta)
    return img / 255 

# Guarda la imagen que se encuentra en la ruta especificada, comprueba si es filtro
# o no para aplicar el cmap gris.
#
# Entrada: 
#   img             - arreglo con los datos de la imagen
#   ruta            - string correspondiente a la ruta de la imagen
#   titulo          - string con el nombre de la imagen
#   filtro          - booleano que representa si la imagen es filtro o espectro

def guardarImagen(img, ruta, titulo, filtro):
    if filtro:
        mpp.imshow(img, cmap="gray",vmin=0, vmax=1)
    else:
        mpp.imshow(img)
    mpp.title(titulo)
    img = mpp.gcf()
    img.savefig(ruta, dpi=100)

# Muestra la imagen que se encuentra en la ruta especificada, comprueba si es filtro
# o no para aplicar el cmap gris.
#
# Entrada: 
#   img             - arreglo con los datos de la imagen
#   titulo          - string con el nombre de la imagen
#   filtro          - booleano que representa si la imagen es filtro o espectro

def visualizarImagen(img, titulo, filtro):
    if filtro:
        mpp.imshow(img, cmap="gray",vmin=0, vmax=1)
    else:
        mpp.imshow(img)
    mpp.title(titulo)
    mpp.show()

# Crea una matriz igual a la matriz imagen, pero rodeada de un contorno de ceros del tamaño necesario para aplicar la convolución.
#
# Entradas 
#   imagen          - arreglo que contiene los datos de la imagen
#   kernel          - arreglo que contiene los datos del filtro
#
# Salidas
#   matriz0  - matriz llena de ceros

def matrizCero(imagen, kernel):
    # MC -> matriz ceros
    NFilasKernel, NColuKernel = kernel.shape
    NFilasImg, NColuImg = imagen.shape
    
    PosFilaMC = NFilasKernel - 1
    PosColuMC = NColuKernel - 1 
    PosFilaImg = 0
    PosColuImg = 0

    matriz0 = np.zeros((NFilasImg + 2 * PosFilaMC, NColuImg + 2 * PosColuMC))

    for i in range(PosFilaMC, PosFilaMC + NFilasImg):
        for j in range(PosColuMC, PosColuMC + NColuImg):
            matriz0[i][j] = imagen[PosFilaImg][PosColuImg]
            PosColuImg+=1
        PosFilaImg += 1
        PosColuImg = 0
    
    return matriz0

# Aplica la convolución en 2 dimensiones entre la imagen y el kernel.
#
# Entrada:
#   imagen          - arreglo que contiene los datos de la imagen
#   kernel          - arreglo que contiene los datos del filtro
#
# Salida:
#   convolucion     - Matriz imagen filtrada

def convolucion2D(imagen, kernel):
    convolucion = []

    # Matriz e indicadores
    matriz0 = matrizCero(imagen, kernel)
    NFilasMC, NColuMC = matriz0.shape

    # Kernel e indicadores
    kernel = np.flipud(np.fliplr(kernel))
    NFilasKernel, NColuImg = kernel.shape

    # Iteradores #
    iterFilas = NFilasMC - NFilasKernel + 1
    iterColumnas = NColuMC - NColuImg + 1
    PosFilaConvolucion = 0

    for fila in range(iterFilas):
        convolucion.append([]) #se inicia la fila vacia
        
        for col in range(iterColumnas):
            suma = 0 #se reinicia la cuenta en cada columna

            for fk in range(NFilasKernel):

                for ci in range(NColuImg):
                    suma += kernel[fk][ci] * matriz0[fila + fk][col + ci]
            convolucion[PosFilaConvolucion].append(suma)

        PosFilaConvolucion += 1
    return convolucion


# Calcula la transformada de Fourier en 2D para una imagen dada.
#
# Entrada:
#   imagen          - arreglo que contiene los datos de la imagen
#
# Salida:
#   Matriz del esprectro de Fourier  - arreglo que contiene los datos de la transformada de Fourier de la imagen

def fourier(imagen):
    np.seterr(divide = 'ignore') #Configuración necesaria (recomendacion de internet)
    # Transformada de Fourier #
    tf = np.fft.fft2(imagen)

    # Shift de zero a la Transformada #
    tfs = np.fft.fftshift(tf)

    # Retorno del valor normalizado positivo #
    return 20 * np.log(np.abs(tfs))

######################## EJECUCION PROGRAMA ########################

## Creacion de filtros ##
filtroBordes = np.array([[1,2,0,-2,-1],[1,2,0,-2,-1],[1,2,0,-2,-1],[1,2,0,-2,-1],[1,2,0,-2,-1]]).astype('float') # Kernel Bordes
filtroSuavizado = np.array([[1,4,6,4,1],[4,16,24,16,4],[6,24,36,24,6],[4,16,24,16,4],[1,4,6,4,1]]).astype('float') * (1/256) # Kernel Gauss


###### Testing  ######
filtroEmbosy = np.array([[-2,-1,0],[-1,1,1],[0,1,2]]).astype('float') # Kernel Embosy para testing

print('... Inicio Testing ...')
print('* Las imagenes de testing no son almacenadas *')
test = leerImagen('img-test.gif')
visualizarImagen(test,'Imagen de Testing',True)
print('Testeando Convolucion 2D con kernel 3x3 "Ebmosy"')
testConv = convolucion2D(test, filtroEmbosy)
visualizarImagen(testConv,'Test Convolucion',True)
print('Testeando kernel Filtro Suavizado')
testFS = convolucion2D(test, filtroSuavizado)
visualizarImagen(testFS,'Test Suavizada',True)
print('Testeando kernel Filtro de Bordes')
testFB = convolucion2D(test, filtroBordes)
visualizarImagen(testFB,'Test Bordes',True)
print('... Fin Testing ...')
######################

## Lectura de imagen ##
imagen = leerImagen('leena512.bmp')


## Mapeo de imagen original ##
visualizarImagen(imagen,'Imagen Original',True)



###### Filtros  ######

## Aplicacion de filtro de suavizado gaussiano ##
imgFS = convolucion2D(imagen, filtroSuavizado)

## Aplicacion de filtro detector de bordes ##
imgFB = convolucion2D(imagen, filtroBordes)

## Mapeo de imagen filtrada suavizado ##
visualizarImagen(imgFS,'Imagen Suavizada',True)
guardarImagen(imgFS,'./Img_Suavizada.png','Imagen Suavizada',True)


## Mapeo de imagen filtrada bordes ##
visualizarImagen(imgFB,'Imagen Bordes Detectados',True)
guardarImagen(imgFB,'./Img_Bordes.png','Imagen Bordes Detectados',True)

###### Transformadas de Fourier ######

## Aplicacion transformada de fourier imagen original ##
espectroImg = fourier(imagen)

## Mapeo fourier imagen original ##
visualizarImagen(espectroImg,'Espectro Fourier Imagen Original',False)
guardarImagen(espectroImg,'./Fourier_Original.png','Espectro Fourier Imagen Original',False)

## Aplicacion transformada de fourier imagen filtrada suavizado ##
espectroImgFS = fourier(imgFS)

## Mapeo fourier imagen filtrada suavizado ##
visualizarImagen(espectroImgFS,'Espectro Fourier Imagen Suavizada',False)
guardarImagen(espectroImgFS,'./Fourier_Suavizada.png','Espectro Fourier Imagen Suavizada',False)

## Aplicacion transformada de fourier imagen filtrada bordes ##
espectroImgFB = fourier(imgFB)

## Mapeo fourier imagen filtrada bordes ##
visualizarImagen(espectroImgFB,'Espectro Fourier Imagen Bordes Detectados',False)
guardarImagen(espectroImgFB,'./Fourier_Bordes.png','Espectro Fourier Imagen Bordes Detectados',False)
