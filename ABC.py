# -*- coding: utf-8 -*-
"""
@author: Abdou
"""

import numpy as np
import matplotlib.pyplot as plt
import pywt
import pywt.data
import cv2 as cv
import skimage.measure
import random
import math
from scipy import ndimage
from skimage import exposure


# ---------------------------------------------------Fonction Traitement d'Image----------------------------------------------------------------*
# Calcule Le filtre LL d'une image et la formate
def HaarWavelet(Img):
    # Applique le Haar Wavelet
    LL, (LH, HL, HH) = pywt.dwt2(Img, 'haar')
    # On formate les intensité de l'image LL
    MatLL = FormaterHaar(LL)

    return MatLL


# Calcule l'histogramme
def MatToHist(Matrice):
    Hist = cv.calcHist([Matrice.astype('uint8')], [0], None, [256], [0, 256])

    return Hist


# Régule les intensités d'une image entre 0 et 255 (Régule une image LL)
def FormaterHaar(inputLL):
    enter = inputLL
    for i in range(len(enter)):
        for j in range(len(enter[i])):
            if enter[i][j] < 0:
                enter[i][j] = 0
            elif enter[i][j] > 255:
                enter[i][j] = 255
            else:
                enter[i][j] = int(enter[i][j])

    return enter


# Calcule le fitness d'une image après matching avec un histogramme
def FitnessABC(hist, image):
    NewImg = hist_match(image, hist)
    fitness = FitnessContrast(NewImg)
    return fitness


# Calcule une nouvelle image a partir d'un histogramme (Matching)
def hist_match(image , histoCible):
    result = image.copy()
    histoOriginale = createHistogramme(image)
    cumulOriginale = [0] * 256
    cumulCible = [0] * 256

    tableDeMapping = [0] * 256

    cumulOriginale = createCumule(histoOriginale)

    cumulCible = createCumule(histoCible)

    sommOriginal = cumulOriginale[255]

    sommCiblre = cumulCible[255]

    distibCumulOriginale = createDistribCumul(cumulOriginale, sommOriginal)
    distibCumulCible = createDistribCumul(cumulCible, sommCiblre)

    # print("cum Original", distibCumulOriginale)
    # print("cum Cible", distibCumulCible)

    for i in range(len(tableDeMapping)):
        tableDeMapping[i] = getMap(i, distibCumulOriginale, distibCumulCible)

    for i in range(len(image)):
        for j in range(len(image[0])):
            result[i][j] = tableDeMapping[image[i][j]]

    return result


def getMap(a, distribO, distribCibl):
    result = 0
    cond = True
    i = 0
    j = 255
    save = distribCibl[0]
    saveindex = 0
    while (cond == True and (i <= j)):

        # cas superieur
        if (distribO[a] > distribCibl[i]):
            i = i + 1
            # cas inferieur
        else:
            distcourent = abs(distribO[a] - distribCibl[i])
            distpasse = abs(distribO[a] - save)
            if (distpasse <= distcourent):

                result = saveindex
                cond = False
            else:
                result = i
                cond = False

        save = distribCibl[i]
        saveindex = i

    return result


def createCumule(histo):
    result = [0] * 256
    somm = 0

    for i in range(len(histo)):
        somm = somm + histo[i]
        result[i] = somm

    return result


def createDistribCumul(cumul, somm):
    result = [0] * 256

    for i in range(len(cumul)):
        result[i] = cumul[i] / somm

    return result



# Calcule La fitness d'une image(LL Formaté)
def FitnessContrast(image):
    result = 0
    log1 = math.log(filtreSobel(image), 10)
    phase1 = math.log(log1, 10)
 
    phase2 = calculerNombreDePixelsContour(image) / (len(image) * len(image[1]))

    phase3 = skimage.measure.shannon_entropy(image)

    result = phase1 * phase2 * phase3
    return result


# Applique le filtre de Sobel sur une image(LL Formaté) et retourne la somme des intensité
def filtreSobel(image):
    im = image.copy()

    im = im.astype('int32')
    dx = ndimage.sobel(im, 0)  # horizontal derivative
    dy = ndimage.sobel(im, 1)  # vertical derivative
    mag = np.hypot(dx, dy)  # magnitude
    mag *= 255.0 / np.max(mag)  # normalize (Q&D)
    return EIZsommeDesIntensite(mag)


# Calculer la somme totales des intensité d'une image
def EIZsommeDesIntensite(enter):
    somme = 0
    for i in range(len(enter)):
        for j in range(len(enter[i])):
            somme = somme + enter[i][j]

    return somme


# Calculer le nombre de pixels des contours
def calculerNombreDePixelsContour(image):
    tigreOriginal = cv.cvtColor(image.astype('uint8'), cv.COLOR_BGR2RGB)
    # plt.imshow(tigreOriginal)
    tigreGris = cv.cvtColor(tigreOriginal, cv.COLOR_RGB2GRAY)
    # plt.imshow(tigreGris)

    # creater une image binnaire avec tresh hold X Y
    #binary = cv.adaptiveThreshold(tigreGris,255,cv.ADAPTIVE_THRESH_MEAN_C,\
    #                                           cv.THRESH_BINARY_INV,11,2)
    _, binary = cv.threshold(tigreGris, 127, 255, cv.THRESH_OTSU)
    #  trouver les contour grace a l'image binnaire
    contours, hierarchy = cv.findContours(binary, cv.RETR_TREE, cv.CHAIN_APPROX_NONE)

    sumCnt=0

    for i in range(len(contours)):

        #print("##############################################################", contours[i].size)
        sumCnt= sumCnt+contours[i].size


    #print("##############################################################", contours)



    return int(sumCnt/2)

    #return (len(contours))

def createHistogramme(image):
    res = [0] * 256

    for i in range(len(image)):
        for j in range(len(image[i])):
            index = int(image[i][j])
            # print("index",  index)
            res[index] = res[index] + 1
    return res

# ---------------------------------------------------------------Fonction ABC------------------------------------------------------------------*
# 1. Start by defining the search bound then generate vectors with given predefined dimensions dim = 4
# dimension of vector pop = 5 limit = 4 # maximum number of visits to any Bee bound = [(-10, 10)]
# (lower, upper) bound for one dimensional vector bounds = [(-10, 10) for i in range(dim)]
# for a 3 dimensionsal vector print(bounds)

# 2. generate a population of random vectors (X) within the bound each with dimension D (dim) and set limit values(Trials)
# for each
# x(i,j) = l(j) + rand(0,1)*(u(i) - u(j)) ...... Eq1
# use to generate initial population of Bee Colony X = [[bounds[i][0] + ( random.uniform(0,1)*(bounds[i][1] - bounds[i][0]) ) for i in range(dim)] for i in range(pop)]
# X = [[random.uniform(bounds[i][0], bounds[i][1]) for i in range(dim)] for i in range(pop)]
# Alternative random generation Trials =[0 for i in range(pop)] X Trials

# Employee Bee
def EBee(X, f, Img, Trials):
    for i in range(len(X)):

        V = []
        R = X.copy()
        R.remove(X[i])
        r = random.choice(R)

        for j in range(len(X[0])):  # x[0] or number of dimensions

            V.append((int(abs(X[i][j] + random.uniform(-1, 1) * (X[i][j] - r[j])))))
            # idea for r, can pick a random particle for every dimendo
        if f(X[i], Img) > f(V, Img):
            Trials[i] += 1

        else:
            X[i] = V
            Trials[i] = 0

    # print(Trials)
    return X, Trials


# P(i) = ( 1 / 1 + f(x) ) / (sum(1..n) 1 / 1 + f(x(n) ) )     Pi is used to choose an onlooker bee
def P(X, f, Img):
    P = []
    sP = sum([1 / (1 + f(i, Img)) for i in X])
    for i in range(len(X)):
        P.append((1 / (1 + f(X[i], Img))) / sP)

    return P


# Onlooker Bee
def OBee(X, f, Img, Trials, phi=0.3):
    Pi = P(X, f, Img)

    for i in range(len(X)):
        # ---------------------------------------------------------------------------------*
        #              chose a bee by probability p
        # ---------------------------------------------------------------------------------*
        if random.random() < Pi[i]:

            V = []
            R = X.copy()
            R.remove(X[i])
            r = random.choice(R)
            # ----------------------------------------------------------------------------------*
            for j in range(len(X[0])):  # x[0] or number of dimensions

                V.append((int(abs(X[i][j] + random.uniform(-1, 1) * (X[i][j] - r[j])))))

            if f(X[i], Img) > f(V, Img):
                Trials[i] += 1

            else:
                X[i] = V
                Trials[i] = 0

    return X, Trials


# Scout Bee
def SBee(X, Trials, bounds, limit):
    for i in range(len(X)):

        if Trials[i] > limit:
            Trials[i] = 0

            X[i] = [int(abs(bounds[i][0] + (random.uniform(0, 1) * (bounds[i][1] - bounds[i][0])))) for i in
                    range(len(X[0]))]

    return X


# la fonction ABC
def ABC(dims, bound, f, Img, limit, pop, runs):


    bounds = [bound[0] for i in range(dims)]  # for a 3 dimensionsal vector

    X = [[int(abs(bounds[i][0] + (random.uniform(0, 1) * (bounds[i][1] - bounds[i][0])))) for i in range(dims)] for i in range(pop)]


    Trials = [0 for i in range(pop)]
    ########################

    ########################
    while runs > 0:
        X, Trials = EBee(X, f, Img, Trials)

        X, Trials = OBee(X, f, Img, Trials)

        X = SBee(X, Trials, bounds, limit)

        #print("runs " + str(61 - runs))

        fx = [f(i, Img) for i in X]
        I = fx.index(max(fx))  # find index of best position
        #print(X[I])
        #print(max(fx))
       

        ###################################################

        runs -= 1

    fx = [f(i, Img) for i in X]
    I = fx.index(max(fx))  # find index of best position
    #print(X[I])
    print(max(fx))

    return X[I]


# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # Main # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #############################################################

from google.colab import drive
drive.mount('/content/gdrive')

dimensions = 256
bound = [(-100,100)]

Input = "/content/gdrive/My Drive/Data Final/Gray/Melanoma/AUG_0_"
Output = "/content/gdrive/My Drive/Data Final/ABC/Melanoma/AUG_0_"

for i in range(378,447):
    InputName = Input + str(i) + ".jpeg"
    OutputName = Output + str(i) + ".jpeg"

    image = cv.imread(InputName,0)
    if image is None:
        print("image non disponible "+ str(i))
    else:
        print("Start "+ str(i))
        BestHist = ABC(dimensions, bound, FitnessABC, image, limit=100, pop=10, runs=80)
        print("End")
        #print("Meilleur Histogramme:" + str(BestHist))
        BestImg = hist_match(image, BestHist)
        cv.imwrite(OutputName, BestImg)


        
print("Fin")