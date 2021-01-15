# -*- coding: utf-8 -*-
"""
Created on Fri Oct 23 11:00:44 2020

@author: AHA
"""

import numpy as np
import matplotlib.pyplot as plt

import pywt
import pywt.data

import cv2 as cv
import random

from skimage.morphology import disk
import skimage.measure    

from skimage.exposure import match_histograms
import numpy
import scipy
from scipy import ndimage
import scipy.misc

from skimage.filters.rank import entropy

import math 

import random
from skimage import exposure

import os

#################################### FULL GEN DEBUT ####################################

def spead(histo):
    result = 0
    
    return result


def quickHistoCumulatif(histo):

    result = [0]*256
    tresult = [0]*256
    somm = 0
    
    for i in range(len(histo)):
        somm = somm + histo[i]
        result[i] = somm
        
    for i in range(len(result)):
        
        tresult[i] = result[i] / somm
    
    
    
    
    return tresult
###########


def getQuantile(a,distribCibl):
    
    result = 0
    cond = True
    i=0
    j=255
    save = distribCibl[0]
    saveindex = 0
    while(cond == True and (i <=j) ):
        
        #cas superieur
        if(a > distribCibl[i]):
            i = i +1
            #cas inferieur
        else:
            distcourent = abs(a - distribCibl[i])
            distpasse = abs(a - save)
            if(distpasse <= distcourent):
                
                
                result = saveindex
                cond = False
            else:
                result = i
                cond = False

        save = distribCibl[i]
        saveindex = i
            
        
    
    
    return result





##################

def calcInterQuartilDist(histo):
    
    result = 0
    # binnedhisto = []    
    # cpt =0
    # for i in range(len(histo)):
    #     if(histo[i]>0):
    #         binnedhisto.append(i)
    #         cpt = cpt +1
        
    histoCumul = createCumule(histo)
    
    
    createDistribCumulX = createDistribCumul(histoCumul,histoCumul[255])
    
    q1 = getQuantile(0.25,createDistribCumulX)
    q3 = getQuantile(0.75,createDistribCumulX)

    dist = q3 - q1
    print("dist ", dist)
    
    #get first
    i =0 
    first = -1
    last = -1
    while (i < 255 and first ==-1 ):
        if histo[i]>0:
            first = i
        i = i +1
           
    
    
    #get last
    
    j = 255
    while (j > 0 and last ==-1 ):
        if histo[j]>0:
            last = j
        j = j  - 1
           
    
    result = dist/(last-first)
    
    

    
    
    
    
    return result



def createCumule(histo):
    result = [0]*256
    somm = 0
    
    for i in range(len(histo)):
        somm = somm + histo[i]
        result[i] = somm
    
    
    return result
    
    
def createDistribCumul(cumul, somm):
    result = [0]*256
    
    for i in range(len(cumul)):
        
        result[i] = cumul[i] / somm
    
    
    return result
     



    
    
def getMap(a,distribO,distribCibl):
    
    result = 0
    cond = True
    i=0
    j=255
    save = distribCibl[0]
    saveindex = 0
    while(cond == True and (i <=j) ):
        
        #cas superieur
        if(distribO[a] > distribCibl[i]):
            i = i +1
            #cas inferieur
        else:
            distcourent = abs(distribO[a] - distribCibl[i])
            distpasse = abs(distribO[a] - save)
            if(distpasse <= distcourent):
                
                
                result = saveindex
                cond = False
            else:
                result = i
                cond = False

        save = distribCibl[i]
        saveindex = i
            
        
    
    
    return result



def histoMatchV3(image , histoCible):
    result = image.copy()
    histoOriginale = createHistogramme(image)
    cumulOriginale = [0]*256
    cumulCible = [0]*256
    
    tableDeMapping = [0]*256
    
    cumulOriginale = createCumule(histoOriginale)
    
    cumulCible = createCumule(histoCible)
    
    
    
    
    sommOriginal = cumulOriginale[255]

    sommCiblre = cumulCible[255]

    
    
    distibCumulOriginale = createDistribCumul(cumulOriginale, sommOriginal)
    distibCumulCible = createDistribCumul(cumulCible, sommCiblre)
    
    # print("cum Original", distibCumulOriginale)
    # print("cum Cible", distibCumulCible)
    
    for i in range(len(tableDeMapping)):
        tableDeMapping[i] = getMap(i,distibCumulOriginale,distibCumulCible)
    
   
    
    for i in range(len(image)):
        for j in range(len(image[0])):
            result[i][j]= tableDeMapping[image[i][j]]
    
    return result

def newContourToZero(image):
    image = cv.cvtColor(image.astype('uint8'), cv.COLOR_BGR2RGB)
    tigreGris = cv.cvtColor(image, cv.COLOR_RGB2GRAY)
    
    im = tigreGris
    imgray = tigreGris
    _, binary = cv.threshold(imgray, 127, 255, cv.THRESH_TOZERO_INV)
    contours, hierarchy = cv.findContours(binary, cv.RETR_TREE, cv.CHAIN_APPROX_NONE)
    sumCnt=0
    for i in range(len(contours)):
    
        sumCnt= sumCnt+contours[i].size
    return sumCnt/2



def newContourOTSU(image):
    image = cv.cvtColor(image.astype('uint8'), cv.COLOR_BGR2RGB)
    tigreGris = cv.cvtColor(image, cv.COLOR_RGB2GRAY)
    
    im = tigreGris
    imgray = tigreGris
    _, binary = cv.threshold(imgray, 127, 255, cv.THRESH_OTSU)
    contours, hierarchy = cv.findContours(binary, cv.RETR_TREE, cv.CHAIN_APPROX_NONE)
    sumCnt=0
    for i in range(len(contours)):
    
        sumCnt= sumCnt+contours[i].size
    return sumCnt/2
    
def newContourBinnaire(image):
    image = cv.cvtColor(image.astype('uint8'), cv.COLOR_BGR2RGB)
    tigreGris = cv.cvtColor(image, cv.COLOR_RGB2GRAY)
    im = tigreGris
    imgray = tigreGris
    _, binary = cv.threshold(imgray, 127, 255, cv.THRESH_BINARY_INV)
    contours, hierarchy = cv.findContours(binary, cv.RETR_TREE, cv.CHAIN_APPROX_NONE)
    sumCnt=0
    for i in range(len(contours)):
    
        sumCnt= sumCnt+contours[i].size
    return sumCnt/2

def newContourTriangle(image):
    image = cv.cvtColor(image.astype('uint8'), cv.COLOR_BGR2RGB)
    tigreGris = cv.cvtColor(image, cv.COLOR_RGB2GRAY)
    im = tigreGris
    imgray = tigreGris
    _, binary = cv.threshold(imgray, 127, 255, cv.THRESH_TRIANGLE)
    contours, hierarchy = cv.findContours(binary, cv.RETR_TREE, cv.CHAIN_APPROX_NONE)
    sumCnt=0
    for i in range(len(contours)):
    
        sumCnt= sumCnt+contours[i].size
    return sumCnt/2
        
    



def nbrPixelsOtsu(image):
    Ot, binaryOtsu = cv.threshold(image,127,255, cv.THRESH_OTSU)
    

    
    
def getImageMin(image):
    best = image[0][0]
    
    for i in range(len(image)):
        for j in range(len(image[1])):
            
            if(image[i][j]<best):
                best = image[i][j]
            
            
    
    
    return best

def getImageMax(image):
    best = image[0][0]
    for i in range(len(image)):
        for j in range(len(image[1])):
            
            if(image[i][j]>best):
                best = image[i][j]

    return best    
    
def normalizeRed(intensity, min,max):

    iI      = intensity

   

    minI    = min

    maxI    = max

 

    minO    = 0

    maxO    = 255

 

    iO      = (iI-minI)*(((maxO-minO)/(maxI-minI))+minO)

    return iO    
    

def normalizeFullImage(image):
    
    result = image.copy()
    min = getImageMin(image)
    print("min ", min)
    max = getImageMax(image)       
    print("max ", max)
    for i in range(len(image)):
            for j in range(len(image[1])):
                norm = normalizeRed(image[i][j], min,max)
                print( image[i][j], " to ", norm)
                result[i][j] = norm
                
    
    
    return result
    




############


    tigreOriginal = cv.cvtColor(image.astype('uint8') , cv.COLOR_BGR2RGB)
    tigreGris = cv.cvtColor(tigreOriginal, cv.COLOR_RGB2GRAY)

    # creater une image binnaire avec tresh hold X Y
    _, binary = cv.threshold(tigreGris,127,255, cv.THRESH_BINARY_INV)

    newImage = tigreGris.copy()
    for i in range(len(newImage)):
        for j in range(len(newImage[0])):
            newImage[i][j]=0
    
    
    
    #  trouver les contour grace a l'image binnaire
    contours, hierarchy = cv.findContours(binary, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    
    NewDraw = cv.drawContours(newImage, contours, -1, (0, 255, 0), 2)



###########

    # contoursOstu, hierarchy6 = cv.findContours(binaryOtsu, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    
    
    
    
    
    
    # contoursOt, hierarchy3 = cv.findContours(binaryOtsu, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    
    
    
    # contoursOstu, hierarchy6 = cv.findContours(binaryOtsu, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    # # draw all contours
    

    
    
    # # for i in range(len(newImage)):
    # #     for j in range(len(newImage[0])):
    # #         if newImage[i][j] == 0:
    # #             calpix = calpix +1
    
    
    # comboOtsu = cv.drawContours(image, contoursOstu, -1, (0, 255, 0), 2)
    
    # NewImageCheckVert = cv.drawContours(newImage, contoursOstu, -1, (0, 255, 0), 2)
    
    
    # NewImageCheckVert = cv.cvtColor(NewImageCheckVert, cv.COLOR_BGR2GRAY) 
    
    
    nBRPixel = 0
    
    for i in range(len(NewDraw)):
        for j in range(len(NewDraw[0])):
            if NewDraw[i][j] > 0:
                nBRPixel = nBRPixel +1
                
    
    return contours




def fullGene(image,mutation, loops):
    result = [0]*256
    best = [0]*256
    bestValue = 0
    bestIndex = 0
    
    saveIndexBest = 0
    # WORSTS
    
    worst = [0]*256
    worstValue = 0
    worstIndexs = [-1]*6
    
    
    # BEST
    
    
    
        
    bestSelected = [0]*256
    bestValueSelected = 0
    bestIndexes = [-1]*3
    
    
    listBest = []
    listWorst = []
    
    
    
    #Generer population
    
    populationN = 20
    
    
    i = 0
    population = numpy.empty(populationN, dtype=object)
    #generer une population de 20 solutions
    while (i<populationN):
        population[i] = genererSolutionV3(len(image),len(image[1]),200)
        i = i +1
    
    
    
    
    
    
    #big loop
    
    cpt = 0
    
    while(cpt < loops):
        # do it
        
        # Select 6 Best
        if(cpt == 0):
            bestSelected = population[0]
            bestValueSelected = globalFitnessFunction(MatchGoFull(image, population[0]))
            saveidex=0
        
        
        for cuck in  range(0,populationN):
                
                cond = globalFitnessFunction(MatchGoFull(image,population[0]))>bestValueSelected
                if(cond):
                      
                      bestSelected = population[cuck]
                      bestValueSelected = globalFitnessFunction(MatchGoFull(image, population[cuck]))
                      saveidex = cuck
            
    

        #tour 1
        bestIndexes1=saveidex
       
        #tour 2
        # Vraie tour 2
        i=0
        while(i<populationN):
                
            if(checkIfExist(i,bestIndexes)==False):
                        
                        cond = globalFitnessFunction(MatchGoFull(image,population[0]))>bestValueSelected
                        if(cond):
                               
                               bestSelected = population[i]
                               bestValueSelected = globalFitnessFunction(MatchGoFull(image, population[i]))
                               saveidex = i
            i = i +1
            bestIndexes2=saveidex
            
        #tour 3
        # Vraie tour 3
        i=0
        while(i<populationN):
                
            if(checkIfExist(i,bestIndexes)==False):
                        
                        cond = globalFitnessFunction(MatchGoFull(image,population[0]))>bestValueSelected
                        if(cond):
                               
                               bestSelected = population[i]
                               bestValueSelected = globalFitnessFunction(MatchGoFull(image, population[i]))
                               saveidex = i
            i = i +1
            bestIndexes3=saveidex
        where = 128
        
        fils1 = genererSolutionV3(len(image),len(image[1]),200)
        fils2 = genererSolutionV3(len(image),len(image[1]),200)
        fils3 = genererSolutionV3(len(image),len(image[1]),200)
        fils4 = genererSolutionV3(len(image),len(image[1]),200)
        fils5 = genererSolutionV3(len(image),len(image[1]),200)
        fils6 = genererSolutionV3(len(image),len(image[1]),200)
        
       
        #reporduction
        
        #where = random.randint(1, 254)
        
        where1 = 80
        where2 = 160
        
        # fils1 = reproDuctionChild1(population[bestIndexes1], population[bestIndexes2], where)
        # fils2 = reproDuctionChild2(population[bestIndexes1], population[bestIndexes2], where)

        # #where = random.randint(1, 254)
        # fils3 = reproDuctionChild1(population[bestIndexes2], population[bestIndexes3], where)
        # fils4 = reproDuctionChild2(population[bestIndexes2], population[bestIndexes3], where)
        

        # #where = random.randint(1, 254)
        # fils5 = reproDuctionChild1(population[bestIndexes1], population[bestIndexes3], where)
        # fils6 = reproDuctionChild2(population[bestIndexes1], population[bestIndexes3], where)
        
        


        wheres = gen2Wheres()
        where1 = wheres[0]
        where2 = wheres[1]
        
          
        fils1 = reproDuctionChild1TwoPoints(population[bestIndexes1], population[bestIndexes2],0,where1,where2)
        fils2 = reproDuctionChild2TwoPoints(population[bestIndexes1], population[bestIndexes2],0,where1,where2)
        
        wheres = gen2Wheres()
        where1 = wheres[0]
        where2 = wheres[1]

        fils3 = reproDuctionChild1TwoPoints(population[bestIndexes2], population[bestIndexes3],0,where1,where2)
        fils4 = reproDuctionChild2TwoPoints(population[bestIndexes2], population[bestIndexes3],0,where1,where2)

        wheres = gen2Wheres()
        where1 = wheres[0]
        where2 = wheres[1]
        
        
        fils5 = reproDuctionChild1TwoPoints(population[bestIndexes1], population[bestIndexes3],0,where1,where2)
        fils6 = reproDuctionChild2TwoPoints(population[bestIndexes1], population[bestIndexes3],0,where1,where2)
        

        listFullPopulation = numpy.empty(populationN+6, dtype=object)
        

        
        listFullPopulation[20] = fils1.copy()
        listFullPopulation[21] = fils2.copy()
        listFullPopulation[22] = fils3.copy()
        listFullPopulation[23] = fils4.copy()
        listFullPopulation[24] = fils5.copy()
        listFullPopulation[25] = fils6.copy()

        

        
        
        for i in range(len(population)):
            listFullPopulation[i] = population[i]


            
        # mutation
        for i in range(len(listFullPopulation)):
            #rez = mutationXX(listFullPopulation[i], 1)
            rez = mutationXX2(listFullPopulation[i], mutation)
            
            listFullPopulation[i] = rez
        
            
        # for  i in range(len(population)):
        #     population[i] = selectBestAndRemove(listBest,image)
        
           # bestSelected = population[0]
           #  bestValueSelected    
        
        #Get Best
        best = listFullPopulation[0]
        bestV =  globalFitnessFunction(MatchGoFull(image,best))
        for  i in range(len(listFullPopulation)):
                cond = globalFitnessFunction(MatchGoFull(image,listFullPopulation[i]))>bestValueSelected
                if(cond):
                      
                      bestSelected = listFullPopulation[i]
                      bestValueSelected = globalFitnessFunction(MatchGoFull(image, listFullPopulation[i]))
                      saveIndexBest = i
                      
        # Elimniation
        
        #get 20 best
        
        worstSixIndexes = [-1]*6
        worstSixIndexes = get6Worst(image,listFullPopulation)
        # print("pires ", worstSixIndexes)
        #population = get20Best(image,listFullPopulation)
        
        
        # new populace
        v = 0
        for i in range(len(listFullPopulation)):
            if(checkIfExist(i,worstSixIndexes)==False):
                
                population[v] = listFullPopulation[i]
                v = v +1
        
        cpt = cpt+1
        print("Best in gen ",cpt," is N:", saveIndexBest," with value = ", bestValueSelected)
        
            


            


        
        
        

    
    
    
    
    return bestSelected



































########################################### FIN FULL GEN ###########################################

########################################### DEBUT FULL GEN PROBA ###########################################

#Probabilistic Gen
"""
def fullGeneProbablistic(image,mutation, loops):
    Archivage = [0]*loops
    result = [0]*256
    best = [0]*256
    bestValue = 0
    bestIndex = 0
    minMaxValue = 200
    
    saveIndexBest = 0
    # WORSTS
    
    worst = [0]*256
    worstValue = 0
    
    nouveaunee = 10
    worstIndexs = [-1]*nouveaunee
    
    
    # BEST
    
    
    
        
    bestSelected = [0]*256
    bestValueSelected = 0
    bestIndexes = [-1]*3
    
    
    listBest = []
    listWorst = []
    
    
    
    #Generer population
    
   
    
    populationN = 20
    
    i = 0
    population = numpy.empty(populationN, dtype=object)
    valuePop  = numpy.empty(populationN, dtype=object)
    #generer une population de 20 solutions
    while (i<populationN):
        population[i] = genererSolutionV3(len(image),len(image[1]),minMaxValue)
        i = i +1
    
    
    # population[0] = histoEg
    # population[1] =histoAd
    #big loop
    
    cpt = 0
    
    while(cpt < loops):
        # do it
        
        # Select 6 Best
        if(cpt == 0):
            bestSelected = population[0]
            bestValueSelected = globalFitnessFunction(MatchGoFull(image, population[0]))
            saveidex=0
        
        
        #saveAll
        
        if(cpt == 0):
            
            for cuck in  range(0,populationN):
                
                valuePop[cuck] = globalFitnessFunction(MatchGoFull(image,population[cuck]))

            
        
        for cuck in  range(0,populationN):
                
                cond = valuePop[0]>bestValueSelected
                if(cond):
                      
                      bestSelected = population[cuck]
                      bestValueSelected = valuePop[cuck]
                      saveidex = cuck
            
    

        #tour 1
        bestIndexes1=saveidex
       
        #tour 2
        # Vraie tour 2
        i=0
        while(i<populationN):
                
            if(checkIfExist(i,bestIndexes)==False):
                        
                        cond = valuePop[0]>bestValueSelected
                        if(cond):
                               
                               bestSelected = population[i]
                               bestValueSelected = valuePop[i]
                               saveidex = i
            i = i +1
            bestIndexes2=saveidex
            
        #tour 3
        # Vraie tour 3
        i=0
        while(i<populationN):
                
            if(checkIfExist(i,bestIndexes)==False):
                        
                        cond = valuePop[0]>bestValueSelected
                        if(cond):
                               
                               bestSelected = population[i]
                               bestValueSelected = valuePop[i]
                               saveidex = i
            i = i +1
            bestIndexes3=saveidex
        where = 128
        
        fils1 = genererSolutionV3(len(image),len(image[1]),minMaxValue)
        fils2 = genererSolutionV3(len(image),len(image[1]),minMaxValue)
        fils3 = genererSolutionV3(len(image),len(image[1]),minMaxValue)
        fils4 = genererSolutionV3(len(image),len(image[1]),minMaxValue)
        fils5 = genererSolutionV3(len(image),len(image[1]),minMaxValue)
        fils6 = genererSolutionV3(len(image),len(image[1]),minMaxValue)
        
       
        #reporduction
        
        
        #Get fitness
        
        
        fitnessPopulation = [0]*populationN
        fitnessPopulationRelative = [0]*populationN
        fitnessCumulative  = [0]*populationN
        sommeDesFitnesse = 0
        
        
        
        
        
        for i in range(len(population)):
            
            # fitnessPopulation[i] = valuePop[i]
            sommeDesFitnesse = sommeDesFitnesse + valuePop[i]
        
        
        # TRI par bulle ? #############################################
        
        
        for i in range(len(population)-1):
            saveValue = [0]*256
            saveIndex = 0
            saveFitness =0
            for j in range(i,len(population)):
                if valuePop[i]<valuePop[j]:
                    # e9lab les valeurs
                    saveValue = population[i]
                    population[i] = population[j]
                    population[j] = saveValue
                    # e9leb les fitness
                    saveFitness = valuePop[i]
                    valuePop[i] = valuePop[j]
                    valuePop[j] = saveFitness
                    
        
        

                    
        
        
        
        #Get fitness relative
        
        for i in range(len(population)):
            

            fitnessPopulationRelative[i] = (valuePop[i] / sommeDesFitnesse) * 100 
            # print("Fitnesse  de ", i, fitnessPopulation[i] )
          
        
        #Get fitness cumulative
        somCum = 0
        for i in range(len(population)):
            somCum = somCum + fitnessPopulationRelative[i]
            fitnessCumulative[i] = somCum
            # print("Fitnesse cumulative de ", i, somCum)
            
            
            
        
        # for i in range(len(population)):
        #     print("Fitnesses cumulative ", i ," est ", fitnessCumulative[i])
    
        #selection par roullette
        
        random1 = random.random()*100
        random2 = random.random()*100
        random3 = random.random()*100
        

        
        # print("Random = ", random1)
        cptChoix = 0
        boolVar = True
        condiChoice = cptChoix < populationN and boolVar == True
        while (cptChoix < populationN and boolVar == True):
           
            if(random1>fitnessCumulative[cptChoix]):
                cptChoix = cptChoix +1
            else:
                reproducteur1 = cptChoix
                boolVar = False

        if boolVar == True:
            reproducteur1 = populationN-1
            
        #Select 2
        
        
        reproducteur2 = reproducteur1
        
        
        while reproducteur1 == reproducteur2:
            random2 = random.random()*100
            # print("Random = ", random2)
            cptChoix = 0
            boolVar = True
            condiChoice = cptChoix < populationN and boolVar == True
            while (cptChoix < populationN and boolVar == True):
                
                if(random2>fitnessCumulative[cptChoix]):
                    cptChoix = cptChoix +1
                else:
                    reproducteur2 = cptChoix
                    boolVar = False
    
            if boolVar == True:
                reproducteur2 = populationN-1




        
        
        reproducteur3 = reproducteur1
        
        #Select 3
        
        while reproducteur1 == reproducteur3 or reproducteur2 == reproducteur3:
            random3 = random.random()*100
            # print("Random = ", random3)
            cptChoix = 0
            boolVar = True
            condiChoice = cptChoix < populationN and boolVar == True
            while (cptChoix < populationN and boolVar == True):
                
                if(random3>fitnessCumulative[cptChoix]):
                    cptChoix = cptChoix +1
                else:
                    reproducteur3 = cptChoix
                    boolVar = False
    
            if boolVar == True:
                reproducteur3 = populationN-1
                
                
        reproducteur4 = reproducteur3

        while reproducteur4 == reproducteur3 or reproducteur4 == reproducteur2 or reproducteur4 == reproducteur1:
            random3 = random.random()*100
            # print("Random = ", random3)
            cptChoix = 0
            boolVar = True
            condiChoice = cptChoix < populationN and boolVar == True
            while (cptChoix < populationN and boolVar == True):
                
                if(random3>fitnessCumulative[cptChoix]):
                    cptChoix = cptChoix +1
                else:
                    reproducteur4 = cptChoix
                    boolVar = False
    
            if boolVar == True:
                reproducteur4 = populationN-1
                
        
        
        
        
        
            
        
        
        
        
        
        # wheres = gen2Wheres()
        # where1 = wheres[0]
        # where2 = wheres[1]
        
          
        # fils1 = reproDuctionChild1TwoPoints(population[bestIndexes1], population[bestIndexes2],0,where1,where2)
        # fils2 = reproDuctionChild2TwoPoints(population[bestIndexes1], population[bestIndexes2],0,where1,where2)
        
        # wheres = gen2Wheres()
        # where1 = wheres[0]
        # where2 = wheres[1]

        # fils3 = reproDuctionChild1TwoPoints(population[bestIndexes2], population[bestIndexes3],0,where1,where2)
        # fils4 = reproDuctionChild2TwoPoints(population[bestIndexes2], population[bestIndexes3],0,where1,where2)

        # wheres = gen2Wheres()
        # where1 = wheres[0]
        # where2 = wheres[1]
        
        
        
        
        # print("les reproducteurs sont ", reproducteur1," ",reproducteur2, " ", reproducteur3, " ", reproducteur4)
        
        #where = random.randint(1, 254)
        
                
        # wheres = gen2Wheres()
        where1 = 80
        where2 = 160
        
        # ########################### 1 point fixe ######################################################
        
        
        
        # fils1 = reproDuctionChild1(population[reproducteur1], population[reproducteur2], where)
        # fils2 = reproDuctionChild2(population[reproducteur1], population[reproducteur2], where)
            
        
        # #where = random.randint(1, 254)
        # fils3 = reproDuctionChild1(population[reproducteur2], population[reproducteur3], where)
        # fils4 = reproDuctionChild2(population[reproducteur2], population[reproducteur3], where)
        
        # #where = random.randint(1, 254)
        # fils5 = reproDuctionChild1(population[reproducteur1], population[reproducteur3], where)
        # fils6 = reproDuctionChild2(population[reproducteur1], population[reproducteur3], where)
        
        
        # ########################### 2 point fixe ######################################################
        
        #1 point
        
        # wheres = gen2Wheres()
        # # where1 = wheres[0]
        # # where2 = wheres[1]
        # where1 = random.randint(1,254)
        # # where2 = 170
        
        # fils1 = reproDuctionChild1(population[reproducteur1], population[reproducteur2],where1)
        # fils2 = reproDuctionChild2(population[reproducteur1], population[reproducteur2],where1)
        # # wheres = gen2Wheres()
        # # where1 = wheres[0]
        # # where2 = wheres[1]     
        
        # #where = random.randint(1, 254)
        # fils3 = reproDuctionChild1(population[reproducteur2], population[reproducteur3],where1)
        # fils4 = reproDuctionChild2(population[reproducteur2], population[reproducteur3],where1)
        # # wheres = gen2Wheres()
        # # where1 = wheres[0]
        # # where2 = wheres[1] 
        # #where = random.randint(1, 254)
        # fils5 = reproDuctionChild1(population[reproducteur1], population[reproducteur3],where1)
        # fils6 = reproDuctionChild2(population[reproducteur1], population[reproducteur3],where1)
        
        #mutli point
        wheres = gen2Wheres()
        where1 = wheres[0]
        where2 = wheres[1]

        # where2 = 170
                
        fils1 = reproDuctionChild1TwoPoints(population[reproducteur1], population[reproducteur2], 0,where1,where2)
        fils2 = reproDuctionChild2TwoPoints(population[reproducteur1], population[reproducteur2],0,where1,where2)
        
        wheres = gen2Wheres()
        where1 = wheres[0]
        where2 = wheres[1]
        
        fils3 = reproDuctionChild1TwoPoints(population[reproducteur2], population[reproducteur3], 0,where1,where2)
        fils4 = reproDuctionChild2TwoPoints(population[reproducteur2], population[reproducteur3],0,where1,where2)
        
        # where1 = wheres[0]
        # where2 = wheres[1] 
        
        wheres = gen2Wheres()
        where1 = wheres[0]
        where2 = wheres[1]
        
        
        #where = random.randint(1, 254)
        fils5 = reproDuctionChild1TwoPoints(population[reproducteur1], population[reproducteur3], 0,where1,where2)
        fils6 = reproDuctionChild2TwoPoints(population[reproducteur1], population[reproducteur3], 0,where1,where2)




        listFullPopulation = numpy.empty(populationN+nouveaunee, dtype=object)
        

        
        listFullPopulation[populationN] = fils1.copy()
        listFullPopulation[populationN+1] = fils2.copy()
        listFullPopulation[populationN+2] = fils3.copy()
        listFullPopulation[populationN+3] = fils4.copy()
        listFullPopulation[populationN+4] = fils5.copy()
        listFullPopulation[populationN+5] = fils6.copy()
        
        
        wheres = gen2Wheres()
        where1 = wheres[0]
        where2 = wheres[1]
        
        
        
        fils7 =  reproDuctionChild1TwoPoints(population[reproducteur4], population[reproducteur1], 0,where1,where2)
        fils8 =  reproDuctionChild2TwoPoints(population[reproducteur4], population[reproducteur1], 0,where1,where2)
       
        wheres = gen2Wheres()
        where1 = wheres[0]
        where2 = wheres[1]        

        fils9 =  reproDuctionChild1TwoPoints(population[reproducteur4], population[reproducteur2], 0,where1,where2)
        fils10 =  reproDuctionChild2TwoPoints(population[reproducteur4], population[reproducteur2], 0,where1,where2)



        
        
        listFullPopulation[populationN+6] = fils7.copy()
        listFullPopulation[populationN+7] = fils8.copy()
        listFullPopulation[populationN+8] = fils9.copy()
        listFullPopulation[populationN+9] = fils10.copy()

                # Full population value
        fullpopValue = numpy.empty(populationN+nouveaunee, dtype=object)
        
        for i in range(populationN):
            fullpopValue[i] = valuePop[i]
            

        
        
        for i in range(len(population)):
            listFullPopulation[i] = population[i]


        # correct he mutation todo    
        # mutation
        for i in range(populationN,len(listFullPopulation)):
            #rez = mutationXX(listFullPopulation[i], 1)
            rez = mutationXX2(listFullPopulation[i], mutation,minMaxValue)
            
            listFullPopulation[i] = rez
        
            
        # for  i in range(len(population)):
        #     population[i] = selectBestAndRemove(listBest,image)
        
           # bestSelected = population[0]
           #  bestValueSelected    
        
        #Get Best
        

        
        # Full population value
        
        
        fullpopValue[populationN] = globalFitnessFunction(MatchGoFull(image,listFullPopulation[populationN]))
        fullpopValue[populationN+1] = globalFitnessFunction(MatchGoFull(image,listFullPopulation[populationN+1]))
        fullpopValue[populationN+2] = globalFitnessFunction(MatchGoFull(image,listFullPopulation[populationN+2]))
        fullpopValue[populationN+3] = globalFitnessFunction(MatchGoFull(image,listFullPopulation[populationN+3]))
        fullpopValue[populationN+4] = globalFitnessFunction(MatchGoFull(image,listFullPopulation[populationN+4]))
        fullpopValue[populationN+5] = globalFitnessFunction(MatchGoFull(image,listFullPopulation[populationN+5]))
        fullpopValue[populationN+6] = globalFitnessFunction(MatchGoFull(image,listFullPopulation[populationN+6]))
        fullpopValue[populationN+7] = globalFitnessFunction(MatchGoFull(image,listFullPopulation[populationN+7]))
        fullpopValue[populationN+8] = globalFitnessFunction(MatchGoFull(image,listFullPopulation[populationN+8]))
        fullpopValue[populationN+9] = globalFitnessFunction(MatchGoFull(image,listFullPopulation[populationN+9]))
        
        
        best = listFullPopulation[0]
        bestV =  globalFitnessFunction(MatchGoFull(image,best))
        for  i in range(len(listFullPopulation)):
                
                cond = fullpopValue[i]>bestValueSelected
                if(cond):
                      
                      bestSelected = listFullPopulation[i]
                      bestValueSelected = fullpopValue[i]
                      saveIndexBest = i
                      
        # Elimniation
        
        #get 20 best
        
        worstSixIndexes = [-1]*nouveaunee
        worstSixIndexes = getNWorstV2(image,listFullPopulation,populationN,nouveaunee,fullpopValue)
        
        #population = get20Best(image,listFullPopulation)
        
        
        # new populace
        v = 0
        for i in range(len(listFullPopulation)):
            
            # print("C",i," Valeur ",fullpopValue[i])
            if(checkIfExist(i,worstSixIndexes)==False):
                
                population[v] = listFullPopulation[i]
                valuePop[v] = fullpopValue[i]
                v = v +1
        
        Archivage[cpt] = bestValueSelected
        cpt = cpt+1
        if cpt % 10 == 0:
            print("Best in gen ",cpt," is N:", saveIndexBest," with value = ", bestValueSelected)
        
        
            


            


        
        
        

    
    plt.plot(Archivage)
    plt.xlabel('Generations')
    plt.ylabel('Qualité du contraste')
    plt.show()
    
    
    
    return bestSelected
"""

def fullGeneProbablistic(image,mutation, loops):
    Archivage = [0]*loops
    result = [0]*256
    best = [0]*256
    bestValue = 0
    bestIndex = 0
    minMaxValue = 200
    
    saveIndexBest = 0
    # WORSTS
    
    worst = [0]*256
    worstValue = 0
    
    nouveaunee = 10
    worstIndexs = [-1]*nouveaunee
    
    
    # BEST
    
    
    
        
    bestSelected = [0]*256
    bestValueSelected = 0
    bestIndexes = [-1]*3
    
    
    listBest = []
    listWorst = []
    
    
    
    #Generer population
    
   
    
    populationN = 20
    
    i = 0
    population = numpy.empty(populationN, dtype=object)
    valuePop  = numpy.empty(populationN, dtype=object)
    #generer une population de 20 solutions
    while (i<populationN):
        population[i] = genererSolutionV3(len(image),len(image[1]),minMaxValue)
        i = i +1
    
    
    # population[0] = histoEg
    # population[1] =histoAd
    #big loop
    
    cpt = 0
    
    while(cpt < loops):
        # do it
        
        # Select 6 Best
        if(cpt == 0):
            bestSelected = population[0]
            bestValueSelected = globalFitnessFunction(MatchGoFull(image, population[0]))
            saveidex=0
        
        
        #saveAll
        
        if(cpt == 0):
            
            for cuck in  range(0,populationN):
                
                valuePop[cuck] = globalFitnessFunction(MatchGoFull(image,population[cuck]))

            
        
        for cuck in  range(0,populationN):
                
                cond = valuePop[0]>bestValueSelected
                if(cond):
                      
                      bestSelected = population[cuck]
                      bestValueSelected = valuePop[cuck]
                      saveidex = cuck
            
    

        #tour 1
        bestIndexes1=saveidex
       
        #tour 2
        # Vraie tour 2
        i=0
        while(i<populationN):
                
            if(checkIfExist(i,bestIndexes)==False):
                        
                        cond = valuePop[0]>bestValueSelected
                        if(cond):
                               
                               bestSelected = population[i]
                               bestValueSelected = valuePop[i]
                               saveidex = i
            i = i +1
            bestIndexes2=saveidex
            
        #tour 3
        # Vraie tour 3
        i=0
        while(i<populationN):
                
            if(checkIfExist(i,bestIndexes)==False):
                        
                        cond = valuePop[0]>bestValueSelected
                        if(cond):
                               
                               bestSelected = population[i]
                               bestValueSelected = valuePop[i]
                               saveidex = i
            i = i +1
            bestIndexes3=saveidex
        where = 128
        
        fils1 = genererSolutionV3(len(image),len(image[1]),minMaxValue)
        fils2 = genererSolutionV3(len(image),len(image[1]),minMaxValue)
        fils3 = genererSolutionV3(len(image),len(image[1]),minMaxValue)
        fils4 = genererSolutionV3(len(image),len(image[1]),minMaxValue)
        fils5 = genererSolutionV3(len(image),len(image[1]),minMaxValue)
        fils6 = genererSolutionV3(len(image),len(image[1]),minMaxValue)
        
       
        #reporduction
        
        
        #Get fitness
        
        
        fitnessPopulation = [0]*populationN
        fitnessPopulationRelative = [0]*populationN
        fitnessCumulative  = [0]*populationN
        sommeDesFitnesse = 0
        
        
        
        
        
        for i in range(len(population)):
            
            # fitnessPopulation[i] = valuePop[i]
            sommeDesFitnesse = sommeDesFitnesse + valuePop[i]
        
        
        # TRI par bulle ? #############################################
        
        
        for i in range(len(population)-1):
            saveValue = [0]*256
            saveIndex = 0
            saveFitness =0
            for j in range(i,len(population)):
                if valuePop[i]<valuePop[j]:
                    # e9lab les valeurs
                    saveValue = population[i]
                    population[i] = population[j]
                    population[j] = saveValue
                    # e9leb les fitness
                    saveFitness = valuePop[i]
                    valuePop[i] = valuePop[j]
                    valuePop[j] = saveFitness
                    
        
        

                    
        
        
        
        #Get fitness relative
        
        for i in range(len(population)):
            

            fitnessPopulationRelative[i] = (valuePop[i] / sommeDesFitnesse) * 100 
            # print("Fitnesse  de ", i, fitnessPopulation[i] )
          
        
        #Get fitness cumulative
        somCum = 0
        for i in range(len(population)):
            somCum = somCum + fitnessPopulationRelative[i]
            fitnessCumulative[i] = somCum
            # print("Fitnesse cumulative de ", i, somCum)
            
            
            
        
        # for i in range(len(population)):
        #     print("Fitnesses cumulative ", i ," est ", fitnessCumulative[i])
    
        #selection par roullette
        
        random1 = random.random()*100
        random2 = random.random()*100
        random3 = random.random()*100
        

        
        # print("Random = ", random1)
        cptChoix = 0
        boolVar = True
        condiChoice = cptChoix < populationN and boolVar == True
        while (cptChoix < populationN and boolVar == True):
           
            if(random1>fitnessCumulative[cptChoix]):
                cptChoix = cptChoix +1
            else:
                reproducteur1 = cptChoix
                boolVar = False

        if boolVar == True:
            reproducteur1 = populationN-1
            
        #Select 2
        
        
        reproducteur2 = reproducteur1
        
        
        while reproducteur1 == reproducteur2:
            random2 = random.random()*100
            # print("Random = ", random2)
            cptChoix = 0
            boolVar = True
            condiChoice = cptChoix < populationN and boolVar == True
            while (cptChoix < populationN and boolVar == True):
                
                if(random2>fitnessCumulative[cptChoix]):
                    cptChoix = cptChoix +1
                else:
                    reproducteur2 = cptChoix
                    boolVar = False
    
            if boolVar == True:
                reproducteur2 = populationN-1




        
        
        reproducteur3 = reproducteur1
        
        #Select 3
        
        while reproducteur1 == reproducteur3 or reproducteur2 == reproducteur3:
            random3 = random.random()*100
            # print("Random = ", random3)
            cptChoix = 0
            boolVar = True
            condiChoice = cptChoix < populationN and boolVar == True
            while (cptChoix < populationN and boolVar == True):
                
                if(random3>fitnessCumulative[cptChoix]):
                    cptChoix = cptChoix +1
                else:
                    reproducteur3 = cptChoix
                    boolVar = False
    
            if boolVar == True:
                reproducteur3 = populationN-1
                
                
        reproducteur4 = reproducteur3

        while reproducteur4 == reproducteur3 or reproducteur4 == reproducteur2 or reproducteur4 == reproducteur1:
            random3 = random.random()*100
            # print("Random = ", random3)
            cptChoix = 0
            boolVar = True
            condiChoice = cptChoix < populationN and boolVar == True
            while (cptChoix < populationN and boolVar == True):
                
                if(random3>fitnessCumulative[cptChoix]):
                    cptChoix = cptChoix +1
                else:
                    reproducteur4 = cptChoix
                    boolVar = False
    
            if boolVar == True:
                reproducteur4 = populationN-1
                
        
        
        
        
        
            
        
        
        
        
        
        # wheres = gen2Wheres()
        # where1 = wheres[0]
        # where2 = wheres[1]
        
          
        # fils1 = reproDuctionChild1TwoPoints(population[bestIndexes1], population[bestIndexes2],0,where1,where2)
        # fils2 = reproDuctionChild2TwoPoints(population[bestIndexes1], population[bestIndexes2],0,where1,where2)
        
        # wheres = gen2Wheres()
        # where1 = wheres[0]
        # where2 = wheres[1]

        # fils3 = reproDuctionChild1TwoPoints(population[bestIndexes2], population[bestIndexes3],0,where1,where2)
        # fils4 = reproDuctionChild2TwoPoints(population[bestIndexes2], population[bestIndexes3],0,where1,where2)

        # wheres = gen2Wheres()
        # where1 = wheres[0]
        # where2 = wheres[1]
        
        
        
        
        # print("les reproducteurs sont ", reproducteur1," ",reproducteur2, " ", reproducteur3, " ", reproducteur4)
        
        #where = random.randint(1, 254)
        
                
        # wheres = gen2Wheres()
        where1 = 80
        where2 = 160
        
        # ########################### 1 point fixe ######################################################
        
        
        
        # fils1 = reproDuctionChild1(population[reproducteur1], population[reproducteur2], where)
        # fils2 = reproDuctionChild2(population[reproducteur1], population[reproducteur2], where)
            
        
        # #where = random.randint(1, 254)
        # fils3 = reproDuctionChild1(population[reproducteur2], population[reproducteur3], where)
        # fils4 = reproDuctionChild2(population[reproducteur2], population[reproducteur3], where)
        
        # #where = random.randint(1, 254)
        # fils5 = reproDuctionChild1(population[reproducteur1], population[reproducteur3], where)
        # fils6 = reproDuctionChild2(population[reproducteur1], population[reproducteur3], where)
        
        
        # ########################### 2 point fixe ######################################################
        
        #1 point
        
        # wheres = gen2Wheres()
        # # where1 = wheres[0]
        # # where2 = wheres[1]
        # where1 = random.randint(1,254)
        # # where2 = 170
        
        # fils1 = reproDuctionChild1(population[reproducteur1], population[reproducteur2],where1)
        # fils2 = reproDuctionChild2(population[reproducteur1], population[reproducteur2],where1)
        # # wheres = gen2Wheres()
        # # where1 = wheres[0]
        # # where2 = wheres[1]     
        
        # #where = random.randint(1, 254)
        # fils3 = reproDuctionChild1(population[reproducteur2], population[reproducteur3],where1)
        # fils4 = reproDuctionChild2(population[reproducteur2], population[reproducteur3],where1)
        # # wheres = gen2Wheres()
        # # where1 = wheres[0]
        # # where2 = wheres[1] 
        # #where = random.randint(1, 254)
        # fils5 = reproDuctionChild1(population[reproducteur1], population[reproducteur3],where1)
        # fils6 = reproDuctionChild2(population[reproducteur1], population[reproducteur3],where1)
        
        #mutli point
        wheres = gen2Wheres()
        where1 = wheres[0]
        where2 = wheres[1]

        # where2 = 170
                
        fils1 = reproDuctionChild1TwoPoints(population[reproducteur1], population[reproducteur2], 0,where1,where2)
        fils2 = reproDuctionChild2TwoPoints(population[reproducteur1], population[reproducteur2],0,where1,where2)
        
        wheres = gen2Wheres()
        where1 = wheres[0]
        where2 = wheres[1]
        
        fils3 = reproDuctionChild1TwoPoints(population[reproducteur2], population[reproducteur3], 0,where1,where2)
        fils4 = reproDuctionChild2TwoPoints(population[reproducteur2], population[reproducteur3],0,where1,where2)
        
        # where1 = wheres[0]
        # where2 = wheres[1] 
        
        wheres = gen2Wheres()
        where1 = wheres[0]
        where2 = wheres[1]
        
        
        #where = random.randint(1, 254)
        fils5 = reproDuctionChild1TwoPoints(population[reproducteur1], population[reproducteur3], 0,where1,where2)
        fils6 = reproDuctionChild2TwoPoints(population[reproducteur1], population[reproducteur3], 0,where1,where2)




        listFullPopulation = numpy.empty(populationN+nouveaunee, dtype=object)
        

        
        listFullPopulation[populationN] = fils1.copy()
        listFullPopulation[populationN+1] = fils2.copy()
        listFullPopulation[populationN+2] = fils3.copy()
        listFullPopulation[populationN+3] = fils4.copy()
        listFullPopulation[populationN+4] = fils5.copy()
        listFullPopulation[populationN+5] = fils6.copy()
        
        
        wheres = gen2Wheres()
        where1 = wheres[0]
        where2 = wheres[1]
        
        
        
        fils7 =  reproDuctionChild1TwoPoints(population[reproducteur4], population[reproducteur1], 0,where1,where2)
        fils8 =  reproDuctionChild2TwoPoints(population[reproducteur4], population[reproducteur1], 0,where1,where2)
       
        wheres = gen2Wheres()
        where1 = wheres[0]
        where2 = wheres[1]        

        fils9 =  reproDuctionChild1TwoPoints(population[reproducteur4], population[reproducteur2], 0,where1,where2)
        fils10 =  reproDuctionChild2TwoPoints(population[reproducteur4], population[reproducteur2], 0,where1,where2)



        
        
        listFullPopulation[populationN+6] = fils7.copy()
        listFullPopulation[populationN+7] = fils8.copy()
        listFullPopulation[populationN+8] = fils9.copy()
        listFullPopulation[populationN+9] = fils10.copy()

                # Full population value
        fullpopValue = numpy.empty(populationN+nouveaunee, dtype=object)
        
        for i in range(populationN):
            fullpopValue[i] = valuePop[i]
            

        
        
        for i in range(len(population)):
            listFullPopulation[i] = population[i]


        # correct he mutation todo    
        # mutation
        for i in range(populationN,len(listFullPopulation)):
            #rez = mutationXX(listFullPopulation[i], 1)
            rez = mutationXX2(listFullPopulation[i], mutation,minMaxValue)
            
            listFullPopulation[i] = rez
        
            
        # for  i in range(len(population)):
        #     population[i] = selectBestAndRemove(listBest,image)
        
           # bestSelected = population[0]
           #  bestValueSelected    
        
        #Get Best
        

        
        # Full population value
        
        
        fullpopValue[populationN] = globalFitnessFunction(MatchGoFull(image,listFullPopulation[populationN]))
        fullpopValue[populationN+1] = globalFitnessFunction(MatchGoFull(image,listFullPopulation[populationN+1]))
        fullpopValue[populationN+2] = globalFitnessFunction(MatchGoFull(image,listFullPopulation[populationN+2]))
        fullpopValue[populationN+3] = globalFitnessFunction(MatchGoFull(image,listFullPopulation[populationN+3]))
        fullpopValue[populationN+4] = globalFitnessFunction(MatchGoFull(image,listFullPopulation[populationN+4]))
        fullpopValue[populationN+5] = globalFitnessFunction(MatchGoFull(image,listFullPopulation[populationN+5]))
        fullpopValue[populationN+6] = globalFitnessFunction(MatchGoFull(image,listFullPopulation[populationN+6]))
        fullpopValue[populationN+7] = globalFitnessFunction(MatchGoFull(image,listFullPopulation[populationN+7]))
        fullpopValue[populationN+8] = globalFitnessFunction(MatchGoFull(image,listFullPopulation[populationN+8]))
        fullpopValue[populationN+9] = globalFitnessFunction(MatchGoFull(image,listFullPopulation[populationN+9]))
        
        
        best = listFullPopulation[0]
        bestV =  globalFitnessFunction(MatchGoFull(image,best))
        for  i in range(len(listFullPopulation)):
                
                cond = fullpopValue[i]>bestValueSelected
                if(cond):
                      
                      bestSelected = listFullPopulation[i]
                      bestValueSelected = fullpopValue[i]
                      saveIndexBest = i
                      
        # Elimniation
        
        #get 20 best
        
        worstSixIndexes = [-1]*nouveaunee
        worstSixIndexes = getNWorstV2(image,listFullPopulation,populationN,nouveaunee,fullpopValue)
        
        #population = get20Best(image,listFullPopulation)
        
        
        # new populace
        v = 0
        for i in range(len(listFullPopulation)):
            
            # print("C",i," Valeur ",fullpopValue[i])
            if(checkIfExist(i,worstSixIndexes)==False):
                
                population[v] = listFullPopulation[i]
                valuePop[v] = fullpopValue[i]
                v = v +1
        
        Archivage[cpt] = bestValueSelected
        cpt = cpt+1
        if cpt % 10 == 0:
            print("Best in gen ",cpt," is N:", saveIndexBest," with value = ", bestValueSelected)
        
        
            


            


        
        
        

    
    # plt.plot(Archivage)
    # plt.xlabel('Generations')
    # plt.ylabel('Qualité du contraste')
    # plt.show()
    
    
    
    return bestSelected

def getNWorst(image, array, pop, nouveaunee):
    res = [-1]*(nouveaunee+1)
    worst = array[0]
    worstValue = globalFitnessFunction(MatchGoFull(image, array[0]))
    saveidex=0
    #optimisation precalculer les fitnesse
    precalculeValue = [0]*len(array)
    
    for i in range(len(array)):
        precalculeValue[i] = globalFitnessFunction(MatchGoFull(image, array[i]))
    
    for cuck in  range(0,pop+nouveaunee):    
    
    
        
        if(precalculeValue[cuck]<=worstValue):
             
                                   
             worst = array[cuck]
             worstValue = precalculeValue[cuck]
             saveidex = cuck
    
    res[0]=saveidex
    
    #true deb
    
    for v in range (0,nouveaunee):
        for cuck in  range(0,pop+nouveaunee):
             
            if(checkIfExist(cuck,res)==False):
                
                 
                worst = array[cuck]
                worstValue = precalculeValue[cuck]
                saveidex = cuck
    
                    
    
        i=0
        while(i<pop+nouveaunee):
            if(checkIfExist(i,res)==False):
    
                cond = precalculeValue[i]<worstValue
                if(cond):
                    
                    worst = array[i]
                    worstValue = precalculeValue[i]
                    saveidex = i
            i = i +1
        
        res[v]=saveidex       
            
    
    
    
    return res
    



def getNWorstV2(image, array, pop, nouveaunee, fitnesses):
    res = [-1]*(nouveaunee+1)
    worst = array[0]
    worstValue = globalFitnessFunction(MatchGoFull(image, array[0]))
    saveidex=0
    #optimisation precalculer les fitnesse
    precalculeValue = [0]*len(array)
    
    precalculeValue = fitnesses
    
    for cuck in  range(0,pop+nouveaunee):    
    
    
        
        if(precalculeValue[cuck]<=worstValue):
             
                                   
             worst = array[cuck]
             worstValue = precalculeValue[cuck]
             saveidex = cuck
    
    res[0]=saveidex
    
    #true deb
    
    for v in range (0,nouveaunee):
        for cuck in  range(0,pop+nouveaunee):
             
            if(checkIfExist(cuck,res)==False):
                
                 
                worst = array[cuck]
                worstValue = precalculeValue[cuck]
                saveidex = cuck
    
                    
    
        i=0
        while(i<pop+nouveaunee):
            if(checkIfExist(i,res)==False):
    
                cond = precalculeValue[i]<worstValue
                if(cond):
                    
                    worst = array[i]
                    worstValue = precalculeValue[i]
                    saveidex = i
            i = i +1
        
        res[v]=saveidex       
            
    
    
    
    return res
    


def get6Worst(image, array, pop, nouveaunee):
    res = [-1]*(nouveaunee+1)
    worst = array[0]
    worstValue = globalFitnessFunction(MatchGoFull(image, array[0]))
    saveidex=0
    
    
    #optimisation precalculer les fitnesse
    precalculeValue = [0]*len(array)
    
    for i in range(len(array)):
        precalculeValue[i] = globalFitnessFunction(MatchGoFull(image, array[i]))
    
    for cuck in  range(0,pop+nouveaunee):    
    
    
        cond = precalculeValue[cuck]<worstValue
        if(cond):
             
                                   
             worst = array[cuck]
             worstValue = precalculeValue[cuck]
             saveidex = cuck
    
    res[0]=saveidex
    
       

    for cuck in  range(0,pop+nouveaunee):
         
        if(checkIfExist(cuck,res)==False):
            
             
            worst = array[cuck]
            worstValue = precalculeValue[cuck]
            saveidex = cuck

                

    i=0
    while(i<pop+nouveaunee):
        if(checkIfExist(i,res)==False):

            cond = precalculeValue[i]<worstValue
            if(cond):
                
                worst = array[i]
                worstValue = precalculeValue[i]
                saveidex = i
        i = i +1
    
    res[1]=saveidex 


    for cuck in  range(0,pop+nouveaunee):
         
        if(checkIfExist(cuck,res)==False):
            
             
            worst = array[cuck]
            worstValue = precalculeValue[cuck]
            saveidex = cuck
            break




    i=0
    while(i<pop+nouveaunee):
        if(checkIfExist(i,res)==False):

            cond = precalculeValue[i]<=worstValue
            if(cond):
                
                worst = array[i]
                worstValue = precalculeValue[i]
                saveidex = i
        i = i +1
    
    res[2]=saveidex 

    for cuck in  range(0,pop+nouveaunee):
         
        if(checkIfExist(cuck,res)==False):
            
             
            worst = array[cuck]
            worstValue = precalculeValue[cuck]
            saveidex = cuck
            break
                

    i=0
    while(i<pop+nouveaunee):
        if(checkIfExist(i,res)==False):
         
            cond = precalculeValue[i]<=worstValue
            if(cond):
                
                worst = array[i]
                worstValue = precalculeValue[i]
                saveidex = i
        i = i +1
    
    res[3]=saveidex 

    for cuck in  range(0,pop+nouveaunee):
         
        if(checkIfExist(cuck,res)==False):
            
             
            worst = array[cuck]
            worstValue = precalculeValue[cuck]
            saveidex = cuck
            break

    i=0
    while(i<pop+nouveaunee):
        if(checkIfExist(i,res)==False):
           
            cond = precalculeValue[i]<=worstValue
            if(cond):
                
                worst = array[i]
                worstValue = precalculeValue[i]
                saveidex = i
        i = i +1
    
    res[4]=saveidex 


    for cuck in  range(0,pop+nouveaunee):
         
        if(checkIfExist(cuck,res)==False):
            
             
            worst = array[cuck]
            worstValue = precalculeValue[cuck]
            saveidex = cuck
            break

    i=0
    while(i<pop+nouveaunee):
        if(checkIfExist(i,res)==False):
            
            cond = precalculeValue[i]<=worstValue
            if(cond):
                
                worst = array[i]
                worstValue = precalculeValue[i]
                saveidex = i
        i = i +1
    
    res[5]=saveidex 

    for cuck in  range(0,pop+nouveaunee):
         
        if(checkIfExist(cuck,res)==False):
            
             
            worst = array[cuck]
            worstValue = precalculeValue[cuck]
            saveidex = cuck
            break

    i=0
    while(i<pop+nouveaunee):
        if(checkIfExist(i,res)==False):
            
            cond = precalculeValue[i]<=worstValue
            if(cond):
                
                worst = array[i]
                worstValue = precalculeValue[i]
                saveidex = i
        i = i +1
    
    res[6]=saveidex 
    
    


    return res


def get20Best(image,array):
    print("go best get gp")
    res = [-1]*20
    bestindex = 0
    bestValue = globalFitnessFunction(MatchGoFull(image, array[0]))
    i = 0
    j =0
    while  i < len(array) and bestindex < 20:
        
        j=0
        akbar = 0
        print("i ",i, "j ",j,"akbar ", akbar)
        while  j < len(array) and akbar < 6 :
            print("IN J > i ",i, "j ",j,"akbar ", akbar)
            currentValue = globalFitnessFunction(MatchGoFull(image,array[i]))
            
            if(checkIfExist(j, res)==False):
                cond = globalFitnessFunction(MatchGoFull(image,array[j]))>currentValue
                if(cond):
                    akbar = akbar + 1
        
            j = j +1
        if(akbar < 6):
            res[bestindex] = i
            bestindex = bestindex + 1
        i =i +1    
            
        
    return res
    
    


def selectBestAndRemove(liste,image):
    best = liste[0]
    bestValue = globalFitnessFunction(MatchGoFull(image, liste[0]))
     
    currentValue = 0
    indexB = 0
    
    for i in range (len(liste)):
        currentValue = globalFitnessFunction(MatchGoFull(image,liste[i]))
        if currentValue > bestValue:
            best = currentValue
            indexB = i
    
    del liste[indexB]
    
    return best

def reproDuctionChild1(cromoA, cromoB, where):
    res  = [0]*256
    
    for i in range(len(res)):
        if i <= where:
            res[i] = cromoA[i]
            #print("je prend a")
        else:
            res[i] = cromoB[i]
            #print("je prend b")
    
    
    return res

def reproDuctionChild2(cromoA, cromoB, where):
    res  = [0]*256
    
    for i in range(len(res)):
        if i <= where:
            res[i] = cromoB[i]
            #print("je prend a")
        else:
            res[i] = cromoA[i]
            #print("je prend b")
    
    
    return res
    

# two points crossOver 1 

def reproDuctionChild1TwoPoints(cromoA, cromoB, rand, where1,where2):
    res  = [0]*256
    
    if rand == 0:
        
        for i in range(len(res)):
            
            if i <= where1  or i > where2:
                res[i] = cromoA[i]
            else:
                res[i] = cromoB[i]
    
    return res

# two points crossOver 1 
                
def reproDuctionChild2TwoPoints(cromoA, cromoB, rand, where1,where2):
    res  = [0]*256
    
    if rand == 0:
        
        for i in range(len(res)):
            
            if i <= where1  or i > where2:
                res[i] = cromoB[i]
            else:
                res[i] = cromoA[i]
    return res  


# get 2 points

def gen2Wheres():
    
    res = [0]*2
    
    where1 = random.randint(1, 255)
    where2 = random.randint(1, 255)
    while where1 == where2:
        
         where1 = random.randint(1, 255)
         where2 = random.randint(1, 255)
    save = 0
    if where1 > where2:
        
       save = where1
       where1 = where2
       where2 = save
        
    res[0] = where1
    res[1] = where2
    return res
            

    
    
    return res

def mutationXX(crom, chance):
    res = crom.copy()
    where = random.randint(0, 255)
    
    if random.randint(0, 100) < chance:
        res[where] = random.randint(0, 200)
    
    
    return res
    


def mutationXX2(crom, chance,max):
    res = crom.copy()

    

    for i in range(len(crom)):
        if random.randint(0, 100) < chance:
            res[i] = random.randint(0, max)
            
        
    
    
    return res
    
    


def genererSolutionV3(x,y,max):
    sol = [0]*256
    
    for indX in range (len(sol)):
        sol[indX] = random.randint(0, max) 
        
    return sol
        



def globalFitnessFunction(image):
    result =0
    
  
   
    log1 =math.log(filtreSobel(image),10)
    
    # phase1 = math.log(math.log(filtreSobel(image)))
    phase1 = math.log(log1,10)
    phase2 = newContourOTSU(image) / (len(image) * len(image[1]))
    #phase2 = newContourBinnaire(image) / (len(image) * len(image[1]))
    phase3 = skimage.measure.shannon_entropy(image)
    result = phase1 * phase2 * phase3
    return result



def calculerNombreDePixelsContour(image):


    tigreOriginal = cv.cvtColor(image.astype('uint8') , cv.COLOR_BGR2RGB)
    tigreGris = cv.cvtColor(tigreOriginal, cv.COLOR_RGB2GRAY)

    # creater une image binnaire avec tresh hold X Y
    _, binary = cv.threshold(tigreGris,127,255, cv.THRESH_BINARY_INV)

    
    
    
    #  trouver les contour grace a l'image binnaire
    contours, hierarchy = cv.findContours(binary, cv.RETR_TREE, cv.CHAIN_APPROX_NONE)


    return(len(contours))


def calculerNombreDePixelsContourTriangle(image):


    tigreOriginal = cv.cvtColor(image.astype('uint8') , cv.COLOR_BGR2RGB)
    tigreGris = cv.cvtColor(tigreOriginal, cv.COLOR_RGB2GRAY)

    # creater une image binnaire avec tresh hold X Y
    _, binary = cv.threshold(tigreGris,127,255, cv.THRESH_TRIANGLE)

    
    
    
    #  trouver les contour grace a l'image binnaire
    contours, hierarchy = cv.findContours(binary, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)


    return(len(contours))

def calculerNombreDePixelsContourOTSU(image):


    tigreOriginal = cv.cvtColor(image.astype('uint8') , cv.COLOR_BGR2RGB)
    tigreGris = cv.cvtColor(tigreOriginal, cv.COLOR_RGB2GRAY)

    # creater une image binnaire avec tresh hold X Y
    _, binary = cv.threshold(tigreGris,127,255, cv.THRESH_OTSU)

    
    
    
    #  trouver les contour grace a l'image binnaire
    contours, hierarchy = cv.findContours(binary, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)


    return(len(contours))







def filtreSobel(image):
    im = image.copy()
    
    im = im.astype('int32')
    dx = ndimage.sobel(im, 0)  
    dy = ndimage.sobel(im, 1)  
    mag = numpy.hypot(dx, dy)  
    mag *= 255.0 / numpy.max(mag)  
    return EIZsommeDesIntensite(mag)

        
def MatchGoFull(image,source):
    # matched = hist_match(image, source)
    matched = histoMatchV3(image,source)
    return matched




        

# calculer somme totales des intensité d'une imager 

def EIZsommeDesIntensite(enter):
    somme = 0
    for i in range(len(enter)):
        for j in range(len(enter[i])):
            somme = somme + enter[i][j]
        
    return somme



def checkIfExist(index,arr):
    result = False
    
    for x in range(0,len(arr)-1):
        if(index == arr[x]):
            result = True
        
    
    
    return result



def createHistogramme(image):
    res = [0]*256
    
    for i in range(len(image)):
        for j in range(len(image[i])):
            index =int(image[i][j])
            # print("index",  index)
            res[index]=res[index] +1
    return res
    

def newMapping(image,ref):
    M = [0]*256;
    Result = [0]*256;
    hist1 = createHistogramme(image)
    hist2 = ref
    cdf1 = [0]*256;
    cdf2 = [0]*256;
    somme1 =0
    somme2=0
    for i in range(len(cdf1)):
        somme1 = somme1 + hist1[i]
        cdf1[i] = somme1
        
    for i in range(len(cdf2)):
        somme2 = somme2 + hist2[i]
        cdf2[i]  = somme2    
        
    print("Yep  ", cdf1[5] , "   ",cdf2[5] )
   
    
    for idx in range(0,256):
        
        if(cdf1[idx] >= cdf2[idx]):
            ind = cdf1[idx] - cdf2[idx]
        else:
            ind = cdf2[idx] - cdf1[idx]
        print("EEEEEEEEEEEEEEEEEEEEEEEEEEEEEE ",ind)    
            
            
        
        
        M[idx] = ind-1;
    
    
    # Now apply the mapping to get first image to make
    # the image look like the distribution of the second image
    return M

#################################                Debut                ###################################
#################################           Parametres                ###################################
from google.colab import drive
drive.mount('/content/gdrive')

#image

#imageColor = cv.imread("/content/gdrive/My Drive/grain6.jpg")

#get gray version

#image1 = cv.cvtColor(imageColor, cv.COLOR_BGR2GRAY)

# nombre de generations

generations = 300

# mutation rate 

mr = 5

#################################           Parametres                ###################################


###################### Solution ##################
#Solution = fullGeneProbablistic(image1,mr,generations)
###################### Solution ##################


#imageResultat = MatchGoFull(image1,Solution)

#figOtsu, (axOtsu) = plt.subplots(nrows=1, ncols=1) # two axes on figure

#axOtsu.imshow(imageResultat, cmap ="gray")



#Input = "/content/gdrive/My Drive/DermMelGray/NotMelanoma/ISIC_0024"
#Output = "/content/gdrive/My Drive/DermMelGenetique/NotMelanoma/ISIC_0024"
#OutputG = "/content/gdrive/My Drive/DermMelGray/NotMelanoma/ISIC_00"

Input = "/content/gdrive/My Drive/Data Final/Gray/Melanoma/AUG_0_"
Output = "/content/gdrive/My Drive/Data Final/Genetique/Melanoma/AUG_0_"

for i in range(422,447): 
    InputName = Input + str(i) + ".jpeg"
    OutputName = Output + str(i) + ".jpeg"
    #OutputGName = OutputG + str(i) + ".jpg"
    image = cv.imread(InputName,0)
    if image is None:
        print("image non disponible "+ str(i))
    else:
        print("Start "+ str(i))
        BestHist = fullGeneProbablistic(image,mr,generations)
        print("End")
        #print("Meilleur Histogramme:" + str(BestHist))
        BestImg = MatchGoFull(image, BestHist)
        cv.imwrite(OutputName, BestImg)
        #cv.imwrite(OutputGName, image)
    

print("Fin")




#################################                Fin                   ###################################