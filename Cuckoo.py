# -*- coding: utf-8 -*-
"""
Created on Fri Oct 23 10:46:09 2020

@author: AHA
"""

import numpy as np
import matplotlib.pyplot as plt
from skimage.exposure import match_histograms
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
# Load image
# C:/Users/AHA/.spyder-py3/amine.jpg
import math 

import random
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


# formater Haar avec des valeur entiere entre 0 et 255



########################################## Pure Cuckoo ##########################################



# get N worst cuckoos

def getNWorstVCuck(image, array, pop, nouveaunee, fitnesses):
    res = [-1]*(nouveaunee)
    worst = array[0]
    worstValue = globalFitnessFunction(MatchGoFull(image, array[0]))
    saveidex=0
    #optimisation precalculer les fitnesse
    precalculeValue = [0]*len(array)
    
    precalculeValue = fitnesses
    
    for cuck in  range(0,pop):    
    
    
        
        if(precalculeValue[cuck]<=worstValue):
             
                                   
             worst = array[cuck]
             worstValue = precalculeValue[cuck]
             saveidex = cuck
    
    res[0]=saveidex
    
    #true deb
    
    for v in range (0,nouveaunee):
        for cuck in  range(0,pop):
             
            if(checkIfExist(cuck,res)==False):
                
                 
                worst = array[cuck]
                worstValue = precalculeValue[cuck]
                saveidex = cuck
    
                    
    
        i=0
        while(i<pop):
            if(checkIfExist(i,res)==False):
    
                cond = precalculeValue[i]<worstValue
                if(cond):
                    
                    worst = array[i]
                    worstValue = precalculeValue[i]
                    saveidex = i
            i = i +1
        
        res[v]=saveidex       
            
    
    
    
    return res

# get N best Cuckoos
    
def getNBestVCuck(image, array, pop, nouveaunee, fitnesses):
    res = [-1]*(nouveaunee)
    worst = array[0]
    worstValue = fitnesses[0]
    saveidex=0
    #optimisation precalculer les fitnesse
    precalculeValue = [0]*len(array)
    
    precalculeValue = fitnesses
    
    for cuck in  range(0,pop):    
    
    
        
        if(precalculeValue[cuck]>=worstValue):
             
                                   
             worst = array[cuck]
             worstValue = precalculeValue[cuck]
             saveidex = cuck
    
    res[0]=saveidex
    
    #true deb
    
    for v in range (0,nouveaunee):
        for cuck in  range(0,pop):
             
            if(checkIfExist(cuck,res)==False):
                
                 
                worst = array[cuck]
                worstValue = precalculeValue[cuck]
                saveidex = cuck
    
                    
    
        i=0
        while(i<pop):
            if(checkIfExist(i,res)==False):
    
                cond = precalculeValue[i]>worstValue
                if(cond):
                    
                    worst = array[i]
                    worstValue = precalculeValue[i]
                    saveidex = i
            i = i +1
        
        res[v]=saveidex       
            
    
    
    
    return res


# matation best version

def mutationXX2(crom, chance,max):
    res = crom.copy()

    

    for i in range(len(crom)):
        if random.randint(0, 100) < chance:
            res[i] = random.randint(0, max)
            
        
    
    
    return res
    

# get two random points

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
            
# generate child 1      

def reproDuctionChild1TwoPoints(cromoA, cromoB, rand, where1,where2):
    res  = [0]*256
    
    if rand == 0:
        
        for i in range(len(res)):
            
            if i <= where1  or i > where2:
                res[i] = cromoA[i]
            else:
                res[i] = cromoB[i]
    
    return res

# generate child 2
                
def reproDuctionChild2TwoPoints(cromoA, cromoB, rand, where1,where2):
    res  = [0]*256
    
    if rand == 0:
        
        for i in range(len(res)):
            
            if i <= where1  or i > where2:
                res[i] = cromoB[i]
            else:
                res[i] = cromoA[i]
    return res  



# fast histo cumul
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



# get quantil for the mapping

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


# calucler distence inter quartil

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
    #print("dist ", dist)
    
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
        
def genererSolution(x,y):
    sol = [0]*256
    i=0
    while i < (x*y):
        rng = random.randint(0, 255)
        sol[rng] = sol[rng] + 1
        i = i +1
    return sol



def genererSolutionV2(x,y):
    sol = [0]*256
    somme = x*y
    
    while somme > 0:
        rng = random.randint(0, 255)
        new = random.randint(0, 50 )
        sol[rng] = sol[rng] + new
        somme = somme - new
        
    return sol

def genererSolutionV3(x,y,max):
    sol = [0]*256
    
    for indX in range (len(sol)):
        sol[indX] = random.randint(0, max) 
        
    return sol
        
        
def MatchGoFull(image,source):
    matched = histoMatchV3(image,source)
    return matched





def fullCuckoo(image, loops):
    x=0
    y=0
    Archivage = [0]*loops
    result = [0]*256
    best = [0]*256
    bestValue = 0
    bestIndex = 0
    
    worst = [0]*256
    worstValue = 0
    worstIndexs = [-1]*5
    
    
    populationN = 20
    
    
    i = 0
    population = numpy.empty(populationN, dtype=object)
    saveFitness = numpy.empty(populationN, dtype=object)
    #generer une population de 20 solutions
    while (i<populationN):
        population[i] = genererSolutionV3(x,y,200)
        i = i +1
        

    #condition d'arret
    z = 0
    condition = z < loops
    while(condition):
        
        if ( z == 0):
            
            for cuck in  range(0,populationN):
    
                matched = MatchGoFull(image, population[cuck])
    
                currentFit = globalFitnessFunction(matched)
                saveFitness[cuck] = currentFit
                if(currentFit > bestValue):
                    bestValue = currentFit
                    best = population[cuck].copy()
                    bestIndex = cuck
            
            
        #walk
        for cuck in  range(0,populationN):
            #mod
            
            
            #optiwalk Test
            # population[cuck] = randomWalkV3(population[cuck], x, y, 40,image)
            
            res =  population[cuck].copy()
    
            
            
            maximax = 200 
            
            plsm = random.uniform(-1, 1)
            indexToChange = random.randint(0, 255)
            res[indexToChange] = abs(res[indexToChange] + ( plsm * random.randint(0, maximax)))
            
            plsm = random.uniform(-1, 1)
            indexToChange = random.randint(0, 255)
            res[indexToChange] = abs(res[indexToChange] + ( plsm * random.randint(0, maximax)))
            
            plsm = random.uniform(-1, 1)
            indexToChange = random.randint(0, 255)
            res[indexToChange] = abs(res[indexToChange] + ( plsm * random.randint(0, maximax)))
            
            plsm = random.uniform(-1, 1)
            indexToChange = random.randint(0, 255)
            res[indexToChange] = abs(res[indexToChange] + ( plsm * random.randint(0, maximax)))
            
            plsm = random.uniform(-1, 1)
            indexToChange = random.randint(0, 255)
            res[indexToChange] = abs(res[indexToChange] + ( plsm * random.randint(0, maximax)))
            
            plsm = random.uniform(-1, 1)
            indexToChange = random.randint(0, 255)
            res[indexToChange] = abs(res[indexToChange] + ( plsm * random.randint(0, maximax)))
            
            plsm = random.uniform(-1, 1)
            indexToChange = random.randint(0, 255)
            res[indexToChange] = abs(res[indexToChange] + ( plsm * random.randint(0, maximax)))
            
            plsm = random.uniform(-1, 1)
            indexToChange = random.randint(0, 255)
            res[indexToChange] = abs(res[indexToChange] + ( plsm * random.randint(0, maximax)))
            
            plsm = random.uniform(-1, 1)
            indexToChange = random.randint(0, 255)
            res[indexToChange] = abs(res[indexToChange] + ( plsm * random.randint(0, maximax)))
            
            plsm = random.uniform(-1, 1)
            indexToChange = random.randint(0, 255)
            res[indexToChange] = abs(res[indexToChange] + ( plsm * random.randint(0, maximax)))
                
            fitnessOld =  saveFitness[cuck]
            fitnessNew = globalFitnessFunction(MatchGoFull(image,res))
            
            if fitnessNew > fitnessOld :
                population[cuck] = res
                saveFitness[cuck] = fitnessNew
                
                    
            
            
            #otiwalk test
        
        #optimisation precaluculer les piftness
        
        
            
            
        # evaluate
        for cuck in  range(0,populationN):

            

            currentFit = saveFitness[cuck]
            
            # print("C: ", currentFit, " I: ",cuck, " Gen: ", z)
            if(currentFit > bestValue):
                bestValue = currentFit
                best = population[cuck].copy()
                bestIndex = cuck
        
        #print("Best Value: ", bestValue, "Best Index: ",bestIndex, " In Generation: ", z)
        Archivage[z] = bestValue
        
       
       
        if z % 5 ==0 :
       ################ hybride ################################################   
            #Gen parrents
            
            parent1 = bestIndex
            
            parent2 = 0
            parent3 = 0
            
            # bestparent2 = saveFitness[0]
            # bestparent3 = saveFitness[0]
            # for cuck in  range(0,populationN):
                
            #     currentFit = saveFitness[cuck]
                 
            
            # # print("C: ", currentFit, " I: ",cuck, " Gen: ", z)
            #     if(currentFit > bestparent2 and cuck != parent1  ):
            #         bestparent2 = currentFit

            #         parent2 = cuck

            # for cuck in  range(0,populationN):
                
            #     currentFit = saveFitness[cuck]
                 
            
            # # print("C: ", currentFit, " I: ",cuck, " Gen: ", z)
            #     if(currentFit > bestparent3 and cuck != parent2  ):
            #         bestparent3 = currentFit

            #         parent3 = cuck

            parentZ = getNBestVCuck(image,population,populationN,3,saveFitness)

            
            parent1 = parentZ[0]
            
            parent2 = parentZ[1]
            parent3 = parentZ[2]


       
            # XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX eliminate XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX
                
        
            
            worstIndexs = getNWorstVCuck(image,population,populationN,5,saveFitness)
            #print("Worst: ",worstIndexs )
            #print("parents: ",parent1, " ", parent2," ",parent3," " )
            #true elimination
            wheres = gen2Wheres()
            where1 = wheres[0]
            where2 = wheres[1]
            

            population[worstIndexs[0]] = reproDuctionChild1TwoPoints(population[parent1], population[parent3], 0,where1,where2)
            population[worstIndexs[1]] = reproDuctionChild2TwoPoints(population[parent1], population[parent3],0,where1,where2)
            
            
            wheres = gen2Wheres()
            where1 = wheres[0]
            where2 = wheres[1]
            
            population[worstIndexs[2]] =  reproDuctionChild1TwoPoints(population[parent1], population[parent2], 0,where1,where2)
            population[worstIndexs[3]] =  reproDuctionChild2TwoPoints(population[parent1], population[parent2], 0,where1,where2)
            wheres = gen2Wheres()
            where1 = wheres[0]
            where2 = wheres[1]
            # population[worstIndexs[4]] = reproDuctionChild1TwoPoints(population[parent2], population[parent3], 0,where1,where2)
            
            
            population[worstIndexs[4]] = genererSolutionV3(x,y,200)
            
            minMaxValue = 200
            mutation = 1
            population[worstIndexs[0]] = mutationXX2(population[worstIndexs[0]], mutation,minMaxValue)
            population[worstIndexs[1]] = mutationXX2(population[worstIndexs[1]], mutation,minMaxValue)
            population[worstIndexs[2]] = mutationXX2(population[worstIndexs[2]], mutation,minMaxValue)
            population[worstIndexs[3]] = mutationXX2(population[worstIndexs[3]], mutation,minMaxValue)
            # population[worstIndexs[4]] = mutationXX2(population[worstIndexs[4]], mutation,minMaxValue)
            
            # print("fitnes ", saveFitness)
            fit1 = globalFitnessFunction(MatchGoFull(image,population[worstIndexs[0]]))
            fit2 = globalFitnessFunction(MatchGoFull(image,population[worstIndexs[1]]))
            fit3 = globalFitnessFunction(MatchGoFull(image,population[worstIndexs[2]]))
            fit4  = globalFitnessFunction(MatchGoFull(image,population[worstIndexs[3]]))
            fit5 = globalFitnessFunction(MatchGoFull(image,population[worstIndexs[4]]))
            

            
            saveFitness[worstIndexs[0]] = fit1
            saveFitness[worstIndexs[1]] = fit2
            saveFitness[worstIndexs[2]] = fit3
            saveFitness[worstIndexs[3]] = fit4
            saveFitness[worstIndexs[4]] = fit5
            # print("fitnes apres", saveFitness)
            worstIndexs = [-1]*5
            

        ################ hybride ################################################
     
            # XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX eliminate XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX
                
            # check worst         XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX
            # worst = population[0]
            # worstValue = saveFitness[0]
            # saveidex=0
            # worstValue = getNWorstVCuck(image,population,populationN,4,saveFitness)
            # print("Worst: ",worstValue )
            
            # #true elimination
            # globalFitnessFunction(MatchGoFull(image,population[cuck]))
            # population[worstIndexs[0]] = genererSolutionV3(x,y,200)
            # population[worstIndexs[1]] = genererSolutionV3(x,y,200)
            # population[worstIndexs[2]] = genererSolutionV3(x,y,200)
            # population[worstIndexs[3]] = genererSolutionV3(x,y,200)
            # population[worstIndexs[4]] = genererSolutionV3(x,y,200)
            
            
            
            # saveFitness[worstIndexs[0]] = globalFitnessFunction(MatchGoFull(image,population[worstIndexs[0]]))
            # saveFitness[worstIndexs[1]] = globalFitnessFunction(MatchGoFull(image,population[worstIndexs[1]]))
            # saveFitness[worstIndexs[2]] = globalFitnessFunction(MatchGoFull(image,population[worstIndexs[2]]))
            # saveFitness[worstIndexs[3]] = globalFitnessFunction(MatchGoFull(image,population[worstIndexs[3]]))
            # saveFitness[worstIndexs[4]] = globalFitnessFunction(MatchGoFull(image,population[worstIndexs[4]]))
            # worstIndexs = [-1]*5
            

        
 
               

        #   # check worst         XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX  
        
                 
              
        #  # XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX eliminate XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX
            
         
            
         
            
         
            
         
            
         
        #print("")
        z = z +1
        
        condition = z < loops
        # print("condition = ", condition)
    result = population[0]
    #gestBest
    
    indexBestFinale = bestIndex
    bestValueFinale = bestValue
    print("YYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYY" )
    print("this is the best value: ", bestValue)
    for cuck in  range(0,populationN):
        if(saveFitness[cuck]>bestValueFinale):
            indexBestFinale = cuck
            bestValueFinale = saveFitness[cuck]
            best = population[cuck]
            

    
    result = best                                       
    
    #plt.plot(Archivage)
    #plt.xlabel('Generations')
    #plt.ylabel('Qualité du contraste')
    #plt.show()
    return result












########################################## Pure Cuckoo ##########################################



def checkIfExist(index,arr):
    result = False
    
    for x in range(0,len(arr)-1):
        if(index == arr[x]):
            result = True
        
    
    
    return result



############################ copie #######################




def createHistogramme(image):
    res = [0]*256
    
    for i in range(len(image)):
        for j in range(len(image[i])):
            index =int(image[i][j])
            # print("index",  index)
            res[index]=res[index] +1
    return res
    



#      return c 
 


##################### fonctions pour la fitness function #####################



def globalFitnessFunction(image):
    result =0
    
  
   
    log1 =math.log(filtreSobel(image),10)
    
    # phase1 = math.log(math.log(filtreSobel(image)))
    phase1 = math.log(log1,10)
    phase2 = newContourOTSU(image) / (len(image) * len(image[1]))
    phase3 = skimage.measure.shannon_entropy(image)
    result = phase1 * phase2 * phase3
    return result


    # result =0
    
  
   
    # log1 =math.log(filtreSobel(image),10)
    
    # # phase1 = math.log(math.log(filtreSobel(image)))
    # phase1 = math.log(log1,10)
    # phase2 = calculerNombreDePixelsContour(image) / (len(image) * len(image[1]))
    # phase3 = skimage.measure.shannon_entropy(image)
    # result = phase1 * phase2 * phase3
    # return result





def filtreSobel(image):
    im = image.copy()
    
    im = im.astype('int32')
    dx = ndimage.sobel(im, 0)  
    dy = ndimage.sobel(im, 1) 
    mag = numpy.hypot(dx, dy)  
    mag *= 255.0 / numpy.max(mag) 
    return EIZsommeDesIntensite(mag)



# calculer le nombre de pixels des contours

def calculerNombreDePixelsContour(image):


    tigreOriginal = cv.cvtColor(image.astype('uint8') , cv.COLOR_BGR2RGB)
    tigreGris = cv.cvtColor(tigreOriginal, cv.COLOR_RGB2GRAY)

    # creater une image binnaire avec tresh hold X Y
    _, binary = cv.threshold(tigreGris,127,255, cv.THRESH_BINARY_INV)

    
    
    
    #  trouver les contour grace a l'image binnaire
    contours, hierarchy = cv.findContours(binary, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)


    return(len(contours))



# calculer le nombre de pixel dont l'intensité est superieur a 0

def countPixelsInContour(image):
    somme = 0
    for i in range(len(image)):
        for j in range(len(image[i])):
            if image[i][j] > 0:
                somme = somme + image[i][j]
        
    return somme




        

# calculer somme totales des intensité d'une imager 

def EIZsommeDesIntensite(enter):
    somme = 0
    for i in range(len(enter)):
        for j in range(len(enter[i])):
            somme = somme + enter[i][j]
        
    return somme








######################################## LEVY FLIGHT / Walk Testing ########################################


def levyFlight(solution,steps, dimx, dimy,beta):
    res = [0]*256
        

    
    theta = numpy.random.uniform(low=0, high= dimx*dimy)
    
    
    return res


def randomWalk(solution, dimx, dimy, stepSize):
    res = solution.copy()
    
    
    for j in range(1,stepSize):
        
        for i in range(0,255):
            save = res[i]
            plusMoins = random.randint(0, 1)
            if(plusMoins == 0):
                plusMoins = -1
            else:
                plusMoins = 1
            # solution[i] = solution[i] + random.gammavariate(0.5, 0.5)*plusMoins
            res[i] = res[i] + random.randint(0, 200)*plusMoins
            
            if( res[i] < 0):
                res[i] = save
            
        

           
        
    
    return res

# Walk true

def randomWalkV3(solution, dimx, dimy, stepSize,image):
    #copier solution
    res = solution.copy()
    
    indexToChange = random.randint(0, 255)
    
    res[indexToChange] = random.randint(0, 200)
    
    fitnessOld = globalFitnessFunction(MatchGoFull(image,solution))
    fitnessNew = globalFitnessFunction(MatchGoFull(image,res))

    if fitnessNew < fitnessOld :
        res = solution.copy()
  
    
  
    
  
    # res = solution.copy()
    # indexToChange = random.randint(0, 255)
    # res[indexToChange] = random.randint(0, 200)
    
        
    return res


def randomWalkV2(solution, dimx, dimy, stepSize,image):
    
    solutions1 = [0]*255
    solutions2 = [0]*255
    solutions3 = [0]*255
    solutions4 = [0]*255
    solutions5 = [0]*255
    solutions6 = [0]*255
    solutions7 = [0]*255
    solutions8 = [0]*255
    solutions9 = [0]*255
    
    solutions1 = randomWalk(solution, dimx, dimy, stepSize)
    solutions2 = randomWalk(solution, dimx, dimy, stepSize)
    solutions3 = randomWalk(solution, dimx, dimy, stepSize)
    solutions4 = randomWalk(solution, dimx, dimy, stepSize)
    solutions5 = randomWalk(solution, dimx, dimy, stepSize)
    solutions6 = randomWalk(solution, dimx, dimy, stepSize)
    solutions7 = randomWalk(solution, dimx, dimy, stepSize)
    solutions8 = randomWalk(solution, dimx, dimy, stepSize)
    solutions9 = randomWalk(solution, dimx, dimy, stepSize)
    
    
    bestFit = globalFitnessFunction(MatchGoFull(image,solutions1))
    best = solutions1
    currentFit = bestFit
    
    
    solArray =  numpy.empty(8, dtype=object)
    solArray[0] = solutions2
    solArray[1] = solutions3
    solArray[2] = solutions4
    solArray[3] = solutions5
    solArray[4] = solutions6
    solArray[5] = solutions7
    solArray[6] = solutions8
    solArray[7] = solutions9
    
    for sol in range (len(solArray)):
        currentFit = globalFitnessFunction(MatchGoFull(image,solArray[sol]))
        if  currentFit > bestFit:
            best = solArray[sol]
            bestFit = currentFit
        

    
    return best



######################################## LEVY FLIGHT / Walk Testing ########################################


# def EIZsommeDesIntensite(enter):
#     somme = 0
#     for i in range(len(enter)):
#         for j in range(len(enter[i])):
#             somme = somme + enter[i][j]
        
#     return somme





############################################################################


def formaterHaar(enter):
    for i in range(len(enter)):
        for j in range(len(enter[i])):
            if enter[i][j]<0:
                enter[i][j] = 0
            elif enter[i][j] > 255:
                enter[i][j] = 255
            else: enter[i][j] = int(enter[i][j])
            
    
           
    sortie = enter
    return sortie

        # if index < 0:
        #     index = 0
        # elif index > 255:
        #     index = 255
        # else: index = int(index)

original = pywt.data.camera()


original = cv.imread("badImage.jpg")








# histo harmonisaion


def histoHarm(x,y):
    result = [0]*255
    
    
    return result







def transformerLLenImage(image, ll):
    
    for i in range(len(ll)):
        for j in range(len(ll[1])):
            image[i][j] = ll[i][j]
    
    return image

 




#################################                Debut                ###################################
#################################           Parametres                ###################################
from google.colab import drive
drive.mount('/content/gdrive')

#image

#imageColor   = cv.imread("grain6.jpg")

#get gray version

#image1 = cv.cvtColor(imageColor, cv.COLOR_BGR2GRAY)


# nombre de generations

generations = 180

#################################           Parametres                ###################################


###################### Solution ##################
#resultatGen1 = fullCuckoo(image1, generations)
###################### Solution ##################


# imageResultat = MatchGoFull(image1,resultatGen1)

# figOtsu, (axOtsu) = plt.subplots(nrows=1, ncols=1) # two axes on figure

# axOtsu.imshow(imageResultat, cmap ="gray")


Input = "/content/gdrive/My Drive/Data Final/Gray/Melanoma/AUG_0_"
Output = "/content/gdrive/My Drive/Data Final/Cuckoo/Melanoma/AUG_0_"

for i in range(395,447): 
    InputName = Input + str(i) + ".jpeg"
    OutputName = Output + str(i) + ".jpeg"
    #OutputGName = OutputG + str(i) + ".jpg"
    image = cv.imread(InputName,0)
    if image is None:
        print("image non disponible "+ str(i))
    else:
        print("Start "+ str(i))
        BestHist = fullCuckoo(image, generations)
        print("End")
        #print("Meilleur Histogramme:" + str(BestHist))
        BestImg = MatchGoFull(image, BestHist)
        cv.imwrite(OutputName, BestImg)
        #cv.imwrite(OutputGName, image)
    

print("Fin")
 

#################################                Fin                   ###################################