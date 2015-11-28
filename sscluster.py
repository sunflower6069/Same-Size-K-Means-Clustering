#!/bin/python

### Same Size Clustering ###
###
### This is a variation of the k-means clustering that produces equally sized clusters.
### The algorithm consists of two phases:
### 1. Initialization:
###     Compute the desired cluster size: n/k
###     Initialize means with k-means
###     Order points by the biggest benefit (delta distance) of best over worst assignment
###     Assign points to their prefered cluster until the cluster is full, then resort remaining
###     objects by excluding the clusters that are full.
### 2. Refinement of clustering:
###     This is done in an iterative fashion, until there is no change in clustering or the max
###     number of iterations have been reached.
###     Interation:
###         Compute current cluster means.
###         For each object, compute the distance to the cluster means.
###         Sort elements based on the delta of the current assignment and the alternative best 
###         possible assignment, sort in descending order.
###         For each element by priority:
###             For each cluster to whom it doesn't belong, by element gain:
###                 if there is an element on the transfer list of the cluster, and swapping the 
###                 two element yields improvement, swap the two elements;
###             End For
###             If the element is not swapped, add to outgoing transfer list
###         End For
###         If no transfers were done, or max iteration was reached, terminate.
###     Since any transfer must decrease variance, thus the clustering will converge.
### 
### Author: Xin Cindy Yan
### Credit: The algorithm is adapted from the java script by Erich Schubert as on the ELKI webpage:
###         http://elki.dbs.ifi.lmu.de/wiki/Releases
### Date:   03/27/2015

import numpy as np
import random
import mykmns
import readdata
import sys

#------------Input---------------
n = 2000 # total number of points
cluster_size = 4 # size of each cluster
k = (n+cluster_size-1) // cluster_size # total number of clusters
infl = 't4p2k.pdb' # the input pdb file with all dataset
outfl = 'cluster2k.pdb' # the output pdb file with the clustering information
boxl = np.array([39.0571, 39.0571, 39.0571])
max_iter = 100

# Choose initial means with K-means method
def ChooseInitialMeans(X):
    means, clusters = mykmns.kmeans_main(X, k)
    return means

# Returns the distance between an object x and a cluster mean c
def GetDist(x, c):
    dist = np.linalg.norm(x-c-boxl*np.around((x-c)/boxl))
    return dist

# Returns the distance matrix of each object to each cluster mean,
# and each row (i'th obj) of the matrix is ordered so that the dist are in ascending order
def SetDistMat(X, means):
    distmat_dtype = [('key',int), ('dist',float)]
    distmat = np.empty((n,k),dtype=distmat_dtype)
    for i in range(n):
        distmat[i,:] = [(c[0], GetDist(X[i], c[1])) for c in enumerate(means)]
        distmat[i,:] = np.sort(distmat[i,:], order='dist')
    return distmat

# Returns the priority list in which objects are ordered by asceding benefit 
# of best over worst assignment
def Get_plst(assigned, distmat, full):
    plst = []
    for i in range(n):
        if (i not in assigned):
            j = 0
            while j<k:
                if (not full[distmat[i,j][0]]):
                    bestkey = distmat[i,j][0]
                    mindist = distmat[i,j][1]
                    break
                else:
                    j += 1
            for j in range(k-1,-1,-1):
                if (not full[distmat[i,j][0]]):
                    maxdist = distmat[i,j][1]
                    break
            plst.append((i, bestkey, maxdist-mindist))
    plst.sort(key=lambda t:t[2])
    return plst

def InitialAssignment(distmat):
    clusters = {}
    full = np.zeros(k,dtype=bool) # a boolean array that records which clusters are full
    assigned = [] # a list of objects who has been assigned to a cluster
    plst = Get_plst(assigned, distmat, full)
    while (len(plst)):
        temp = plst.pop()
        try:
            if (len(clusters[temp[1]])<cluster_size):
                clusters[temp[1]].append(temp[0])
                assigned.append(temp[0])
            else:
                full[temp[1]] = True
                plst = Get_plst(assigned, distmat, full)
        except KeyError:
            clusters[temp[1]] = [temp[0]]
            assigned.append(temp[0])
    return clusters

def CalcMeans(X, oldmeans, clusters):
    means = np.zeros((k,3))
    keys = sorted(clusters.keys())
    for key in keys:
        for i in clusters[key]:
            means[key] += X[i]-boxl*np.around((X[i]-oldmeans[key])/boxl)
        means[key] /= len(clusters[key])
        means[key] -= boxl*np.around(means[key]/boxl)
    return means

def SortObj(X, clusters, means, distmat):
    objlst = [] # list of objects ordered in asceding delta of the current 
                # assignment and the best possible alternate assignment
    keys = sorted(clusters.keys())
    for key in keys:
        for i in clusters[key]:
            currdist = GetDist(X[i],means[key])
            mindist = distmat[i,0][1]
            objlst.append((i, key, currdist-mindist))
    objlst.sort(key=lambda t:t[2], reverse=True)
    return objlst

def Transfer(obj, clufrom, cluto, clusters):
    clusters[clufrom].remove(obj)
    clusters[cluto].append(obj)
    return clusters

def WriteResult(file, X, means, clusters):
    fl = open(file, "w")
    fl.write("REMARK 0   Result of k-means cluster:\n")
    fl.write("REMARK 1   k = %d\n"%k)
    fl.write("TER\n")
    keys = sorted(clusters.keys())
    i = 1
    for key in keys:
        for obj in clusters[key]:
            fl.write("ATOM  %5d OW   WAT  %4d    %8.3f%8.3f%8.3f      %6.2f\n"\
                    %(i, i, X[obj][0], X[obj][1], X[obj][2], key)) 
            fl.write("TER\n")
            i = i + 1
    for c in enumerate(means):
        fl.write("HETATM%5d Ar   HET  %4d    %8.3f%8.3f%8.3f      %6.2f\n"\
                %(i, i, c[1][0], c[1][1], c[1][2], c[0]))
        fl.write("TER\n")
        i = i + 1
    fl.write("END\n")
    fl.close()
    return

# This function will perform statistical analysis to the clustering results
def ClusterStat(X, means, clusters):
    # Average distance between means
    means_avg = 0.
    for i in range(k-1):
        for j in range(i+1,k):
            means_avg += GetDist(means[i], means[j])
    means_avg /= (k*(k-1)/2.)
    # Average distance between obj and mean in a cluster
    obj2mean_avg = np.zeros(k)
    # Variance of the distances between obj and mean in a cluster
    obj2mean_var = np.zeros(k)
    keys = sorted(clusters.keys())
    for key in keys:
        for i in clusters[key]:
            obj2mean = GetDist(X[i], means[key])
            obj2mean_avg[key] += obj2mean 
            obj2mean_var[key] += obj2mean*obj2mean
        obj2mean_avg[key] /= len(clusters[key])
        obj2mean_var[key] /= len(clusters[key])
        obj2mean_var[key] = np.sqrt(obj2mean_var[key])
    # Average within cluster distances between objects
    winclu_avg = np.zeros(k)
    # Average of within cluster distances of all clusters
    winclu_grandavg = 0.
    for key in keys:
        for i in clusters[key]:
            x = X[i]
            for j in clusters[key]:
                if j>i:
                    winclu_avg[key] += GetDist(x, X[j])
        s = len(clusters[key])
        winclu_avg[key] /= (s*(s-1)/2) 
        winclu_grandavg += winclu_avg[key]
    winclu_grandavg /= k
    # write the summary 
    print("average distance among means: %f"%means_avg)
    #print("average distance from objects to the mean of a cluster:")
    #for i in range(k):
    #    print("cluster %i: %f"%(i, obj2mean_avg[i]))
    #print("variance of distances from objects to the mean of a cluster:")
    #for i in range(k):
    #    print("cluster %i: %f"%(i, obj2mean_var[i]))
    #print("within-cluster average distances:")
    #for i in range(k):
    #    print("cluster %i: %f"%(i, winclu_avg[i]))
    print("grand average of within-cluster average distances: %f"%winclu_grandavg)
    return 

def main():
    # Set up the database of objects
    X = readdata.read_data(infl)
    # Choose initial means with K-means
    means = ChooseInitialMeans(X)
    # Set up initial clusters
    distmat = SetDistMat(X, means) 
    clusters = InitialAssignment(distmat) 
    ## debug code
    #keys = sorted(clusters.keys())
    #for key in keys:
    #    print("cluster %i:"%key)
    #    print(clusters[key])
    ## end of debug
    # Iteration step
    for iter in range(max_iter):
        active = 0 # indicate the number of transfers in the current iteration
        tranlst = (-1)*np.ones(k, dtype='int') # set up transfer list for each cluster
        # Compute the cluster means
        oldmeans = means.copy()
        means = CalcMeans(X, oldmeans, clusters)
        # Get statistics about the clustering
        #ClusterStat(X, means, clusters)
        ## debug code
        #print("old means:")
        #print(oldmeans)
        #print("new means:")
        #print(means)
        ## end of debug
        # For each object, compute the distances to the cluster means
        distmat = SetDistMat(X, means)
        # Sort objects based on the delta of the current assignment and the best 
        # possible alternate assignment
        objlst = SortObj(X, clusters, means, distmat)
        ##debug code
        #print(objlst)
        ##return
        #end of debug
        # For each element by prioty:
        while (len(objlst)):
            (i, key, temp) = objlst.pop()
            obj2key = GetDist(X[i], means[key])
            transferred = False #record if any transfering has occured to i 
            if (key == distmat[i,0][0]):
                ##debug
                #print("%i is already the opt cluster for obj %i. no transfer"%(clu, i))
                ##end of debug
                continue
            # For each other clusters by element gain:
            else:
                for j in range(k):
                    clu = distmat[i,j][0] # the key of another cluster
                    objgain = obj2key - distmat[i,j][1] # gain by transfering i from cluster key to clu
                    if (clu==key): # already in the cluster
                        continue
                    if (len(clusters[clu]) < cluster_size):
                        active += 1
                        transferred = True
                        clusters = Transfer(i, key, clu, clusters)
                        ##debug
                        #print("cluster %i not full. transfer obj %i from cluster %i to it."%(clu, i, key))
                        ##end of debug
                        break
                    elif (tranlst[clu] != -1): # if the tranlst of another cluster is not empty
                        # distance between the obj in the tranlst and the current cluster
                        tran2key = GetDist(X[tranlst[clu]], means[key])
                        tran2clu = GetDist(X[tranlst[clu]], means[clu])
                        # gain by transfering the obj in tranlst from cluster clu to key
                        trangain = tran2clu - tran2key
                        if (objgain + trangain > 0): # transfer if the sum of gains are positive, ie net gain
                            active += 2
                            transferred = True
                            clusters = Transfer(i, key, clu, clusters)
                            clusters = Transfer(tranlst[clu], clu, key, clusters)
                            ##debug
                            #print("obj %i is transfered from cluster %i to %i"%(i, key, clu))
                            #print("obj %i is transfered from cluster %i to %i"%(tranlst[clu], clu, key))
                            #print("objgain: %f, trangain: %f"%(objgain, trangain))
                            ##end of debug
                            tranlst[clu] = -1 # reset the tranlst to empty
                            break
                if (not transferred):
                    tranlst[key] = i
                    ##debug
                    #print("add obj %i in cluster %i to the transfer list"%(i, key))
                    ##end of debug
        # nothing is transferred during this iteration, return the clustering result
        if (not active):
                break
        #debug code
        print("number of transfers in iter %i: %i\n"%(iter+1, active))
        #end of debug
    print("K-means clustering converged in %d iterations!\n"%(iter+1))
    # Output the clustering results
    WriteResult(outfl, X, means, clusters)
    ClusterStat(X, means, clusters)
    return(0)

if __name__ == "__main__":
    sys.exit(main())
