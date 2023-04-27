
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>
#include <errno.h>
#include <stdarg.h>
#include <ctype.h>
#include <stdint.h>
#include <math.h>

#include <limits.h>



typedef struct KMClust {
    int nclust;   // number of clusters, the "k" in kmeans
    int dim;      // dimension of features for data
    float* features; // 2D indexing for individual cluster center features
    int* counts;
} KMClust;

typedef struct KMData { // ndata is the number of features we have 
    int ndata;    // count of data
    int dim;      // dimension of features for data
    float* features; // pointers to individual features also a 2D array storing all the elements for every feature
    int* assigns; // cluster to which data is assigned
    int* labels;  // label for data if available
    int nlabels;  // max value of labels +1, number 0,1,...,nlabel0
} KMData;

#ifndef KMEANS_UTIL_H
#define KMEANS_UTIL_H
int filestats(char* filename, long* npoints, long* nfeatures);
int intMax(int* arr, int n);
KMData* kmdata_load(char* filename);
void helper_print(float* arr, int n, int m);
KMClust* kmclust_new(int nclust, int dim);
void save_pgm_files(KMClust* clust, char* filename);
#endif // KMEANS_UTIL_H

