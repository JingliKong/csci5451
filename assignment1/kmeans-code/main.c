#include "filestats.c" 
#include <stdlib.h>
#include <string.h>
#include <limits.h>
#include <math.h>

#define LINELENGTH 3139

typedef struct  {
    int ndata; // set to 0 
    int dim; // set to 0  
    float **features;
    int *assigns;
    int *labels;  
    int *nlabels;                    
} KMData_t;

typedef struct  {
    int nclust; // set to 0  
    int dim; // set to 0 
    float **features; 
    int *counts;  //is counts type int?
} KMClust_t;


KMData_t* kmdata_load(char *datafile) {
    KMData_t *data = malloc(sizeof(KMData_t)); 
    
    data->ndata = 0;  
    data->dim = 0;
    ssize_t tot_tokens = 0; //number of tokens in datafile 
    ssize_t tot_lines = 0;  // number of lines in datafile

    int fileStats = filestats(datafile, &tot_tokens, &tot_lines); 
    
    FILE *f = fopen(datafile, "r");
    if(f == NULL){
        printf("Failed to open file");
        return NULL; 
    }
    size_t currentRead; // current line of file we are reading 
    char buffer[LINELENGTH]; 
    float tokens[tot_tokens]; 
    int currentLine = 0; 
    int tokenIndex = 0; 
    while (currentLine < tot_lines) {
        data->ndata += 1; 
        currentRead = fread(buffer, 1, LINELENGTH, f); 
        for (int i = 0; i < LINELENGTH; i++) {
            if (isspace(buffer[i]) == 0) { // if the current character is not a white space 
                tokens[tokenIndex] = atof(buffer[i]); // add character to our tokens list
            }
        }
        // appending features 
        float *feats = malloc(sizeof(float) * (tot_tokens - 2)); 
        for (int i = 2; i < tot_tokens; i++) {
            feats[i-2] = tokens[i]; // adding features to feature array 
        }
        data->features = feats; 
        currentRead++; 
    }
    fclose(f); 
    return data; 
}

//not finished yet
KMClust_t* kmclust_new(nclust, dim) {
    KMClust_t* clust = malloc(sizeof(KMClust_t)); 
    clust->nclust = nclust;
    clust->dim = dim;
    // float* features = malloc();
    // int* counts = malloc();

    
    for (int c = 0; c < nclust; c++){
        
    }

    return clust;
}


int main(int argc, char *argv[]) {
    if (argc < 3){
        exit(-1);
    }

    char* datafile;
    int nclust;
    char* savedir = ".";
    int MAXITER = 100;
    
    strcpy(datafile, argv[1]);
    nclust = atoi(argv[2]);

    if (argc > 3){
        strcpy(savedir, argv[3]);
        mkdir(savedir);
    }

    if (argc > 4){
        MAXITER = atoi(argv[4]);
    }

    printf("datafile: %s\n", datafile);
    printf("nclust: %d\n", nclust);
    printf("savedir: %s\n", savedir);

    KMData_t* data = kmdata_load(datafile);
    KMClust_t* clust = kmclust_new(nclust, data->dim);
    
    printf("ndata: %d\n", data->ndata);
    printf("dim: %d\n\n", data->dim);

    for (int i = 0; i < data->ndata; i++){
        int c = i % clust->nclust;
        data->assigns[i] = c;
    }

    for (int c = 0; c < clust->nclust; c++){
        //is this supposed to be int???
        int icount = data->ndata / clust->nclust;
        int extra = (c < (data->ndata % clust->nclust)) ? 1 : 0;
        clust->counts[c] = icount + extra;
    }
    
    // Main Algorithm
    int curiter = 1;
    int nchanges = data->ndata;
    printf("==CLUSTERING: MAXITER {%d}==\n", MAXITER);
    printf("ITER NCHANGE CLUST_COUNTS\n");

    while ((nchanges > 0) && (curiter <= MAXITER)){

        //DETERMINE NEW CLUSTER CENTERS
        //reset cluster centers to 0.0
        for (int c = 0; c < clust->nclust; c++){ 
            for (int d = 0; d < clust->dim; d++){
                clust->features[c][d] = 0.0;
            }
        }

        //sum up data in each cluster
        for (int i = 0; i < data->ndata; i++){
            int c = data->assigns[i];
            for (int d = 0; d < clust->dim; d++){
                clust->features[c][d] += data->features[i][d];
            }
        }

        // divide by ndatas of data to get mean of cluster center
        for (int c = 0; c < clust->nclust; c++){
            if (clust->counts[c] > 0){
                for (int d = 0; d < clust->dim; d++){
                    clust->features[c][d] = clust->features[c][d] / clust->counts[c];
                }
            }
        }
        
        // DETERMINE NEW CLUSTER ASSIGNMENTS FOR EACH DATA
        for (int c = 0; c < clust->nclust; c++){
            clust->counts[c] = 0;
        }

        nchanges = 0;
        for (int i = 0; i < data->ndata; i++){
            int best_clust = INT_MIN;
            float best_distsq = INFINITY;
            for (int c = 0; c < clust->nclust; c++){
                float distsq = 0.0;
                for (int d = 0; d < clust->dim; d++){
                    float diff = data->features[i][d] - clust->features[c][d];
                    distsq += diff*diff;
                }
                if (distsq < best_distsq){
                    best_clust = c;
                    best_distsq = distsq;
                }
            }
            clust->counts[best_clust] += 1;
            if (best_clust != data->assigns[i]){
                nchanges += 1;
                data->assigns[i] = best_clust;
            }
        }


        // Print iteration information at the end of the iter
        printf("%d: %d |", curiter, nchanges);
        for (int c = 0; c < nclust; c++){
            printf(" %d", clust->counts);
        }
        printf("\n");
        curiter += 1;
    }

    // Loop has converged
    if (curiter > MAXITER){
        printf("WARNING: maximum iteration %d exceeded, may not have conveged", MAXITER);
    } else {
        printf("CONVERGED: after %d iterations", curiter);
    }
    print("\n");

    //=================================
    // CLEANUP + OUTPUT

    // CONFUSION MATRIX
}
