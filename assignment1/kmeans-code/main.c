// #include "filestats.c" 
#include <stdlib.h>
#include <string.h>
#include <limits.h>
#include <math.h>
#include <stdio.h>

#define LINELENGTH 3139
#define MIN(X, Y) (((X) < (Y)) ? (X) : (Y))
#define MAX(X,Y) ((X>Y)?X:Y)

float max(float *arr, int len) {
    float currentMax = 0; //in the mnist dataset the min # is 0    
    for (int i = 0; i < len; i++) {
        if ((arr[i] - currentMax) > 0.001) { // checking if the floating point # is larger 
            currentMax = arr[i];
        } 
    }
    return currentMax; 
}
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
    
    
    FILE *f = fopen(datafile, "r");
    if(f == NULL){
        printf("Failed to open file\n");
        return NULL; 
    }

    
    ssize_t tot_tokens = 0; //number of tokens in datafile 
    ssize_t tot_lines = 0;  // number of lines in datafile

    int fileStats = filestats(datafile, &tot_tokens, &tot_lines); 
    
    KMData_t *data = malloc(sizeof(KMData_t)); 
    data->ndata = 0;  
    data->dim = 0;
    data->features = malloc(sizeof(float) * (tot_tokens * tot_lines));  // mallocing 2d array for features array 
    size_t currentRead = 0; // current line of file we are reading 
    char buffer[LINELENGTH]; 
    float tokens[tot_tokens]; 
    int currentLine = 0; 
    int tokenIndex = 0; 
    while (currentLine < tot_lines) {
        data->ndata += 1; 
        currentRead = fread(buffer, 1, LINELENGTH, f); 
        char *token = strtok(buffer, " "); 
        while (token != NULL) {
            if (isspace(token) == 0) { // if the current character is not a white space 
                tokens[tokenIndex] = atof(token); 
                tokenIndex++; 
            }
            token = strtok(NULL, " "); 
        }

        // appending features 
        float *feats = malloc(sizeof(float) * (tot_tokens - 2)); 
        for (int i = 2; i < tot_tokens; i++) {
            feats[i-2] = tokens[i]; // adding features to feature array 
        }
        data->features[currentLine*tot_tokens]= feats; // the features array is 2d and currentline will just be the index into that array 
        currentLine++; 
    }
    fclose(f); 
    return data; 
}


KMClust_t* kmclust_new(int nclust, int dim) {
    KMClust_t* clust = malloc(sizeof(KMClust_t)); 
    clust->nclust = nclust;
    clust->dim = dim;
    clust->features = malloc(sizeof(float)*nclust*dim); //mallocing enough space for 2d array 
    clust->counts = malloc(sizeof(int) * nclust);

    
    for (int c = 0; c < nclust; c++){
        clust->features[c] = calloc(dim, sizeof(float)); 
        clust->counts[c] = 0; 
    }

    return clust;
}

void save_pgm_files(KMClust_t *clust, char *savedir) {
    int nclust = clust->nclust; 
    int dim = clust->dim; 
    int dim_root = (int) sqrt(dim); 
    if (clust->dim % dim_root == 0) {
        printf("Saving cluster centers to %p/cent_0000.pgm ...\n", savedir); 
        
        float maxClusterFeatures[nclust]; // we have nclust number of max features to compare 
        float maxfeat = 0;  
        for (int i = 0; i < nclust; i++) { //equivalent to the map in python finding the max 
            maxClusterFeatures[i] = max(clust->features[i], dim); 
        }
        maxfeat = max(maxClusterFeatures, nclust); 
        for (int c = 0; c < nclust; c++) {
            char *outfile = strcat(strcat("Saving cluster centers to ", savedir), "/cent_0000.pgn ...\n");
            FILE *pgm = fopen(outfile, "w"); 
            char *p2 = strcat(strcat("P2", outfile), "\n");
            fwrite(p2, strlen(p2), 1, pgm); 
            char temp[100]; 
            sprintf(temp, "%d %d\n", dim_root, dim_root);      
            fwrite(temp, strlen(temp), 1, pgm);            
            sprintf(temp, "%.0f\n", maxfeat); 
            fwrite(temp, sizeof(temp), 1, pgm); 
            for (int d = 0; d < dim; d++) {
                if ((d > 0 && d%dim_root) == 0) {
                    fwrite("\n", 1, 1, pgm); 
                }
                sprintf(temp, "%.3f\n", clust->features[c][d]); 
                fwrite(temp, sizeof(temp), 1, pgm); 
            }
            fwrite("\n", 1, 1, pgm); 
            fclose(pgm);
        }  
    }
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
        printf("%3d: %5d |", curiter, nchanges);
        for (int c = 0; c < nclust; c++){
            printf(" %4d", clust->counts[c]);
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
    printf("\n");

    //=================================
    // CLEANUP + OUTPUT

    // CONFUSION MATRIX
    int confusion[sizeof(data->nlabels)/sizeof(int)][nclust];
    for (int i = 0; i < sizeof(data->nlabels)/sizeof(int); i++){
        for (int j = 0; j < nclust; j++){
            confusion[i][j] = 0;
        }
    }

    for (int i = 0; i < data->ndata; i++){
        confusion[data->labels[i]][data->assigns[i]] += 1;
    }

    printf("==CONFUSION MATRIX + COUNTS==");
    printf("LABEL \\ CLUST");

    // confusion matrix header
    printf("%2s", "");
    for (int j = 0; j < clust->nclust; j++){
        printf(" %4d", j);
    }
    printf(" %4s\n", "TOT");

    int tot;

    // each row of confusion matrix
    for (int i = 0; i < data->nlabels; i++){
        printf("%2d", i);
        tot = 0;
        for (int j = 0; j < clust->nclust; j++){
            printf(" %4d", confusion[i][j]);
            tot += confusion[i][j];
        }
        printf(" %4d\n", tot);
    }


    // final total row of confusion matrix
    printf("TOT");
    tot = 0;
    for (int c = 0; c < clust->nclust; c++){
        printf(" %4d", clust->counts[c]);
        tot += clust->counts[c];
    }
    printf(" %4d\n\n", tot);


    // LABEL FILE OUTPUT
    char* outfile;
    strcpy(outfile, savedir);
    strcat(outfile, "/labels.txt");
    printf("Saving cluster labels to file %s", outfile);

    FILE* file = fopen(outfile, "w");
    for (int i = 0; i < data->ndata; i++){
        fprintf(file, "%2d %2d", data->labels[i], data->assigns[i]);
    }
    

    // SAVE PGM FILES CONDITIONALLY
    save_pgm_files(clust, savedir);
}
