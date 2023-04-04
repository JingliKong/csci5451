// Running: ./kmeans_serial sample-mnist-data/digits_all_1e2.txt 10 outdir
// gdb -tui --args kmeans sample-mnist-data/digits_all_1e2.txt 10 outdir  
// Compile: gcc -g kmeans_serial.c kmeans_util.c -o kmeans_serial -lm
// running python: ./kmeans.py sample-mnist-data/digits_all_1e2.txt 10 outdir
// gdb -tui --args kmeans_serial sample-mnist-data/test.txt 3 outdir 

// valgrind --leak-check=yes ./kmeans sample-mnist-data/digits_all_1e2.txt 10 outdir

// test 1 failure: ./kmeans_serial sample-mnist-data/digits_all_1e2.txt 10 test-results/outdir1 500
/*
Test 1: 
./kmeans.py sample-mnist-data/digits_all_1e2.txt 10 test-results/outdir1 500
./kmeans_serial sample-mnist-data/digits_all_1e2.txt 10 test-results/outdir1 500

Test 3: 

./kmeans_serial sample-mnist-data/digits_all_1e3.txt 13 test-results/outdir3 500

*/
#include <stdlib.h>
#include <string.h>
#include <limits.h>
#include <math.h>
#include <stdio.h>
#include <ctype.h>  // for isspace() etc.
#include <time.h> // for comparing time with python version 
#include <sys/types.h>
#include <sys/stat.h>
#include <dirent.h>
#include <unistd.h>
// only on M1 
#include <ctype.h>

int filestats(char *filename, ssize_t *tot_tokens, ssize_t *tot_lines); 

typedef struct KMClust {
    int nclust;   // number of clusters, the "k" in kmeans
    int dim;      // dimension of features for data
    float* features; // 2D indexing for individual cluster center features
    int* counts;
} KMClust;

typedef struct KMData {
    int ndata;    // count of data
    int dim;      // dimension of features for data
    float* features; // pointers to individual features
    int* assigns; // cluster to which data is assigned
    int* labels;  // label for data if available
    int nlabels;  // max value of labels +1, number 0,1,...,nlabel0
} KMData;



int intMax (int *arr, int len) {
    int current_min = INT_MIN; 
    for (int i = 0; i < len; i++) {
        if (arr[i] > current_min) {
            current_min = arr[i]; 
        }
    }
    return current_min; 
}

float floatMax(float *arr, int len) {
    float currentMax = -INFINITY; //in the mnist dataset the min # is 0    
    for (int i = 0; i < len; i++) {
        if ((arr[i] - currentMax) > 0.001) { // checking if the floating point # is larger 
            currentMax = arr[i];
        } 
    }
    return currentMax; 
}


KMData* kmdata_load(char* datafile) {
    // KMData* data = malloc(sizeof(KMData));
    // memset(data, 0, sizeof(KMData)); // zeroing out all data 
    KMData* data = calloc(1, sizeof(KMData)); 
    FILE* fin = fopen(datafile, "r");
    if (fin == NULL) {
        printf("Error opening file\n");
        free(data);
        return NULL;
    }
    ssize_t tot_tokens = 0; //number of tokens in datafile 
    ssize_t tot_lines = 0;  // number of lines in datafile

    // Getting file data we need to allocate correct amount of space 
    int fileStats = filestats(datafile, &tot_tokens, &tot_lines); 
 
    // allocating space for the number of labels in the dataset 
    data->labels = (int *) calloc(tot_lines, sizeof(int)); 
    //length of features array 
    int featuresLength = tot_tokens/tot_lines - 2; 
    // allocating space for all the features arrays 
    data->features = malloc(tot_lines * featuresLength * sizeof(float)); // allocating a 2d array for the features 

    int ndata = 0; // keeping track of ndata 
    int currentIntToken = 0; // used to store the current feature token 
    char colon[1]; 
    for (int i = 0; i < tot_lines; i++) {
        ndata++; 
        fscanf (fin, "%d %s", &currentIntToken, colon);
        data->labels[i] = currentIntToken; // appending label to labels array 
        for (int j = 0; j < featuresLength; j++) {
            fscanf(fin, "%d", &currentIntToken); // FIXME: Currently segfaulting here in test case 5  
            data->features[i * featuresLength + j] = currentIntToken; // appending feature to feature array   
        } // FIXME unsure as to why test case 4 is segfaulting 
    }
    fclose(fin);
    data->ndata = ndata; 
    data->dim = featuresLength; 
    data->nlabels = intMax(data->labels, tot_lines) + 1; // note since I increment the labelIdx when I add a new label this should be the length
    data->assigns = malloc(sizeof(int) * data->ndata); //allocating assigns array for later 
    memset(data->assigns, 0, sizeof(int) * data->ndata); //zerioing out assigns for now 
    
    return data;
}


KMClust* kmclust_new(int nclust, int dim) {
    KMClust* clust = malloc(sizeof(KMClust)); 
    memset(clust, 0, sizeof(KMClust)); // zeroing out all data 
    clust->nclust = nclust;
    clust->dim = dim;

    clust->features = malloc(sizeof(float) * nclust * dim); 
    
    clust->counts = malloc(sizeof(int) * nclust); 
  
    for (int c = 0; c < nclust; c++) {
        for (int d = 0; d < dim; d++) {
            clust->features[c * dim + d] = 0.0; 
        }
        clust->counts[c] = 0.0; 
    }
    return clust;
}
void save_pgm_files(KMClust* clust, char* savedir) {
    int nclust = clust->nclust; 
    int dim = clust->dim; 
    int dim_root = (int) sqrt(dim); 
    // if (clust->dim % dim_root == 0) {
    if (clust->dim % dim_root == 0) {
        printf("Saving cluster centers to %s/cent_0000.pgm ...\n", savedir); 
        
        float maxfeat = -INFINITY;  

        for (int i = 0; i < nclust; i++) {
            for (int j = 0; j < dim; j++) {
                float element = clust->features[i * dim + j]; 
                if (element > maxfeat) {
                    maxfeat = element;  
                }
            }
        }
        for (int c = 0; c < nclust; c++) {
            char outfile[1024]; 
            sprintf(outfile, "%s/cent_%04d.pgm", savedir, c);
            FILE *pgm = fopen(outfile, "w+"); 

            fprintf(pgm,"P2\n");
     
            fprintf(pgm, "%d %d\n", dim_root, dim_root);
            
            fprintf(pgm,"%.0f", maxfeat);

            for (int d = 0; d < dim; d++) {
                if ((d > 0 && d%dim_root) == 0) {
                    // fwrite("\n", 1, 1, pgm); 
                    fprintf(pgm, "\n");
                }
				fprintf(pgm, "%3.0f ", clust->features[c*dim + d]); 
                // int result = round(c * dim + d);
                // fprintf(pgm, "%3d ", result);
 
            }
            // fwrite("\n", 1, 1, pgm); 
            fprintf(pgm, "\n");
            fclose(pgm);
            
        } 

    }
}
int main(int argc, char **argv) {
    // clock_t start = clock(), diff; //from the following stackoverflow post (https://stackoverflow.com/questions/459691/best-timing-method-in-c)

    if (argc < 3) {
        exit(-1); 
    }
    char* datafile = argv[1]; 
    int nclust = atoi(argv[2]);
    char *savedir = malloc(strlen(argv[3]) + 1); // Allocating enough space for the dir len + 1 for null terminator? 
    int MAXITER = 100; 

    if (argc > 3) {
        strcpy(savedir, argv[3]);
        int status = mkdir(savedir, S_IRWXU); 
        
    }
    if (argc > 4) {
        MAXITER = atoi(argv[4]); 
    }
    printf("datafile: %s\n", datafile);
    printf("nclust: %d\n", nclust);
    printf("savedir: %s\n", savedir);

    KMData* data = kmdata_load(datafile); 
    KMClust* clust = kmclust_new(nclust, data->dim);     
    printf("ndata: %d\n", data->ndata);
    printf("dim: %d\n\n", data->dim);


   
    for (int i = 0; i < data->ndata; i++){
        int c = i % clust->nclust;
        data->assigns[i] = c;
    }

    for (int c = 0; c < clust->nclust; c++){
        //is this supposed to be int???
        float icount = data->ndata / clust->nclust;
        float extra = (c < (data->ndata % clust->nclust)) ? 1 : 0;
        clust->counts[c] = icount + extra;
    }
    
    // Main Algorithm
    int curiter = 1;
    int nchanges = data->ndata;
    printf("==CLUSTERING: MAXITER %d==\n", MAXITER);
    printf("ITER NCHANGE CLUST_COUNTS\n");

    while ((nchanges > 0) && (curiter <= MAXITER)){

        //DETERMINE NEW CLUSTER CENTERS
        //reset cluster centers to 0.0
        for (int c = 0; c < clust->nclust; c++){ 
            for (int d = 0; d < clust->dim; d++){
                clust->features[c * clust->dim + d] = 0.0;
            }
        }

        //sum up data in each cluster
        for (int i = 0; i < data->ndata; i++){
            int c = data->assigns[i];
            for (int d = 0; d < clust->dim; d++){
                clust->features[c * clust->dim + d] += data->features[i * clust->dim + d];
            }
        }

        // divide by ndatas of data to get mean of cluster center
        for (int c = 0; c < clust->nclust; c++){
            if (clust->counts[c] > 0){
                for (int d = 0; d < clust->dim; d++){
                    clust->features[c * clust->dim + d] = clust->features[c * clust->dim + d] / clust->counts[c];
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
                    float diff = data->features[i * clust->dim + d] - clust->features[c  * clust->dim + d];
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
    int confusion[data->nlabels][nclust]; // am I initializing this right? 
    for (int i = 0; i < data->nlabels; i++){
        for (int j = 0; j < nclust; j++){
            confusion[i][j] = 0;
        }
    }
    
    for (int i = 0; i < data->ndata; i++){
        confusion[data->labels[i]][data->assigns[i]] += 1;
    }

    printf("==CONFUSION MATRIX + COUNTS==\n");
    printf("LABEL \\ CLUST");

    // confusion matrix header
    printf("%2s\n", "");
    for (int j = 0; j < clust->nclust; j++){
        printf(" %4d", j);
    }
    printf(" %4s\n", "TOT");

    int tot = 0;

    // each row of confusion matrix
    // printf("nlabels: %d\n", data->nlabels);
    for (int i = 0; i < data->nlabels; i++){
        printf("%2d:", i);
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
    // char* outfile;
    // strcpy(outfile, savedir);
    // strcat(outfile, "/labels.txt");

    char *outfile = malloc(strlen(argv[3]) + strlen("/labels.txt") + strlen(argv[3]));  // recall the size of savedir is argv[3]
    sprintf(outfile, "%s/labels.txt", savedir); 
    printf("Saving cluster labels to file %s\n", outfile);

    FILE* file = fopen(outfile, "w");
    for (int i = 0; i < data->ndata; i++){
        fprintf(file, "%2d %2d\n", data->labels[i], data->assigns[i]);
    }
    fclose(file); 

    // SAVE PGM FILES CONDITIONALLY
    save_pgm_files(clust, savedir);

    //Freeing allocated memory  

    // freeing KMDATA struct 
    free(data->features); 
    free(data->assigns);
    free(data->labels); 
	free(data); 

	// freeing KMClust struct 
	free(clust->features); 
	free(clust->counts); 
	free(clust);

	// Mischalenous frees 
	free(savedir); 
	free(outfile); 

    // diff = clock() - start;
    // int msec = diff * 1000 / CLOCKS_PER_SEC;
    // printf("Time taken %d seconds %d milliseconds\n", msec/1000, msec%1000);
    // free(savedir);
} 

// b 369
// b 184