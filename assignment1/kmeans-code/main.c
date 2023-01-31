// Running: f
// gdb -tui --args kmeans sample-mnist-data/digits_all_1e2.txt 10 outdir  
// Compile: gcc -g main.c -o kmeans -lm
// mine: ./kmeans sample-mnist-data/test.txt 3 outdir
// running python: ./kmeans.py sample-mnist-data/digits_all_1e2.txt 10 outdir
// gdb -tui --args kmeans sample-mnist-data/test.txt 3 outdir 
#include <stdlib.h>
#include <string.h>
#include <limits.h>
#include <math.h>
#include <stdio.h>
#include <ctype.h>  // for isspace() etc.

int filestats(char *filename, ssize_t *tot_tokens, ssize_t *tot_lines){
// Sets number of lines and total number of whitespace separated
// tokens in the file. Returns -1 if file can't be opened, 0 on
// success. 
//
// EXAMPLE: int ret = filestats("digits_all_1e1.txt", &toks, &lines);
// toks  is now 7860 : 10 lines with 786 tokens per line, label + ":" + 28x28 pixels
// lines is now 10   : there are 10 lines in the file
  FILE *fin = fopen(filename,"r");
  if(fin == NULL){
    printf("Failed to open file '%s'\n",filename);
    return -1;
  }

  ssize_t ntokens=0, nlines=0, column=0;
  int intoken=0, token;
  while((token = fgetc(fin)) != EOF){
    if(token == '\n'){          // reached end of line
      column = 0;
      nlines++;
    }
    else{
      column++;
    }
    if(isspace(token) && intoken==1){ // token to whitespace
      intoken = 0;
    }
    else if(!isspace(token) && intoken==0){ // whitespace to token
      intoken = 1;
      ntokens++;
    }
  }
  if(column != 0){              // didn't end with a newline
    nlines++;                   // add a line on to the count
  }
  *tot_tokens = ntokens;
  *tot_lines = nlines;
  fclose(fin);
  // printf("DBG: tokens: %lu\n",ntokens);
  // printf("DBG: lines: %lu\n",nlines);
  return 0;
}


typedef struct KMClust {
    int nclust;   // number of clusters, the "k" in kmeans
    int dim;      // dimension of features for data
    float** features; // 2D indexing for individual cluster center features
    int* counts;
} KMClust;

typedef struct KMData {
    int ndata;    // count of data
    int dim;      // dimension of features for data
    float** features; // pointers to individual features
    int* assigns; // cluster to which data is assigned
    int* labels;  // label for data if available
    int nlabels;  // max value of labels +1, number 0,1,...,nlabel0
} KMData;

float **malloc2dFloatArray(int dim1, int dim2) { // This is where I learned how to malloc a 2d array https://www.youtube.com/watch?v=aR7tkVj3UU0 
    float **ipp; 
    ipp = (float **) malloc (dim1*sizeof(float*));
    for (int i =0; i < dim1; i++) {
        ipp[i] = (float *) malloc(dim2 * sizeof(float)); 
    }
    return ipp; 
} 

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
    KMData* data = (KMData*)malloc(sizeof(KMData));
    memset(data, 0, sizeof(KMData)); // zeroing out all data 
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
    data->labels = (int *) calloc(tot_lines, sizeof(int) * tot_lines); 
    //length of features array 
    int featuresLength = tot_tokens/tot_lines - 2; 
    // allocating space for all the features arrays 
    data->features = malloc2dFloatArray(tot_lines, tot_tokens/tot_lines); // allocating a 2d array for the features 

    int ndata = 0; // keeping track of ndata 
    int currentIntToken = 0; // used to store the current feature token 
    char colon[1]; 
    for (int i = 0; i < tot_lines; i++) {
        ndata++; 
        fscanf (fin, "%d %s", &currentIntToken, &colon);
        data->labels[i] = currentIntToken; // appending label to labels array 
        for (int j = 0; j < featuresLength; j++) {
            fscanf(fin, "%d", &currentIntToken);
            data->features[i][j] = currentIntToken; // appending feature to feature array   
        }
    }
    fclose(fin);
    data->ndata = ndata; 
    data->dim = featuresLength; 
    data->nlabels = intMax(data->labels, featuresLength) + 1; // note since I increment the labelIdx when I add a new label this should be the length
    data->assigns = malloc(sizeof(int) * data->ndata); //allocating assigns array for later 
    memset(data->assigns, 0, sizeof(int) * data->ndata); //zerioing out assigns for now 
    
    return data;
}


KMClust* kmclust_new(int nclust, int dim) {
    KMClust* clust = malloc(sizeof(KMClust)); 
    memset(clust, 0, sizeof(KMClust)); // zeroing out all data 
    clust->nclust = nclust;
    clust->dim = dim;

    clust->features = malloc2dFloatArray(nclust, dim); 
    clust->counts = malloc(sizeof(int) * nclust); 
  
    for (int c = 0; c < nclust; c++) {
        for (int d = 0; d < dim; d++) {
            clust->features[c][d] = 0.0; 
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
    if (1){
        printf("Saving cluster centers to %p/cent_0000.pgm ...\n", savedir); 
        
        float* maxClusterFeatures = malloc(nclust * sizeof(float)); // we have nclust number of max features to compare 
        float maxfeat = 0;  
        for (int i = 0; i < nclust; i++) { //equivalent to the map in python finding the max 
            maxClusterFeatures[i] = floatMax(clust->features[i], dim); 
        }
        maxfeat = floatMax(maxClusterFeatures, nclust); 
        for (int c = 0; c < nclust; c++) {
            char outfile[100]; 
            sprintf(outfile, "%s/cent%.04d.pgm\0", savedir, c);
            FILE *pgm = fopen(outfile, "w+"); 
            // char *p2 = strcat(strcat("P2", outfile), "\n");
            // fwrite("P2\n", sizeof(char), 1, pgm); 
            fprintf(pgm,"P2\n");
            // char temp[100]; 
            // sprintf(temp, "%d %d\n", dim_root, dim_root);      
            // fwrite(temp, sizeof(char), strlen(temp), pgm);            
            // sprintf(temp, "%.0f\n", maxfeat); 
            // fwrite(temp, sizeof(char), sizeof(temp), pgm); 
            fprintf(pgm, "%d %d\n", dim_root, dim_root);
            fprintf(pgm,"%d\n",maxfeat);

            for (int d = 0; d < dim; d++) {
                if ((d > 0 && d%dim_root) == 0) {
                    // fwrite("\n", 1, 1, pgm); 
                    fprintf(pgm, "\n");
                }

                // for (int w = 0; w < sizeof(clust->features[c][d]); w++){
                //     fprintf(pgm, " ");
                // }
                int result = round(clust->features[c][d]);

                fprintf(pgm, "%3d ", result);
                // sprintf(temp, "%.3f\n", clust->features[c][d]); 
                // fwrite(temp, sizeof(temp), 1, pgm); 
            }
            // fwrite("\n", 1, 1, pgm); 
            fprintf(pgm, "\n");
            fclose(pgm);
            
        } 
        free(maxClusterFeatures); 
    }
}
int main(int argc, char **argv) {
    if (argc < 3) {
        exit(-1); 
    }
    char* datafile = argv[1]; 
    int nclust = atoi(argv[2]);
    // char *savedir = malloc(100*sizeof(char)); //for now we are just going to allocate 100 bytes for the savedir name  
    char savedir[100]; 
    int MAXITER = 100; 

    if (argc > 3) {
        strcpy(savedir, argv[3]);
        int status = mkdir(savedir); // maybe later do error checking       
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
        int icount = data->ndata / clust->nclust;
        int extra = (c < (data->ndata % clust->nclust)) ? 1 : 0;
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
    // char* outfile;
    // strcpy(outfile, savedir);
    // strcat(outfile, "/labels.txt");

    char outfile[50]; 
    sprintf(outfile, "%s/labels.txt", savedir); 
    printf("Saving cluster labels to file %s\n", outfile);

    FILE* file = fopen(outfile, "w");
    for (int i = 0; i < data->ndata; i++){
        fprintf(file, "%2d %2d", data->labels[i], data->assigns[i]);
    }
    fclose(file); 

    // SAVE PGM FILES CONDITIONALLY
    save_pgm_files(clust, savedir);

    // free(savedir);
} 