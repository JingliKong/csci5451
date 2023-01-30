// Compile: gcc -g main.c filestats.c -o kmeans -lm
// Running: ./kmeans sample-mnist-data/digits_all_1e2.txt 10 outdir
// gdb -tui --args kmeans sample-mnist-data/digits_all_1e2.txt 10 outdir  
// Compile: gcc -g main.c -o kmeans -lm

// running python ./kmeans.py sample-mnist-data/digits_all_1e2.txt 10 outdir
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
    data->labels = (int *) calloc(tot_lines, sizeof(int) * tot_lines); // just the pointers we need 
    // allocating space for all the features arrays 
    data->features = (double*) malloc(tot_tokens/tot_lines * sizeof(double*));
    char line[100000]; //each line of a file is 3130 bytes long I think  
    int labelIdx = 0; 
    int featureIdx = 0; 
    int lenFeatures0 = 0;  
    while (fgets(line, sizeof(line), fin) != NULL) {
        data->ndata++;
        char* tokens = strtok(line, " "); //Grabbing the label e.g. 7 
        data->labels[labelIdx] = atoi(tokens); //adding num to arr of labels 
        
        tokens = strtok(NULL, " "); // grabbing the : token out 
        
        // allocating features array 
        // we get the current feature array size by taking tot_tokens/tot_lines
        // data->features[featureIdx] = (double*)calloc(tot_tokens/tot_lines, sizeof(double*)); //currently segfaults here 
        // allocating enough space for the feature array which contains doubles  
        data->features[featureIdx] = (double *) calloc (tot_tokens/tot_lines, sizeof(double)); 
        int i = 0;
        tokens = strtok(NULL, "   "); // starting to tokenize past the : note the 3 spaces 
        while (tokens != NULL) {
            data->features[featureIdx][i] = atof(tokens);
            tokens = strtok(NULL, "  ");
            if (featureIdx == 0) {
                lenFeatures0++; 
            }
        }
        featureIdx++; 
        labelIdx++; 
    }
    data->dim = lenFeatures0;  
    data->nlabels = intMax(data->labels, labelIdx) + 1; // note since I increment the labelIdx when I add a new label this should be the length
    data->assigns = malloc(sizeof(int) * data->ndata); //allocating assigns array for later 
    memset(data->assigns, 0, sizeof(int) * data->ndata); 
    fclose(fin);
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
void save_pgm_files(KMClust *clust, char *savedir) {
    int nclust = clust->nclust; 
    int dim = clust->dim; 
    int dim_root = (int) sqrt(dim); 
    if (clust->dim % dim_root == 0) {
        printf("Saving cluster centers to %p/cent_0000.pgm ...\n", savedir); 
        
        float maxClusterFeatures[nclust]; // we have nclust number of max features to compare 
        float maxfeat = 0;  
        for (int i = 0; i < nclust; i++) { //equivalent to the map in python finding the max 
            maxClusterFeatures[i] = floatMax(clust->features[i], dim); 
        }
        maxfeat = floatMax(maxClusterFeatures, nclust); 
        for (int c = 0; c < nclust; c++) {
            char *msg; 
            sprintf(msg, "Saving cluster centers to %s /cent_0000.pgn ...\n", savedir); 
            printf(msg); 
            char *outfile; 
            sprintf(outfile, "%s/cent%.04d.pgm", savedir, c);
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
int main(int argc, char **argv) {
    if (argc < 3) {
        exit(-1); 
    }
    char* datafile = argv[1]; 
    int nclust = atoi(argv[2]);
    char *savedir = malloc(100); //for now we are just going to allocate 100 bytes for the savedir name  
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
    printf("nlabels: %d\n", data->nlabels);
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

    char *outfile[50]; 
    sprintf(outfile, "%s/labels.txt", savedir); 
    printf("Saving cluster labels to file %s", outfile);

    FILE* file = fopen(outfile, "w");
    for (int i = 0; i < data->ndata; i++){
        fprintf(file, "%2d %2d", data->labels[i], data->assigns[i]);
    }
    fclose(file); 

    // SAVE PGM FILES CONDITIONALLY
    save_pgm_files(clust, savedir);
} 