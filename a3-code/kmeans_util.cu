#include "kmeans_util.h"

// Sets number of lines and total number of whitespace separated
// tokens in the file. Returns -1 if file can't be opened, 0 on
// success.
//
// EXAMPLE: int ret = filestats("digits_all_1e1.txt", &toks, &lines);
// toks  is now 7860 : 10 lines with 786 tokens per line, label + ":" + 28x28 pixels
// lines is now 10   : there are 10 lines in the file
int filestats(char *filename, ssize_t *tot_tokens, ssize_t *tot_lines){
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

int intMax (int *arr, int len) {
    int current_min = INT_MIN; 
    for (int i = 0; i < len; i++) {
        if (arr[i] > current_min) {
            current_min = arr[i]; 
        }
    }
    return current_min; 
}


KMData* kmdata_load(char* datafile) {
    // KMData* data = malloc(sizeof(KMData));
    // memset(data, 0, sizeof(KMData)); // zeroing out all data 
    KMData* data = (KMData*) calloc(1, sizeof(KMData)); 
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
    data->features = (float *) malloc(tot_lines * featuresLength * sizeof(float)); // allocating a 2d array for the features 

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
    data->assigns = (int *) malloc(sizeof(int) * data->ndata); //allocating assigns array for later 
    memset(data->assigns, 0, sizeof(int) * data->ndata); //zerioing out assigns for now 
    
    return data;
}


void helper_print(float* arr, int num_row, int num_col) {
	// for (int i = 0; i < num_row; i++) {
		for (int j = 0; j < num_col; j++) {
			printf("%.0f ", arr[(0) * num_col + j]);
		// }
		// printf("------------------------------------------------------------------------------\n");
	}
  printf("\n------------------\n");
  return;
}


KMClust* kmclust_new(int nclust, int dim) {
    KMClust* clust = (KMClust*) malloc(sizeof(KMClust)); 
    memset(clust, 0, sizeof(KMClust)); // zeroing out all data 
    clust->nclust = nclust;
    clust->dim = dim;

    clust->features = (float *) malloc(sizeof(float) * nclust * dim); 
    
    clust->counts = (int *) malloc(sizeof(int) * nclust); 
  
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