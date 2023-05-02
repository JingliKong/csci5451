#include <stdio.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include "kmeans_util.h"

__global__ void zero_centers(int dim, int nclust, float* clust_features) {
  int i = threadIdx.x; // one thread for every cluster

  if (i < nclust) {
    for (int d = 0; d < dim; d++) {
      clust_features[i*dim + d] = 0.0;
    }
    
    __syncthreads();
  }
  
}

__global__ void cluster_centers(int dim, int ndata, float* clust_features, 
                                float* data_features, int* data_assigns) {
  int dp = (blockDim.x * blockIdx.x) + threadIdx.x; // one thread for every data point
  // convert this later to dimensions instead of data points

  if (dp < ndata) {
    // sum up data in each cluster
    int c = data_assigns[dp];
    for (int d = 0; d < dim; d++) {
      // clust_features[c * dim + d] += data_features[dp * dim + d];
      atomicAdd(&clust_features[c*dim + d], data_features[dp * dim + d]);
    }

    __syncthreads();
  }
}

__global__ void divide_centers(int dim, int nclust, float* clust_features, int* clust_counts) {
  int i = threadIdx.x; // one thread for every cluster

  if (i < nclust) {
    // divide by ndatas of data to get mean of cluster center
    if (clust_counts[i] > 0) {
      for (int d = 0; d < dim; d++) {
        clust_features[i * dim + d] = clust_features[i * dim + d] / clust_counts[i];
      }
    }

    __syncthreads();

  }
  
}

__global__ void cluster_assignment(int nclust, int ndata, int dim, int* nchanges_ptr, 
                                    int* clust_counts, float* clust_features, 
                                    int* data_assigns, float* data_features) {
  // int dp = threadIdx.x; // one thread for every data point
  int dp = (blockDim.x * blockIdx.x) + threadIdx.x; // one thread for every data point

  // DETERMINE NEW CLUSTER ASSIGNMENTS FOR EACH DATA
  if (dp < nclust) {
    clust_counts[dp] = 0;
  }
  __syncthreads();

  if (dp < ndata) {
    // calculate the best cluster for the data point
    int best_clust = INT_MIN;
    float best_distsq = INFINITY;

    for (int c = 0; c < nclust; c++) {
      float distsq = 0.0;
      for (int d = 0; d < dim; d++) {
        float diff = data_features[dp * dim + d] - clust_features[c * dim + d];
        distsq += diff * diff;
      }
      if (distsq < best_distsq) {
        best_clust = c;
        best_distsq = distsq;
      }
    }
    atomicAdd(&clust_counts[best_clust], 1);
    // clust_counts[best_clust] += 1;

    if (best_clust != data_assigns[dp]) {
      atomicAdd(nchanges_ptr, 1);
      data_assigns[dp] = best_clust;
    }

    __syncthreads();
  }

}

void checkCUDAError(const char *msg)
{
    cudaError_t err = cudaGetLastError();
    if( cudaSuccess != err)
    {
        fprintf(stderr, "CUDA Error: %s: %s.\n", msg, cudaGetErrorString(err) );
        exit(EXIT_FAILURE); 
    }
}

int main(int argc, char **argv) {
  if (argc < 3) {
    exit(-1);
  }
  char *datafile = argv[1];
  int nclust = atoi(argv[2]);
  // Allocating enough space for the dir len + 1 for null terminator?
  char *savedir = (char *) malloc( strlen(argv[3]) + 1);  
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

  KMData *data = kmdata_load(datafile);
  KMClust *clust = kmclust_new(nclust, data->dim);
  printf("ndata: %d\n", data->ndata);
  printf("dim: %d\n\n", data->dim);
  for (int i = 0; i < data->ndata; i++) {
    int c = i % clust->nclust;
    data->assigns[i] = c;  // give every feature array a cluster
  }

  for (int c = 0; c < clust->nclust; c++) {
    float icount = data->ndata / clust->nclust;
    float extra = (c < (data->ndata % clust->nclust)) ? 1 : 0;
    clust->counts[c] =
        icount + extra;  // counts is saying how may features are per cluster
  }

  // Main Algorithm
  int curiter = 1;
  int nchanges = data->ndata;
  printf("==CLUSTERING: MAXITER %d==\n", MAXITER);
  printf("ITER NCHANGE CLUST_COUNTS\n");

  // allocating shared memory
  float* clust_features;
  int* clust_counts;
  float* data_features;
  int* data_assigns;

  cudaMalloc((void**) &clust_features, clust->dim * clust->nclust * sizeof(float));
//   checkCUDAError("Error allocating clust_features");
//   cudaMemset(clust_features, 0, clust->dim * clust->nclust * sizeof(float));
//   checkCUDAError("Error memset clust_features");

  cudaMalloc((void**) &clust_counts, clust->nclust * sizeof(int));
  checkCUDAError("Error allocating clust_counts");
  cudaMemset(clust_counts, 0, clust->nclust * sizeof(int));
  checkCUDAError("Error memset clust_counts");

  cudaMalloc((void**) &data_features, data->ndata * data->dim * sizeof(float));
//   checkCUDAError("Error allocating data_features");
//   cudaMemset(data_features, 0, data->ndata * data->dim * sizeof(float));
//   checkCUDAError("Error memset data_features");

  cudaMalloc((void**) &data_assigns, data->ndata * sizeof(int));
//   checkCUDAError("Error allocating data_assigns");
//   cudaMemset(data_assigns, 0, data->ndata * sizeof(int));
//   checkCUDAError("Error memset data_assigns");


  int* nchanges_gpu;
  cudaMalloc((void**) &nchanges_gpu, sizeof(int));
  checkCUDAError("Error allocating nchanges_gpu");
  cudaMemset(nchanges_gpu, 0, sizeof(int));
  checkCUDAError("nchanges memset");

  // copy over the initialization for assigns and counts
  cudaMemcpy(data_assigns, data->assigns, data->ndata * sizeof(int), 
            cudaMemcpyHostToDevice);
  checkCUDAError("Error memcpy host data_assigns");

  cudaMemcpy(data_features, data->features, data->ndata * data->dim * sizeof(float), 
            cudaMemcpyHostToDevice);
  checkCUDAError("Error memcpy host data_assigns");

  cudaMemcpy(clust_counts, clust->counts, clust->nclust * sizeof(int), cudaMemcpyHostToDevice);
  checkCUDAError("Error memcpy host clust_counts");

  int nblocks = ceil(((double)data->ndata)/512);
  while ((nchanges > 0) && (curiter <= MAXITER)) {

    // cluster sums
    zero_centers<<<1, clust->dim>>>(clust->dim, clust->nclust, clust_features); // TODO: more than 1 block??
    checkCUDAError("zero_centers");

    // cluster_centers<<<1, data->ndata>>>(data->dim, data->ndata, clust_features, data_features, data_assigns);
    cluster_centers<<<nblocks, 512>>>(data->dim, data->ndata, clust_features, data_features, data_assigns);
    checkCUDAError("cluster_centers");

    divide_centers<<<1, clust->dim>>>(data->dim, clust->nclust, clust_features, clust_counts);
    checkCUDAError("divide_centers");

    cudaMemcpy(clust->counts, clust_counts, clust->nclust * sizeof(int), cudaMemcpyDeviceToHost);
  

    // cluster_assignment
    cudaMemset(nchanges_gpu, 0, sizeof(int));
    checkCUDAError("nchanges memset");

    // cluster_assignment<<<1, data->ndata>>>(clust->nclust, data->ndata, data->dim, nchanges_gpu,
    //                                     clust_counts, clust_features, data_assigns, data_features);
    cluster_assignment<<<nblocks, 512>>>(clust->nclust, data->ndata, data->dim, nchanges_gpu,
                                        clust_counts, clust_features, data_assigns, data_features);
    checkCUDAError("cluster_assignment");


    cudaMemcpy(clust->counts, clust_counts, clust->nclust * sizeof(int), cudaMemcpyDeviceToHost);
    checkCUDAError("Error memcpy clust_counts");

    cudaMemcpy(&nchanges, nchanges_gpu, sizeof(int), cudaMemcpyDeviceToHost); //QUESTION: do i need to put nchanges as a separate var?
    checkCUDAError("Error memcpy nchanges_gpu");
    
    printf("%3d: %5d |", curiter, nchanges);
    for (int c = 0; c < nclust; c++) {
      printf(" %4d", clust->counts[c]);
    }
    printf("\n");
    curiter += 1;

  } // end while loop

  // copy over the rest of the arrays
  cudaMemcpy(clust->features, clust_features, clust->dim * clust->nclust * sizeof(float), 
            cudaMemcpyDeviceToHost); 
  checkCUDAError("Error memcpy clust_features");

            
  cudaMemcpy(data->features, data_features, data->ndata * data->dim * sizeof(float), 
            cudaMemcpyDeviceToHost);
  checkCUDAError("Error memcpy data_features");
  
  cudaMemcpy(data->assigns, data_assigns, data->ndata * sizeof(int), 
            cudaMemcpyDeviceToHost);
  checkCUDAError("Error memcpy data_assigns");


  if (curiter > MAXITER) {
    printf("WARNING: maximum iteration %d exceeded, may not have conveged",
           MAXITER);
  } else {
    printf("CONVERGED: after %d iterations", curiter);
  }
  printf("\n");

  //=================================
  // CLEANUP + OUTPUT

  // CONFUSION MATRIX
  int confusion[data->nlabels][nclust];  // am I initializing this right?
  for (int i = 0; i < data->nlabels; i++) {
    for (int j = 0; j < nclust; j++) {
      confusion[i][j] = 0;
    }
  }

  for (int i = 0; i < data->ndata; i++) {
    confusion[data->labels[i]][data->assigns[i]] += 1;
  }

  printf("==CONFUSION MATRIX + COUNTS==\n");
  printf("LABEL \\ CLUST");

  // confusion matrix header
  printf("%2s\n", "");
  for (int j = 0; j < clust->nclust; j++) {
    printf(" %4d", j);
  }
  printf(" %4s\n", "TOT");

  int tot = 0;

  // each row of confusion matrix
  for (int i = 0; i < data->nlabels; i++) {
    printf("%2d:", i);
    tot = 0;
    for (int j = 0; j < clust->nclust; j++) {
      printf(" %4d", confusion[i][j]);
      tot += confusion[i][j];
    }
    printf(" %4d\n", tot);
  }

  // final total row of confusion matrix
  printf("TOT");
  tot = 0;
  for (int c = 0; c < clust->nclust; c++) {
    printf(" %4d", clust->counts[c]);
    tot += clust->counts[c];
  }
  printf(" %4d\n\n", tot);

  // LABEL FILE OUTPUT

  char *outfile =
      (char *) malloc(strlen(argv[3]) + strlen("/labels.txt") +
             strlen(argv[3]));  // recall the size of savedir is argv[3]
  sprintf(outfile, "%s/labels.txt", savedir);
  printf("Saving cluster labels to file %s\n", outfile);

  FILE *file = fopen(outfile, "w");
  for (int i = 0; i < data->ndata; i++) {
    fprintf(file, "%2d %2d\n", data->labels[i], data->assigns[i]);
  }
  fclose(file);

  // SAVE PGM FILES CONDITIONALLY
  save_pgm_files(clust, savedir);

  // Freeing allocated memory

  // free CUDA allocated
  cudaFree(data_assigns);
  cudaFree(data_features);
  cudaFree(clust_counts);
  cudaFree(clust_features);

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
  return 0;
}


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
    // zeroing out all data 
    KMData* data = (KMData*) calloc(1, sizeof(KMData)); 

    FILE* fin = fopen(datafile, "r");
    if (fin == NULL) {
        printf("Error opening file\n");
        cudaFree(data); // free(data);
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
            fscanf(fin, "%d", &currentIntToken);   
            data->features[i * featuresLength + j] = currentIntToken; // appending feature to feature array   
        } 
    }
    fclose(fin);
    data->ndata = ndata; 
    data->dim = featuresLength; 
    data->nlabels = intMax(data->labels, tot_lines) + 1; // note since I increment the labelIdx when I add a new label this should be the length
    data->assigns = (int *) malloc(sizeof(int) * data->ndata); //allocating assigns array for later 
    memset(data->assigns, 0, sizeof(int) * data->ndata); //zerioing out assigns for now 
    
    return data;
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
                    fprintf(pgm, "\n");
                }
				fprintf(pgm, "%3.0f ", clust->features[c*dim + d]);  
            }
            fprintf(pgm, "\n");
            fclose(pgm);
            
        } 

    }
}
