#include "kmeans_util.h"

#include <cuda.h>
#include <cuda_runtime.h>

// ./kmeans_cuda mnist-data/digits_all_1e2.txt 10 test-results/outdir_cuda01 500
__global__ void initAssignments(int *assigns, int ndata, int nclust) {
    // gives us the actual index of the feature same as in serial version
    int idx = blockIdx.x * blockDim.x + threadIdx.x; 
    if (idx < ndata) { // making sure we stay inbounds
		assigns[idx] = idx % nclust;
    }
}

for (int i = 0; i < data->ndata; i++) { // instead of doing this loop we create nclust blocks with ndata per block?
    int c = data->assigns[i]; // need to copy this memory over to device 
    for (int d = 0; d < clust->dim; d++) { // this is fixed an needs to be passed in 
    clust->features[c * clust->dim + d] += // instead of adding to clust->features we need to allocate some shared memory for the threads to work on 
        data->features[i * clust->dim + d]; // features is passed in and we use the index and stuff of the thread to know which elements to add 
    }
}
__global__ void updateCentroids (int ndata, int clust_dim, int *assigns, float *data_features, float* clust_features) 
{
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    
    // Allocate shared memory for storing sums
    extern __shared__ float shared_data[];

    // Initialize shared memory to zero
    memset(shared_data, 0, sizeof(float) * blockDim.x * clust_dim);
    // Use indexing to access only the part of the array that this thread needs
    float* my_data = shared_data + threadIdx.x * clust_dim;

    int c = assigns[idx];
    
    for (int d = 0; d < clust_dim; d++)
    {
        my_data[d] += data_features[clust_dim * idx + d];
    }
    __syncthreads();

    if (threadIdx.x == 0) // only one thread needs to edit global memory
    {   
        float *clust_feat_ptr = idx 
        for (int i = 0; i < clust_dim; i++) {
            atomicAdd(&clust_features[idx], shared_data[i]);
        }
    }
}





// int filestats(char *filename, ssize_t *tot_tokens, ssize_t *tot_lines);
// int intMax(int *arr, int len);

int main(int argc, char **argv) {
  if (argc < 3) {
    exit(-1);
  }
  char *datafile = argv[1];
  int nclust = atoi(argv[2]);
  char *savedir = (char *)malloc(
      strlen(argv[3]) +
      1);  // Allocating enough space for the dir len + 1 for null terminator?
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
  
	
	// To parallelize the assignments the only memory we need to copy to the device is the assigns array
	int blockSize = 256; // 256 threads per block 
	int gridSize = (data->ndata / blockSize + (data->ndata % blockSize));
	int *d_assigns; 
	// allocating memory on device  
	cudaMalloc((void **) &d_assigns, sizeof(int) * data->ndata); 
	initAssignments <<<gridSize, blockSize>>> (d_assigns, data->ndata, clust->nclust); 

	// sync and copy memory back to CPU
	cudaDeviceSynchronize(); 
	cudaMemcpy(data->assigns, d_assigns, sizeof(int) * data->ndata, cudaMemcpyDeviceToHost); 

	// TODO: This should be done locally per thread probably
  for (int c = 0; c < clust->nclust; c++) {
    float icount = data->ndata / clust->nclust;
    float extra = (c < (data->ndata % clust->nclust)) ? 1 : 0;
    // counts is saying how may features are per cluster
    clust->counts[c] = icount + extra;  
  }

  // Main Algorithm
  int curiter = 1;
  int nchanges = data->ndata;
  printf("==CLUSTERING: MAXITER %d==\n", MAXITER);
  printf("ITER NCHANGE CLUST_COUNTS\n");

  while ((nchanges > 0) && (curiter <= MAXITER)) {
    // DETERMINE NEW CLUSTER CENTERS
    // reset cluster centers to 0.0

    for (int c = 0; c < clust->nclust; c++) {
      for (int d = 0; d < clust->dim; d++) {
        clust->features[c * clust->dim + d] = 0.0;
      }
    }
		// so we can get a block for each cluster which has as many threads as features we need to add 
		dim3 blockDim(data->dim); // threads per blocks 
		dim3 numBlocks(clust->nclust); // number of blocks per grid

		// allocating variables used for parallelization
		int *d_feature_assigns;
		float *d_data_features, *d_clust_features;
		int *d_clust_counts;
    // Allocate memory on the device
    cudaMalloc((void**)&d_feature_assigns, data->ndata * sizeof(int));
    cudaMalloc((void**)&d_data_features, data->ndata * data->dim * sizeof(float));
    cudaMalloc((void**)&d_clust_features, clust->dim * sizeof(float));		
		cudaMalloc((void**)&d_clust_counts, clust->nclust * sizeof(int));
		cudaMemset(d_clust_counts, 0, clust->nclust * sizeof(int));
		// Copy data from host to device
    cudaMemcpy(d_feature_assigns, data->assigns, data->ndata * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_data_features, data->features, data->ndata * data->dim * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_clust_features, clust->features, clust->dim * sizeof(float), cudaMemcpyHostToDevice);

		updateCentroids<<<numBlocks, blockDim, clust->dim * sizeof(float)>>> (data->ndata, data->dim, d_feature_assigns, d_data_features, d_clust_features, d_clust_counts); // TODO: memcpy over this data to device 
		
		// Copy the result from device to host
    cudaMemcpy(clust->features, d_clust_counts, clust->dim * sizeof(float), cudaMemcpyDeviceToHost);
		cudaMemcpy(clust->counts, d_clust_counts, clust->nclust * sizeof(int), cudaMemcpyDeviceToHost);

		printf("clust_features before kernel: ");
		for (int i = 0; i < clust->dim; i++) {
				printf("%f ", clust->features[i]);
		}
		printf("\n");

		// divide by ndatas of data to get mean of cluster center
    for (int c = 0; c < clust->nclust; c++) {
      if (clust->counts[c] > 0) {
        for (int d = 0; d < clust->dim; d++) {
          clust->features[c * clust->dim + d] =
              clust->features[c * clust->dim + d] / clust->counts[c];
        }
      }
    }
	

    // DETERMINE NEW CLUSTER ASSIGNMENTS FOR EACH DATA
    for (int c = 0; c < clust->nclust; c++) {
      clust->counts[c] = 0;
    }

    nchanges = 0;
    for (int i = 0; i < data->ndata; i++) {
      int best_clust = INT_MIN;
      float best_distsq = INFINITY;
      for (int c = 0; c < clust->nclust; c++) {
        float distsq = 0.0;
        for (int d = 0; d < clust->dim; d++) {
          float diff = data->features[i * clust->dim + d] -
                       clust->features[c * clust->dim + d];
          distsq += diff * diff;
        }
        if (distsq < best_distsq) {
          best_clust = c;
          best_distsq = distsq;;
        }
      }

      clust->counts[best_clust] += 1;
      if (best_clust != data->assigns[i]) {
        nchanges += 1;
        data->assigns[i] = best_clust;
      }
    cudaFree(d_feature_assigns);
		cudaFree(d_data_features);
		cudaFree(d_clust_features);
		cudaFree(d_clust_counts);
    }

    printf("%3d: %5d |", curiter, nchanges);
    for (int c = 0; c < nclust; c++) {
      printf(" %4d", clust->counts[c]);
    }
    printf("\n");
    curiter += 1;
  }
 
  // Loop has converged
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
  // printf("nlabels: %d\n", data->nlabels);
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
  // char* outfile;
  // strcpy(outfile, savedir);
  // strcat(outfile, "/labels.txt");

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

	// Freeing cuda stuff 
	cudaFree(d_assigns);

	
	
  return 0;
}


