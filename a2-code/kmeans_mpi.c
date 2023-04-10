#include <ctype.h>  // for isspace() etc.
#include <dirent.h>
#include <limits.h>
#include <math.h>
#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <time.h>  // for comparing time with python version
#include <unistd.h>
// mpirun -np 1 kmeans_mpi mnist-data/digits_all_1e1.txt 10 outdir1

void here() { // just for DEBUG
	printf("HERE!!!!!!!!!!!!!!!!\n"); // DEBUG
}

void helper_print(float* arr, int num_row, int num_col) {
	for (int i = 0; i < num_row; i++) {
		for (int j = 0; j < num_col; j++) {
			float e = arr[i * num_col + j]; 
			if (e != 0.0) {
				printf("%.0f ", arr[i * num_col + j]);
			}
		}
		printf("------------------------------------------------------------------------------\n");
	}
}
int filestats(char* filename, ssize_t* tot_tokens, ssize_t* tot_lines);
int intMax(int* arr, int len);

typedef struct KMClust {
  int nclust;       // number of clusters, the "k" in kmeans
  int dim;          // dimension of features for data
  float* features;  // 2D indexing for individual cluster center features
  int* counts;
} KMClust;

typedef struct KMData {  // ndata is the number of features we have
  int ndata;             // count of data
  int dim;               // dimension of features for data
  float* features;  // pointers to individual features also a 2D array storing
                    // all the elements for every feature
  int* assigns;     // cluster to which data is assigned
  int* labels;      // label for data if available
  int nlabels;      // max value of labels +1, number 0,1,...,nlabel0
} KMData;

void freeKMData(KMData* data) {
  free(data->features);
  free(data->assigns);
  free(data->labels);
  free(data);
}

void freeKMClust(KMClust* clust) {
  free(clust->features);
  free(clust->counts);
  free(clust);
}

KMClust* kmclust_new(int nclust, int dim);
KMData* kmdata_load(char* datafile);
void save_pgm_files(KMClust* clust, char* savedir);

int main(int argc, char** argv) {
  if (argc < 3) {
    exit(-1);
  }

  char* datafile = argv[1];
  int nclust = atoi(argv[2]);
  char* savedir = malloc(
      strlen(argv[3]) + 1);  // Allocating enough space for the dir len + 1 for null terminator?

  int MAXITER = 100;
  if (argc > 3) {
    strcpy(savedir, argv[3]);
    int status = mkdir(savedir, S_IRWXU);
  }
  if (argc > 4) {
    MAXITER = atoi(argv[4]);
  }


  MPI_Init(&argc, &argv);
  int proc_id, total_procs;
  MPI_Comm_rank(MPI_COMM_WORLD, &proc_id);      // get current process id
  MPI_Comm_size(MPI_COMM_WORLD, &total_procs);  // get number of processes

  int root_proc = 0;
  // everyone gets broadcasted dim and ndata to initialize their own clusters
  // int ndata = 0; 
	// int dim = 0;

	int ndata, dim;

  KMData* global_data = NULL;

  if (proc_id == root_proc) {
		printf("datafile: %s\n", datafile);
  	printf("nclust: %d\n", nclust);
  	printf("savedir: %s\n", savedir);
    // only the root will read in the file into global_data
    // then we have to broadcast ndata and dim
    global_data = kmdata_load(datafile);
    printf("ndata: %d\n", global_data->ndata);
    printf("dim: %d\n\n", global_data->dim);
    // Broadcasting ndata and dimensions to other procs
    ndata = global_data->ndata;
    dim = global_data->dim;
    
    
  } 
	MPI_Bcast(&ndata, 1, MPI_INT, root_proc, MPI_COMM_WORLD);
	MPI_Bcast(&dim, 1, MPI_INT, root_proc, MPI_COMM_WORLD);
	// else {
	// 	MPI_Barrier(MPI_COMM_WORLD);
	// }
  // everyone initializes data required for scatter
  int* feature_counts = calloc(total_procs, sizeof(int));
  int* feature_displ = calloc(total_procs, sizeof(int));
  int features_per_proc = ndata / total_procs;
  int feature_surplus = ndata % total_procs;
  for (int i = 0; i < total_procs; i++) {
    feature_counts[i] = (i < feature_surplus) ? features_per_proc + 1 : features_per_proc;
    feature_displ[i] = (i == 0) ? 0 : feature_displ[i - 1] + feature_counts[i - 1]; 
  }
	for (int i = 0; i < total_procs; i++) { // need to multiply the #rows by #cols to know how many elements each proc gets
		feature_counts[i] = feature_counts[i] * dim;
		feature_displ[i] = feature_displ[i] * dim; 
	}
  /*
  everyone gets:
  1) their own feature array holding only the rows of features assigned to them
  2) assigns array keeping track of where their feature rows are assigned. Note
  this can be calculated by: proc_id * dim + i where i is the number of data
  items in the local_feature array 3) their own cluster which to update with
  their own features each cluster keeps track of their own features and counts
  */
  // the number of rows of features each proc get is stored in feature_counts
  int local_ndata = feature_counts[proc_id] / dim;
	
  float* local_features = calloc(local_ndata*dim, sizeof(float));
  // we can allocate our local assigns array to keep track of where out
  // algorithm decides where each row should be assigned
  int* local_assigns = calloc(local_ndata, sizeof(int));
  // for (int i = 0; i < local_ndata; i++) {
  //   // remember we can use the proc_id to know which rows of the larger feature
  //   // array we get and c is calculated the same as the serial version doing the  following
  //   int c = proc_id * dim + i;
  //   // now we just keep track of where are feature is assigned in our local array
  //   local_assigns[i] = c;
  // }
	// printf("nclust %d\n", nclust); //DEBUG
	// for (int i = 0; i<local_ndata;i++){//DEBUG
	// 	printf("%d ", local_assigns[i]);
	// }

	// printf("features_count[0]: %d\n", feature_counts[root_proc]); // DEBUG 
	// printf("features_displ[0]: %d\n", feature_displ[root_proc]); // DEBUG
	// printf("local_ndata[0]: %d\n", local_ndata); // DEBUG
	// helper_print(global_data->features, global_data->ndata, global_data->dim); // DEBUG
	// helper_print(local_features, local_ndata, dim); // DEBUG
	printf("Proc id: %d\n", proc_id);  // DEBUG
	for (int i = 0; i < total_procs; i++) { // DEBUG
		// printf("%d feature_counts[%d]: %d\n", i, i, feature_counts[i]);
	}
		for (int i = 0; i < total_procs; i++) { // DEBUG
		// printf("%d feature_displ[%d]: %d\n", i, i, feature_displ[i]);
	}
	printf("local_ndata %d\n", local_ndata); //DEBUG
	if (proc_id != root_proc) {
		  MPI_Scatterv(NULL, feature_counts, feature_displ, MPI_FLOAT,
               local_features, feature_counts[proc_id], MPI_FLOAT, root_proc,
               MPI_COMM_WORLD);
	}
	else {
		MPI_Scatterv(global_data->features, feature_counts, feature_displ, MPI_FLOAT,
               local_features, feature_counts[proc_id], MPI_FLOAT, root_proc,
               MPI_COMM_WORLD);
	}

	
	// for (int i = 0; i < local_ndata; i++) { // DEBUG
	// 	for (int j = 0; j < dim; j++) {
	// 		float e = local_features[i*dim + j];
	// 		if (e != 0) {
	// 			printf("%0.f\n", e);
	// 		}
	// 	}
	// }							 
	// helper_print(local_features, local_ndata, dim); // DEBUG
  // We will decide which cluster center our feature reside in
  KMClust* local_clust = kmclust_new(nclust, dim);  
	// we will use nclust which is a argument passed in and dim which is broadcasted from the root 
  // now based on what our proc_id is we can figure out where are features are
  // initially assignd
  for (int i = 0; i < local_ndata; i++) {
    int c = (proc_id * dim + i) % (local_clust->nclust);
    local_assigns[i] = c;  // give every feature array a cluster based on our
                           // newly calculated c
  }

  // Calculating every num features per cluster
  for (int c = 0; c < local_clust->nclust; c++) {
    int icount = local_ndata / local_clust->nclust;  // TODO: verify if this should be an int or a float
    int extra = (c < (local_ndata % local_clust->nclust)) ? 1 : 0;
    local_clust->counts[c] = icount + extra;  // setting the counts for each local_clust based on how many features they have
		// printf("icount: %d\n", icount); //DEBUG
		// printf("extra: %d\n", extra); //DEBUG
		// printf("local_clust[c]: %d\n", local_clust->counts[c]); //DEBUG
  }
	
	// helper_print(global_data->features, global_data->ndata, global_data->dim); // DEBUG
	// helper_print(local_features, local_ndata, dim); // DEBUG
  // Main Algorithm
  // Note we will need to do an all-to-all reduce to ensure that we sync these
  // termination conditions among procs
  int curiter = 1;
  int nchanges = ndata;  // recall we broadcasted ndata from root which is all
                         // the features we have in total
  // declaring local version of curiter and nchanges we will allreduce these to
  // sync terminating conditions
  int local_curiter = 1;  // unsure about using this we probably don't need to
  int local_nchanges = 0;  // this is needed though for sure
  if (proc_id == root_proc) {
    printf("==CLUSTERING: MAXITER %d==\n", MAXITER);
    printf("ITER NCHANGE CLUST_COUNTS\n");
  }

  while ((nchanges > 0) && (curiter <= MAXITER)) {
    printf("start here p:%d iter:%d\n", proc_id, curiter);
    // DETERMINE NEW CLUSTER CENTERS
    // reset cluster centers to 0.0
    for (int c = 0; c < local_clust->nclust;
         c++) {  // remember our local clust is the same size as the global one
      for (int d = 0; d < local_clust->dim; d++) {
        local_clust->features[c * local_clust->dim + d] = 0.0;
      }
    }
		
    // sum up data in each cluster
		for (int i = 0; i < local_ndata; i++){
				int c = local_assigns[i];
				// float temp = 0; // DEBUG 
				for (int d = 0; d < local_clust->dim; d++){
					// temp += local_features[i * local_clust->dim + d]; // DEBUG 
					local_clust->features[c * local_clust->dim + d] += local_features[i * local_clust->dim + d]; 
				}
				// printf("%f ", temp);// DEBUG 
		}

    

    // for (int i = 0; i < 1; i++) { //DEBUG
		// 	for (int j = 0; j < dim; j++) {
		// 		float e = local_clust->features[i * local_clust->dim + j];
		// 		if (e != 0) {
		// 			printf("%0.f ", e);
		// 		}
		// 	}
		// 	printf("\n"); 
		// }		
		// for (int i = 0; i < local_clust->nclust; i++) { //DEBUG
		// 	int e = local_clust->counts[i];
		// 	if (e != 0) {
		// 		printf("%d ", e);
		// 	}
		// }
		// printf("\n");// DEBUG

		/*
    At this point after every proc performs its local sum we need to do an
    all-to-all reduce to synchronize all the local clust.
    */
    // Allocating a recv_features to recv everyone's elses reduction before
    // copying back into local_clust
    printf("here3 p:%d iter:%d\n", proc_id, curiter);
    float recv_features[local_clust->nclust * local_clust->dim];
    MPI_Allreduce(local_clust->features, recv_features,
                  local_clust->nclust * local_clust->dim, MPI_FLOAT, MPI_SUM,
                  MPI_COMM_WORLD);
		// helper_print(recv_features, 1, local_clust->dim); // DEBUG

    printf("here4 p:%d iter:%d\n", proc_id, curiter);
    // copying features back into local clust
    memcpy(local_clust->features, recv_features, 
           sizeof(float) * local_clust->dim * local_clust->nclust);
					 
    // for (int i = 0; i < 1; i++) { //DEBUG
		// 	for (int j = 0; j < dim; j++) {
		// 		float e = local_clust->features[i * local_clust->dim + j];
		// 		if (e != 0) {
		// 			printf("%0.f ", e);
		// 		}
		// 	}
		// 	printf("\n"); 
		// }

    // for (int i = 0; i < 1; i++) { //DEBUG
		// 	for (int j = 0; j < dim; j++) {
		// 		float e = local_clust->features[i * local_clust->dim + j];
		// 		// if (e != 0.0) {
		// 		// 	printf("%0.f ", e);
		// 		// }
		// 		printf("%0.f ", e);
		// 	}
		// 	printf("\n");
		// } 
		
    // at this point everyone has all the cluster features they need to do the
    // division step we all have the same local_clust divide by ndatas of data
    // to get mean of cluster center
    for (int c = 0; c < local_clust->nclust; c++) {
      if (local_clust->counts[c] > 0) {
        for (int d = 0; d < local_clust->dim; d++) {
          local_clust->features[c * local_clust->dim + d] = local_clust->features[c * local_clust->dim + d] / local_clust->counts[c];
        }
      }
    }
		
    // for (int i = 0; i < 1; i++) { //DEBUG
		// 	for (int j = 0; j < dim; j++) {
		// 		float e = local_clust->features[i * local_clust->dim + j];
		// 		if (e != 0.00) {
		// 			printf("%.2f ", e);
		// 		}
		// 		// printf("%.2f ", e);
		// 	}
		// 	printf("\n");
		// } 
		// // for (int i = 0; i < local_clust->nclust; i++) { //DEBUG
		// 	int e = local_clust->counts[i];
		// 	if (e != 0) {
		// 		printf("%d ", e);
		// 	}
		// }
		// printf("\n");// DEBUG
    // DETERMINE NEW CLUSTER ASSIGNMENTS FOR EACH DATA
    for (int c = 0; c < local_clust->nclust; c++) {
      local_clust->counts[c] = 0;
    }

		nchanges = 0;
		local_nchanges = 0;
		// printf("global_data->ndata %d\n", global_data->ndata); //DEBUG
    // Determining the best clusters for all the data features
    for (int i = 0; i < local_ndata; i++) {
      int best_clust = INT_MIN;
      float best_distsq = INFINITY;
      for (int c = 0; c < local_clust->nclust; c++) {
        float distsq = 0.0;
        for (int d = 0; d < dim; d++) {
          float diff = local_features[i * local_clust->dim + d] - local_clust->features[c * local_clust->dim + d];
          distsq += diff * diff;
        //   printf("i: %d, d: %d, c: %d\n", i, d, c); //DEBUG
        }
        if (distsq < best_distsq) {
          best_clust = c;
          best_distsq = distsq;
        }
      }
      // I will need to reduce these counts among different procs
      local_clust->counts[best_clust] += 1;
      if (best_clust != local_assigns[i]) {
        nchanges += 1;  // note I am updating my local_nchanges instead of nchanges
        local_assigns[i] = best_clust;
				
      }
			// for (int i = 0; i < local_clust->nclust; i++) { //DEBUG
			// 	printf("%d ", local_clust->counts[i]);
			// }
			// printf("\n"); //DEBUG

    }
		// printf("proc_id: %d\n", proc_id);
		// printf("%d nchanges before: %d\n", proc_id, nchanges); //DEBUG
    MPI_Allreduce(&nchanges, &nchanges, 1, MPI_INT, MPI_SUM,
                  MPI_COMM_WORLD);
		// printf("%d nchanges before: %d\n", proc_id, nchanges); //DEBUG
    // space to reduce cluster counts in each proc
    float recv_counts[local_clust->nclust];
    // Here I am allreducing the various local_clust->counts among the various procs
    // 
    MPI_Allreduce(local_clust->counts, recv_counts, local_clust->nclust,
                  MPI_INT, MPI_SUM, MPI_COMM_WORLD);
    // copying counts back into local_cluster
    memcpy(local_clust->counts, recv_counts, sizeof(int) * local_clust->nclust);
		
    // After this we need to do an all-to-all reduce to make sure our loop
    // termination conditions still hold Synchronize the nchanges variable among
    // all processes

    // I still need to set nchanges to local_changes to make sure loop
    // terminates
    // nchanges = local_nchanges;
		// printf("proc_id: %d nchanges: %d\n", proc_id, nchanges); //DEBUG
		// printf("proc_id: %d local_nchanges: %d\n", proc_id, local_nchanges); //DEBUG
		// printf("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!"); //DEBUG
    // Now from the root proc I can print out the current iteration information

    if (proc_id == root_proc) {
			printf("%3d: %5d |", curiter, nchanges);
      for (int c = 0; c < nclust; c++) { 
        printf(" %4d", local_clust->counts[c]);
      }
			
			printf("\n");
    }
		
    curiter += 1;  // TODO: unsure if this is still alright or if I should also sync this with other procs
    // printf("proc %d, curiter: %d\n", proc_id, curiter); //DEBUG
		
    // After we decide on the best cluster for all our local features we need to
    // do an all-to-one we can use a gather to get everyone's local features
    // back into the global_data struct everyone sends data back to root proc

		if (proc_id != root_proc) {
			MPI_Gather(local_features, local_ndata * local_clust->dim, MPI_FLOAT,
						NULL, local_ndata * local_clust->dim, MPI_FLOAT,
						root_proc, MPI_COMM_WORLD);
		}
		else {
					MPI_Gather(local_features, local_ndata * local_clust->dim, MPI_FLOAT,
										global_data->features, local_ndata * local_clust->dim, MPI_FLOAT,
										root_proc, MPI_COMM_WORLD);
		}
		
    // we also need to gather the local_assigns back to the root
		if (proc_id == root_proc) {
				MPI_Gather(local_assigns, local_ndata, MPI_INT, global_data->assigns,
							local_ndata, MPI_INT, root_proc, MPI_COMM_WORLD);
		}
		else {
			MPI_Gather(local_assigns, local_ndata, MPI_INT, NULL,
							local_ndata, MPI_INT, root_proc, MPI_COMM_WORLD);
		}
    
    printf("end of p:%d iter:%d\n", proc_id, curiter);
		
  }
  
  printf("end here p:%d iter:%d\n", proc_id, curiter);
  // At this point everyone should free their own locally allocated information
  // freeing stuff from the scatter

  free(feature_counts);
  free(feature_displ);
  // freeing stuff used in the actuall algorithm
  free(local_features);
  free(local_assigns);

  // The following code is all done inside root. We only really need to print
  // the confusion matrix inside one proc
  if (proc_id == root_proc) {
    if (curiter > MAXITER) {
      printf("WARNING: maximum iteration %d exceeded, may not have conveged",
             MAXITER);
    } else {
      printf("CONVERGED: after %d iterations", curiter);
    }
    printf("\n");
    // CONFUSION MATRIX
    int confusion[global_data->nlabels]
                 [nclust];  // initilizing the confusion martix on the stack
    for (int i = 0; i < global_data->nlabels; i++) {
      for (int j = 0; j < nclust; j++) {
        confusion[i][j] = 0;
      }
    }
    for (int i = 0; i < global_data->ndata; i++) {
      confusion[global_data->labels[i]][global_data->assigns[i]] += 1;
    }
    printf("==CONFUSION MATRIX + COUNTS==\n");
    printf("LABEL \\ CLUST");

    // confusion matrix header
    printf("%2s\n", "");
    for (int j = 0; j < local_clust->nclust; j++) {
      printf(" %4d", j);
    }
    printf(" %4s\n", "TOT");

    int tot = 0;

    // each row of confusion matrix
    for (int i = 0; i < global_data->nlabels; i++) {
      printf("%2d:", i);
      tot = 0;
      for (int j = 0; j < local_clust->nclust; j++) {
        printf(" %4d", confusion[i][j]);
        tot += confusion[i][j];
      }
      printf(" %4d\n", tot);
    }
    // final total row of confusion matrix
    printf("TOT");
    tot = 0;
    for (int c = 0; c < local_clust->nclust; c++) {
      printf(" %4d", local_clust->counts[c]);
      tot += local_clust->counts[c];
    }
    printf(" %4d\n\n", tot);
    // recall the size of savedir is argv[3]
    char* outfile =
        malloc(strlen(argv[3]) + strlen("/labels.txt") + strlen(argv[3]));
    sprintf(outfile, "%s/labels.txt", savedir);
    printf("Saving cluster labels to file %s\n", outfile);

    FILE* file = fopen(outfile, "w");
    for (int i = 0; i < global_data->ndata; i++) {
      fprintf(file, "%2d %2d\n", global_data->labels[i],
              global_data->assigns[i]);
			
    }
    fclose(file);
    save_pgm_files(local_clust, savedir);
		
    free(outfile);
		
  }

  freeKMClust(local_clust);
  // Mischalenous frees
  free(savedir);
  // FIXME: unsure if we need to deallocate global_data for other procs that
  // arent root google says it fine to not because I initialized it to null if
  // we are root we have to free the global_data
	
  if (proc_id == root_proc) {
    freeKMData(global_data);
  }
	printf("---------------------------------------------------------------------------PROC_ID: %d ---------------------------------------------------------------------------\n", proc_id);
  MPI_Finalize();
	
  return 0;
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
  ssize_t tot_tokens = 0;  // number of tokens in datafile
  ssize_t tot_lines = 0;   // number of lines in datafile

  // Getting file data we need to allocate correct amount of space
  int fileStats = filestats(datafile, &tot_tokens, &tot_lines);

  // allocating space for the number of labels in the dataset
  data->labels = (int*)calloc(tot_lines, sizeof(int));
  // length of features array
  int featuresLength = tot_tokens / tot_lines - 2;
  // allocating space for all the features arrays
  data->features =
      malloc(tot_lines * featuresLength *
             sizeof(float));  // allocating a 2d array for the features

  int ndata = 0;            // keeping track of ndata
  int currentIntToken = 0;  // used to store the current feature token
  char colon[1];
  for (int i = 0; i < tot_lines; i++) {
    ndata++;
    fscanf(fin, "%d %s", &currentIntToken, colon);
    data->labels[i] = currentIntToken;  // appending label to labels array
    for (int j = 0; j < featuresLength; j++) {
      fscanf(fin, "%d", &currentIntToken);  
      data->features[i * featuresLength + j] =
          currentIntToken;  // appending feature to feature array
    }                        
  }
  fclose(fin);
  data->ndata = ndata;         // number of feature arrays
  data->dim = featuresLength;  // the length of each feature array they are all
                               // the same length
  data->nlabels = intMax(data->labels, tot_lines) +
                  1;  // note since I increment the labelIdx when I add a new
                      // label this should be the length
  data->assigns =
      malloc(sizeof(int) * data->ndata);  // allocating assigns array for later
  memset(data->assigns, 0,
         sizeof(int) * data->ndata);  // zerioing out assigns for now

  return data;
}

KMClust* kmclust_new(int nclust, int dim) {
  KMClust* clust = malloc(sizeof(KMClust));
  memset(clust, 0, sizeof(KMClust));  // zeroing out all data
  clust->nclust = nclust;
  clust->dim = dim;

  clust->features = malloc(sizeof(float) * nclust * dim);

  clust->counts = malloc(sizeof(int) * nclust);

  for (int c = 0; c < nclust; c++) {
    for (int d = 0; d < dim; d++) {
      clust->features[c * dim + d] = 0.0;
    }
    clust->counts[c] = 0;
  }
  return clust;
}
void save_pgm_files(KMClust* clust, char* savedir) {
	
  int nclust = clust->nclust;
  int dim = clust->dim;
  int dim_root = (int)sqrt(dim);
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
      FILE* pgm = fopen(outfile, "w+");

      fprintf(pgm, "P2\n");

      fprintf(pgm, "%d %d\n", dim_root, dim_root);

      fprintf(pgm, "%.0f", maxfeat);

      for (int d = 0; d < dim; d++) {
        if ((d > 0 && d % dim_root) == 0) {
          // fwrite("\n", 1, 1, pgm);
          fprintf(pgm, "\n");
        }
        fprintf(pgm, "%3.0f ", clust->features[c * dim + d]);
        // int result = round(c * dim + d);
        // fprintf(pgm, "%3d ", result);
      }
      // fwrite("\n", 1, 1, pgm);
			
      fprintf(pgm, "\n");
			// printf("writing cluster: %d\n", c); //DEBUG
      fclose(pgm);
			
    }
		
  }
}
