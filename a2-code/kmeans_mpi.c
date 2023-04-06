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
#include <limits.h>
#include <mpi.h>


int filestats(char *filename, ssize_t *tot_tokens, ssize_t *tot_lines); 
int intMax (int *arr, int len); 


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


int main(int argc, char **argv) {
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

    MPI_Init (&argc, &argv);
    int proc_id, total_procs;
    MPI_Comm_rank (MPI_COMM_WORLD, &proc_id);     // get current process id 
    MPI_Comm_size (MPI_COMM_WORLD, &total_procs); // get number of processes 

    int root_processor = 0; 
    // Initializing variable for all procs to access 
    KMData* global_data = NULL;  
	// everyone needs to know the dimension and ndata 
    if (proc_id == root_processor) {
        global_data = kmdata_load(datafile); 

        printf("ndata: %d\n", global_data->ndata);
        printf("dim: %d\n\n", global_data->dim);
        // we still want to assign every data to a cluster be for we scatter the data off to other procs 
        for (int i = 0; i < global_data->ndata; i++){
                int c = i % global_clust->nclust;
                global_data->assigns[i] = c; // give every feature array a cluster 
        }

        // Calculating every num features per cluster 
        for (int c = 0; c < global_clust->nclust; c++){
                float icount = global_data->ndata / global_clust->nclust;
                float extra = (c < (global_data->ndata % global_clust->nclust)) ? 1 : 0;
                global_clust->counts[c] = icount + extra; // counts is saying how may features are per cluster
        }       
    } 

    // everyone initializing these for a future scatterv to distribute features among the procs 
    int *feature_counts = calloc(total_procs, sizeof(int)); 
    int *feature_displ = calloc(total_procs, sizeof(int));
    // setting up variables for scatterv essentially the same as the code pack (scatterv_demo.c in 04-mpi-code)
    int features_per_proc = global_data->ndata / total_procs; // TODO: Ask about this unsure that this is how we decide how many features go to each proc  
    int feature_surplus = global_data->ndata % total_procs; 

    for(int i = 0; i<total_procs; i++){
        feature_counts[i] = (i < feature_surplus) ? features_per_proc+1 : features_per_proc;
        feature_displ[i] = (i == 0) ? 0 : feature_displ[i-1] + feature_counts[i-1];
    }		 

    float *local_features = calloc(feature_counts[proc_id], sizeof(float)); 
    // Everyone calls scatterv the root is going to give each proc a portion of the feature array 
    MPI_Scatterv(global_data->features, feature_counts, feature_displ, MPI_FLOAT,
                            local_features, feature_counts[proc_id], MPI_FLOAT,
                            root_processor, MPI_COMM_WORLD); 
    // everyone initilaizes own cluster 
    KMClust* local_clust = kmclust_new(nclust, global_data->dim); 

    // At this point everyone should have their own feature array to work on so I can start the main algorithm 
    // Main Algorithm
    int curiter = 1; 
    int nchanges = localKMData->ndata; 
    if (proc_id == root_processor) {
        printf("==CLUSTERING: MAXITER %d==\n", MAXITER);
        printf("ITER NCHANGE CLUST_COUNTS\n");
    } 
    while ((nchanges > 0) && (curiter <= MAXITER)) {
        //DETERMINE NEW CLUSTER CENTERS
        //reset cluster centers to 0.0
        for (int c = 0; c < local_clust->nclust; c++){ 
            for (int d = 0; d < local_clust->dim; d++){
                local_clust->local_features[c * local_clust->dim + d] = 0.0;
            }
        }
        //sum up data in each cluster 
        for (int i = 0; i < data->ndata; i++){
            int c = data->assigns[i];
            for (int d = 0; d < clust->dim; d++){
                local_clust->features[c * clust->dim + d] += data->features[i * clust->dim + d]; 
            }
        }    
        // TODO After the summing we need to do a All-to-All to make sure that all local clusters get updated to be the same     
        if (proc_id == root_processor) {
            printf("%3d: %5d |", curiter, nchanges);
            for (int c = 0; c < nclust; c++){
                printf(" %4d", clust->counts[c]);
            }
            printf("\n");
            curiter += 1;            
        }
    }			
    // Everyone frees their own local variables 
    free(feature_counts);
    free(feature_displ); 
    freeKMData(localKMData); 
    if (proc_id == root_processor) {
        // root frees the global data 
        freeKMData(global_data);
        freeKMClust(global_clust);  
    }
		
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
    data->ndata = ndata; // number of feature arrays 
    data->dim = featuresLength; // the length of each feature array they are all the same length  
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

