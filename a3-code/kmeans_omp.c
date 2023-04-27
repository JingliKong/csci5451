#include "kmeans_util.h"
#include <omp.h>

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

  while ((nchanges > 0) && (curiter <= MAXITER)) {
    // DETERMINE NEW CLUSTER CENTERS
    // reset cluster centers to 0.0
    for (int c = 0; c < clust->nclust; c++) {
      for (int d = 0; d < clust->dim; d++) {
        clust->features[c * clust->dim + d] = 0.0;
      }
    }
    // cluster locks 
    omp_lock_t *cluster_locks = (omp_lock_t *) malloc(clust->nclust * sizeof(omp_lock_t));
    // Initialize the locks
    for (int i = 0; i < clust->nclust; i++) {
        omp_init_lock(&cluster_locks[i]);
    }
    // sum up data in each cluster
    #pragma omp parallel for
    for (int i = 0; i < data->ndata; i++) {
      int c = data->assigns[i];
      for (int d = 0; d < clust->dim; d++) {
        omp_set_lock(&cluster_locks[c]);
        clust->features[c * clust->dim + d] += data->features[i * clust->dim + d];
        omp_unset_lock(&cluster_locks[c]);
      }
    }
    // Destroy the locks
    for (int i = 0; i < clust->nclust; i++) {
        omp_destroy_lock(&cluster_locks[i]);
    }
    free(cluster_locks);

    // divide by ndatas of data to get mean of cluster center
    // since all threads are doing a division on a differnt portion of the array all we have to do is a normal parallel for loop though we still have false sharing
    #pragma omp for
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
    // initializing locks for when we update counts 
    omp_lock_t *counts_locks = (omp_lock_t *)malloc(clust->nclust * sizeof(omp_lock_t)); // we need as many locks as cluster so we can update each cluster count
    for (int i = 0; i < clust->nclust; i++) {
        omp_init_lock(&counts_locks[i]);
    }
    nchanges = 0;
		int local_nchanges = 0;
		#pragma omp parallel for reduction(+:local_nchanges)
    for (int i = 0; i < data->ndata; i++) {
      int best_clust = INT_MIN;
      float best_distsq = INFINITY;
      for (int c = 0; c < clust->nclust; c++) {
        float distsq = 0.0;
        for (int d = 0; d < clust->dim; d++) {
          float diff = data->features[i * clust->dim + d] - clust->features[c * clust->dim + d];
          distsq += diff * diff;
        }
        if (distsq < best_distsq) {
          best_clust = c;
          best_distsq = distsq;
        }
      }
    	omp_set_lock(&counts_locks[best_clust]);
      clust->counts[best_clust] += 1;
			omp_unset_lock(&counts_locks[best_clust]);

      if (best_clust != data->assigns[i]) {
        local_nchanges += 1;
        data->assigns[i] = best_clust;
      }
    }
		nchanges = local_nchanges;
		for (int i = 0; i < clust->nclust; i++) {
			omp_destroy_lock(&counts_locks[i]);
		}
		free(counts_locks);
		
    printf("%3d: %5d |", curiter, nchanges);
    for (int c = 0; c < nclust; c++) {
      printf(" %4d", clust->counts[c]);
    }
    printf("\n");
    curiter += 1;
  }

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