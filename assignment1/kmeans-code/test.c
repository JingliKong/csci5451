
#include <stdio.h>
#include <stdlib.h>

int main(int argc, char *argv[]) {
    // Write C code here
    FILE *pgm = fopen("testFile.txt", "w");
    fwrite("Hello\n", 7, 1, pgm); 
    fwrite("\n", 1, 2, pgm); 
    fclose(pgm);  
    return 0; 
}


// int main(int argc, char *argv[]) {
//     // Write C code here
//     int nclust = 3; 
//     int dim = 2; 
//     float **features = malloc(sizeof(float)*nclust*dim); 
//     for (int c = 0; c < nclust; c++) {
//         features[c] = calloc(dim, sizeof(float)); 
//     }
//     for (int i = 0; i < nclust; i++) {
//         for (int j = 0; j < dim; j++) {
//             printf("%f ", features[i][j]); 
//         }
//         printf("\n");
//     } 
//     return 0;
// }