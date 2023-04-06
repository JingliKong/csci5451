#include <stdio.h>
#include <mpi.h>
#include <stdlib.h>

// // #define MGET(mat, i, j) ((mat)->data[((i)*((mat)->ncols)) + (j)])
// // #define MSET(mat, i, j, x) ((mat)->data[((i)*((mat)->ncols)) + (j)] = (x))
// #define HGET(mat, i, j) (mat[i * (width) + j])
// #define HSET()
// // #define LGET()
// // #define LSET()
#define PROC_GET(proc_data, cols_per_proc, i, j) ((proc_data)[(i * cols_per_proc) + (j)])
#define PROC_SET(proc_data, cols_per_proc, i, j, x) (((proc_data)[(i * cols_per_proc) + (j)]) = (x))


int main(int argc, char **argv){
    if(argc < 4){
        printf("usage: %s max_time width print\n", argv[0]);
        printf("  max_time: int\n");
        printf("  width: int\n");
        printf("  print: 1 print output, 0 no printing\n");
        return 0;
    }

    int max_time = atoi(argv[1]); // Number of time steps to simulate
    int width = atoi(argv[2]);    // Number of cells in the rod
    int print = atoi(argv[3]);
    double initial_temp = 50.0;   // Initial temp of internal cells 
    double L_bound_temp = 20.0;   // Constant temp at Left end of rod
    double R_bound_temp = 10.0;   // Constant temp at Right end of rod
    double k = 0.5;               // thermal conductivity constant
    // double **H;                   // 2D array of temps at times/locations
    double* H;

    //Begin MPI code:
    MPI_Init (&argc, &argv); 

    //MPI variables
    int proc_id, total_procs, cols_per_proc;
    MPI_Comm_rank (MPI_COMM_WORLD, &proc_id);     // get current process id 
    MPI_Comm_size (MPI_COMM_WORLD, &total_procs); // get number of processes 
    int root_processor = 0; // have the root processor be processor 0

    //Confirm unsupported configurations fail
    if ((width % total_procs) != 0) {
        printf("Number of columns not divisible by procs\n");
        return 0;

    } else if ((cols_per_proc = width / total_procs) < 3) {
        printf("Not emough columns per proc\n");
        return 0;
    }
    
    // Allocate space for local data
    double* proc_data = calloc(max_time * cols_per_proc, sizeof(double));
    

    if (proc_id == 0) {
        // Allocate memory for collective data
        // H = malloc(sizeof(double*)*max_time); 
        H = malloc(sizeof(double*)*max_time*width);
        // for(int t = 0; t < max_time; t++){
        //     // H[t] = malloc(sizeof(double)*width);
        //     H[t] = calloc(width, sizeof(double)); 
        // }

        // // Initialize constant left/right boundary temperatures
        // for(int t=0; t < max_time; t++){
        //     H[t][0] = L_bound_temp;
        //     H[t][width-1] = R_bound_temp;
        // }
        // // Initialize temperatures at time 0
        // for(int p=1; p < width-1; p++){
        //     H[0][p] = initial_temp;
        // }

        // Proc 0 initializes partition of H to starting conditions 
        for (int i = 1; i < cols_per_proc; i++) {
            PROC_SET(proc_data, cols_per_proc, 0, i, initial_temp);
        }
        // Initialize constant left boundary temperatures
        for(int t=0; t < max_time; t++){
            PROC_SET(proc_data, cols_per_proc, t, 0, L_bound_temp); 
        }

    }

    if (proc_id == total_procs - 1) { // we are the leftmost portion of the pipe 
        // Proc n-1 initializes partition of H to starting conditions 
        int i = (total_procs == 1) ? 1 : 0;
        for (; i < cols_per_proc - 1; i++) {
            PROC_SET(proc_data, cols_per_proc, 0, i, initial_temp);
        }
        // Initialize constant right boundary temperatures
        for(int t=0; t < max_time; t++){
            PROC_SET(proc_data, cols_per_proc, t, cols_per_proc-1, R_bound_temp);
        }

    }

    if ((proc_id != 0) && (proc_id != total_procs - 1)) { // we are in the middle all the initial temps are just 0 
        for (int i = 0; i < cols_per_proc; i++) {
            PROC_SET(proc_data, cols_per_proc, 0, i, initial_temp);
        }
    }
    
    // Simulate the temperature changes for internal cells
    for(int t = 0; t < max_time-1; t++){
        for(int p = 1; p < cols_per_proc - 1; p++){
            // printf("i: %d, LEFTVAL: %f, RIGHTVAL: %f\n", p, PROC_GET(proc_data, cols_per_proc, t, p - 1), PROC_GET(proc_data, cols_per_proc, t, p + 1));
            double prev = PROC_GET(proc_data, cols_per_proc, t, p);
            // printf("PREV: %f\n", prev);

            // printf("t:%d ", t);
            double left_diff = prev - PROC_GET(proc_data, cols_per_proc, t, p - 1);
            double right_diff = prev - PROC_GET(proc_data, cols_per_proc, t, p + 1);
            double delta = -k*( left_diff + right_diff);
            // printf("E: %f, A: %f", prev, proc_data[t][p]); 
            
            // printf(" %5.1f:",prev);

            // printf(" index:%d \n", (t+1) * cols_per_proc + p);
            proc_data[(t+1) * cols_per_proc + p] = prev+delta;
            // PROC_SET(proc_data, cols_per_proc, t+1, p, prev + delta);
            // PROC_SET(proc_data, cols_per_proc, t+1, p, -123);
            // printf(":%5.1f ",PROC_GET(proc_data, cols_per_proc, t + 1, p));


            
        }
        // printf("proc:%d --> t: %d\n", proc_id, t);

        

        // Communicate accross processors
        double recv_left, recv_right;
        double send_left, send_right; 

        if (total_procs <= 1) {
            // printf("\n");
            // printf("should not happen");

            continue;
        } else if (proc_id == 0) { // Leftmost processor case
            send_right = PROC_GET(proc_data, cols_per_proc, t, cols_per_proc - 1);
            MPI_Sendrecv(&send_right, 1, MPI_DOUBLE, 1, 1,
                          &recv_right, 1, MPI_DOUBLE, 1, 1, 
                          MPI_COMM_WORLD, MPI_STATUS_IGNORE);

            double left = PROC_GET(proc_data, cols_per_proc, t, cols_per_proc - 2);
            double delta = -k*(2*send_right - left - recv_right);
            // PROC_SET(proc_data, cols_per_proc, t+1, cols_per_proc-1, send_right + delta);
            proc_data[(t+1) * cols_per_proc + (cols_per_proc-1)] = send_right+delta;
    
        } else if ((proc_id == (total_procs - 1))) { // Rightmost processor case
            send_left = PROC_GET(proc_data, cols_per_proc, t, 0);
            MPI_Sendrecv(&send_left, 1, MPI_DOUBLE, proc_id - 1, 1,
                          &recv_left, 1, MPI_DOUBLE, proc_id - 1, 1, 
                          MPI_COMM_WORLD, MPI_STATUS_IGNORE);

            double right = PROC_GET(proc_data, cols_per_proc, t, 1); 
            double delta = -k*(2*send_left - recv_left - right);
            // PROC_SET(proc_data, cols_per_proc, t+1, 0, send_left + delta);
            proc_data[(t+1) * cols_per_proc] = send_left+delta;          
            
        } else { // Middle processor case 
            // We need 2 MPI_Sendrecv one to the proc to the right and one to the left 
            // sending my right most element to the recv_left because we are sending 
            if (proc_id % 2 == 0) {
                send_right = PROC_GET(proc_data, cols_per_proc, t, cols_per_proc - 1); 
                MPI_Sendrecv(&send_right, 1, MPI_DOUBLE, proc_id + 1, 1, 
                            &recv_right, 1, MPI_DOUBLE, proc_id + 1, 1,
                            MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                send_left = PROC_GET(proc_data, cols_per_proc, t, 0);             
                MPI_Sendrecv(&send_left, 1, MPI_DOUBLE, proc_id - 1, 1,
                            &recv_left, 1, MPI_DOUBLE, proc_id - 1, 1, 
                            MPI_COMM_WORLD, MPI_STATUS_IGNORE); 
            } else {
                send_left = PROC_GET(proc_data, cols_per_proc, t, 0);              
                MPI_Sendrecv(&send_left, 1, MPI_DOUBLE, proc_id - 1, 1,
                            &recv_left, 1, MPI_DOUBLE, proc_id - 1, 1, 
                            MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                send_right = PROC_GET(proc_data, cols_per_proc, t, cols_per_proc - 1); 
                MPI_Sendrecv(&send_right, 1, MPI_DOUBLE, proc_id + 1, 1, 
                            &recv_right, 1, MPI_DOUBLE, proc_id + 1, 1,
                            MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            }


            // Then we just calculate the ends of our local array using the values we just recieved 
            // calculating the end 
            double left = PROC_GET(proc_data, cols_per_proc, t, 0);
            double left_diff = left - recv_left;  
            double right_diff = left - PROC_GET(proc_data, cols_per_proc, t, 1);
            double delta = -k*( left_diff + right_diff );
            // PROC_SET(proc_data, cols_per_proc, t+1, 0, send_left + delta); 
            proc_data[(t+1) * cols_per_proc] = send_left+delta;
             

            // now computing for the rightmost element of local array 
            double right = PROC_GET(proc_data, cols_per_proc, t, cols_per_proc - 1);
            left_diff = right - PROC_GET(proc_data, cols_per_proc, t, cols_per_proc - 2);
            right_diff =  right - recv_right; 
            delta = -k*( left_diff + right_diff );
            // PROC_SET(proc_data, cols_per_proc, t+1, cols_per_proc - 1, send_right + delta);  
            proc_data[(t+1) * cols_per_proc + (cols_per_proc-1)] = send_right+delta;
        }
    }

    // Gather local data and store into corresponding section of 1D array
    for (int i = 0; i < max_time; i++){
         MPI_Gather(&proc_data[i*cols_per_proc], cols_per_proc, MPI_DOUBLE,
                    &H[i*width], cols_per_proc, MPI_DOUBLE, 
                    root_processor, MPI_COMM_WORLD);
    }


    // Print out the temperatures
    if((proc_id == 0) && print){
        // Column headers
        printf("%3s| ","");
        for(int p=0; p<width; p++){
            printf("%5d ",p);
        }
        printf("\n");
        printf("%3s+-","---");
        for(int p=0; p<width; p++){
            printf("------");
        }
        printf("\n");
            // Row headers and data
            for(int t=0; t<max_time; t++){
                printf("%3d| ",t);
                for(int p=0; p<width; p++){
                // printf("%5.1f ",H[t][p]); // check value of p
                printf("%5.1f ",H[t*width + p]);
                }
                printf("\n");
            }
    }

    // Clean up and deallocation

    free(proc_data);
    if (proc_id == 0){
        free(H);
    }

    MPI_Finalize();
    return 0;
}