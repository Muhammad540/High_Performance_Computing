# include <stdio.h>
# include <stdlib.h>
# include <math.h>
# include <time.h>
# include <string.h>
# include <mpi.h>

#define BUFSIZE 512


// Function definitions
/******************************************************************************/
int main ( int argc, char *argv[] );
double exactSoln( double c, double x, double y, double t );
void applyBC(double *data,  double *x, double *y, double c, double time, int nx, int ny);
void solverPlot(char *fileName, double *x, double *y, int nx, int ny, double *data); 
double readInputFile(char *fileName, char* tag); 

// Solver Info
/******************************************************************************
  Purpose:
    wave2d solves the wave equation in parallel using MPI.
  Discussion:
    Discretize the equation for u(x,t):
      d^2 u/dt^2  =  c^2 * (d^2 u/dx^2 + d^2 u/dy^2)  
      for 0 < x < 1, 0 < y < 1, t>0
    with boundary conditions and Initial conditions obtained from the exact solutions:
      u(x,y, t) = sin ( 2 * pi * ( x - c * t ) )
   Usage: serial -> ./wave input.dat  parallel> mpirun -np 4 ./wave input.dat 
******************************************************************************/

int main ( int argc, char *argv[] ){

  // Read input file for solution parameters
  double tstart = readInputFile(argv[1], "TSART"); // Start time
  double tend   = readInputFile(argv[1], "TEND");  // End time
  double dt     = readInputFile(argv[1], "DT");    // Time step size

  // Global node number in x and y
  int NX        = (int) readInputFile(argv[1], "NX"); // Global node numbers in x direction
  int NY        = (int) readInputFile(argv[1], "NY"); // Global node numbers in y direction

  double xmax = readInputFile(argv[1], "XMAX"); // domain boundaries
  double xmin = readInputFile(argv[1], "XMIN"); // domain boundaries
  double ymax = readInputFile(argv[1], "YMAX"); // domain boundaries
  double ymin = readInputFile(argv[1], "YMIN"); // domain boundaries
  double c = readInputFile(argv[1], "CONSTANT_C");

  double *qn, *q0, *q1;               // Solution field at t+dt, t and t-dt
  static int frame = 0; 

  // Initialize MPI  
  MPI_Init(&argc, &argv);
  double start_time = MPI_Wtime();
  // get the rank number
  int rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  // get the number of ranks
  int size;
  MPI_Comm_size(MPI_COMM_WORLD, &size);
  
  // DOMAIN DECOMPOSITION
  int nx = NX;           // local number of nodes in x direction
  int ny = NY/size + 2;  // local number of nodes in y direction add 2 for ghost nodes 
  // should be equally divided between the ranks, plus 2 for the ghost nodes

  // // ALLOCATE MEMORY for COORDINATES (x, y) and compute them
  double *x = ( double * ) malloc ( nx*ny * sizeof ( double ) );
  double *y = ( double * ) malloc ( nx*ny * sizeof ( double ) );
  // find uniform spacing in x and y directions
  double hx = (xmax - xmin)/(NX-1.0); 
  double hy = (ymax - ymin)/(NY-1.0); 
  // // Compute coordinates of the nodes 
    int internal_ny = NY / size;
    // Now calculate the y-coordinate without including ghost nodes
    for (int j = 0; j < ny; ++j) {
        for (int i = 0; i < nx; ++i) {
            double xn = xmin + i * hx;
            double yn;
            // The y-coordinate should not include the ghost nodes
            if (j == 0 || j == ny-1) {
                // Handle ghost nodes separately, if needed
                yn = (j == 0) ? ymin + (rank * internal_ny - 1) * hy : ymin + ((rank + 1) * internal_ny) * hy;
            } else {
                // Calculate the y-coordinate for internal nodes
                yn = ymin + ((rank * internal_ny) + (j - 1)) * hy;
            }
            x[i + j * nx] = xn;
            y[i + j * nx] = yn;
        }
    } 


  // ALLOCATE MEMORY for SOLUTION and its HISTORY
  // Solution at time (t+dt)
  qn = ( double * ) malloc ( nx*ny * sizeof ( double ) );
  // Solution at time (t)
  q0 = ( double * ) malloc ( nx*ny * sizeof ( double ) );
  // Solution at time t-dt
  q1 = ( double * ) malloc ( nx*ny * sizeof ( double ) );

  // USE EXACT SOLUTION TO FILL HISTORY
   for(int i=0; i<nx; i++){
      for(int j=1; j<ny; j++){
      const double xn = x[i+ j*nx]; 
      const double yn = y[i+ j*nx]; 
      // Exact solutions at history tstart and tstart+dt
      q0[i + j*nx] = exactSoln(c, xn, yn, tstart + dt);  
      q1[i + j*nx] = exactSoln(c, xn, yn, tstart);  
    }
  }

 
  // Write the initial solution 
  {
    char fname[BUFSIZ];
    sprintf(fname, "test_%04d.csv", frame++);
    solverPlot(fname, x, y, nx, ny, q1);
  }

// RUN SOLVER 
  int Noutput = 10000; 
  int Nsteps=(tend - tstart)/dt;     // Assume  dt divides (tend- tstart)
  double alphax2 = pow((c*dt/hx),2); 
  double alphay2 = pow((c*dt/hy),2);
  
  // We already have 2 steps computed with exact solution
  double time = dt; 
  // for every time step
  for(int tstep = 2; tstep<=Nsteps+1; ++tstep){
    // increase  time
    time = tstart + tstep*dt; 
    
    // Apply Boundary Conditions i.e. at i, j = 0, i,j = nx-1, ny-1
    applyBC(q0, x, y, c, time, nx, ny); 
    // MPI communication
    // send the ghost nodes to the neighbouring ranks and receive the ghost nodes from the neighbouring ranks
    // communicate the ghost nodes in the y direction
    // defination of MPI_Sendrecv: MPI_Sendrecv(sendbuf, sendcount, sendtype, dest, sendtag, recvbuf, recvcount, recvtype, source, recvtag, comm, status)
    if(rank == 0){
      MPI_Sendrecv(&q0[nx*(ny-2)], nx, MPI_DOUBLE, rank+1, 0, 
                  &q0[nx*(ny-1)], nx, MPI_DOUBLE, rank+1, 0, 
                  MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    }
    else if(rank == size-1){
      MPI_Sendrecv(&q0[nx], nx, MPI_DOUBLE, rank-1, 0, 
                  &q0[0], nx, MPI_DOUBLE, rank-1, 0, 
                  MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    }
    else{
      MPI_Sendrecv(&q0[nx*(ny-2)], nx, MPI_DOUBLE, rank+1, 0, 
                  &q0[nx*(ny-1)], nx, MPI_DOUBLE, rank+1, 0, 
                  MPI_COMM_WORLD, MPI_STATUS_IGNORE);
      MPI_Sendrecv(&q0[nx], nx, MPI_DOUBLE, rank-1, 0, 
                  &q0[0], nx, MPI_DOUBLE, rank-1, 0, 
                  MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    }
    // Update solution using second order central differencing in time and space
    // important to exclude the boundaries so i = 1 to nx-1 and j = 1 to ny-1 
    for(int i=1; i<nx-1; i++){ // exclude left right boundaries
      for(int j= 1; j<ny-1 ; j++){ // exclude top and bottom boundaries
        const int n0   = i + j*nx; 
        const int nim1 = i - 1 + j*nx; // node i-1,j
        const int nip1 = i + 1 + j*nx; // node i+1,j
        const int njm1 = i + (j-1)*nx; // node i, j-1
        const int njp1 = i + (j+1)*nx; // node i, j+1
        // update solution 
        qn[n0] = 2.0*q0[n0] - q1[n0] + alphax2*(q0[nip1]- 2.0*q0[n0] + q0[nim1])
                                     + alphay2*(q0[njp1] -2.0*q0[n0] + q0[njm1]); 
      }
    }
    // perform another round of MPI communication to exchange the updated boundary data between 
    // neighbouring ranks, to ensure that the ghost nodes have the correct values when we update the history
      if(rank == 0){
        MPI_Sendrecv(&qn[nx*(ny-2)], nx, MPI_DOUBLE, rank+1, 0, 
                    &qn[nx*(ny-1)], nx, MPI_DOUBLE, rank+1, 0, 
                    MPI_COMM_WORLD, MPI_STATUS_IGNORE);
      }
      else if(rank == size-1){
        MPI_Sendrecv(&qn[nx], nx, MPI_DOUBLE, rank-1, 0, 
                    &qn[0], nx, MPI_DOUBLE, rank-1, 0, 
                    MPI_COMM_WORLD, MPI_STATUS_IGNORE);
      }
      else{
        MPI_Sendrecv(&qn[nx*(ny-2)], nx, MPI_DOUBLE, rank+1, 0, 
                    &qn[nx*(ny-1)], nx, MPI_DOUBLE, rank+1, 0, 
                    MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        MPI_Sendrecv(&qn[nx], nx, MPI_DOUBLE, rank-1, 0, 
                    &qn[0], nx, MPI_DOUBLE, rank-1, 0, 
                    MPI_COMM_WORLD, MPI_STATUS_IGNORE);
      }
    // Update history q1 = q0; q0 = qn, except the boundaries
    for(int i=1; i<nx-1; i++){
      for(int j=1; j<ny-1; j++){
        q1[i + j*nx] = q0[i + j*nx]; 
        q0[i + j*nx] = qn[i + j*nx]; 
        
      }
    }

    // Dampout a csv file for postprocessing
    if (tstep % Noutput == 0) {
        char fname[BUFSIZ];
        sprintf(fname, "rank_%02d_test_%04d.csv", rank, frame++);
        solverPlot(fname, x, y, nx, ny, q0);
    }

  }

  // Compute Linf norm of error at tend
    double linf = 0.0; 
    for(int i=0; i<nx; i++){
      for(int j=0; j<ny; j++){
         double xn = x[i+ j*nx]; 
         double yn = y[i+ j*nx]; 
         // solution and the exact one
         double qn = q0[i+ j*nx]; 
         double qe = exactSoln(c, xn, yn, time);  
         linf  = fabs(qn-qe)>linf ? fabs(qn -qe):linf; 
      }
    }
    // use MPI_Reduce to find the maximum absolute value of the error at all the nodes
    double linf_global;
    // MPI_Reduce(sendbuf, recvbuf, count, datatype, op, root, comm)
    MPI_Reduce(&linf, &linf_global, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    if(rank == 0){
      printf("Infinity norm of the error: %.4e %.8e \n", linf, time);
    }
    // calculate the L2 norm of the solution vector qn
    // double l2 = 0.0;
    // for(int i=0; i<nx; i++){
    //   for(int j=0; j<ny; j++){
    //      double qn = q0[i+ j*nx]; 
    //      l2 += qn*qn; 
    //   }
    // }
    // l2 = sqrt(l2/(nx*ny));
    // double l2_global;
    // MPI_Reduce(&l2, &l2_global, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    // if(rank == 0){
    //   printf("L2 norm of the solution: %.4e %.8e \n", l2, time);
    // }
    
    // Now print the time taken
    double end_time = MPI_Wtime();
    // Only print once, not on every process 
    if (rank == 0) { 
        printf("Time taken: %f seconds\n", end_time - start_time);
    }
    MPI_Finalize();
    free(qn);
    free(q0);
    free(q1);
    free(x);
    free(y);

  return 0;
}

/***************************************************************************************/
double exactSoln( double c, double x, double y, double t){
  const double pi = 3.141592653589793;
  double value = sin( 2.0*pi*( x - c*t));
  return value;
}

/***************************************************************************************/
void applyBC(double *data,  double *x, double *y, double c, double time, int nx, int ny){

  // Apply Boundary Conditions
  double xn, yn; 

  for(int j=0; j<ny;++j){ // left right boundaries i.e. i=0 and i=nx-1
    xn = x[0 + j*nx]; 
    yn = y[0 + j*nx];    
    data[0 + j*nx] = exactSoln(c, xn, yn, time); 

    xn = x[nx-1 + j*nx]; 
    yn = y[nx-1 + j*nx];    
    data[nx-1 + j*nx] = exactSoln(c, xn, yn, time); 
  }

  
  for(int i=0; i< nx; ++i){ // top and  right boundaries i.e. j=0 and j=ny-1
    xn = x[i+ 0*nx]; 
    yn = y[i+ 0*nx]; 
    data[i + 0*nx] = exactSoln(c, xn, yn, time); 

    xn = x[i+ (ny-1)*nx]; 
    yn = y[i+ (ny-1)*nx];       
    data[i +  (ny-1)*nx] = exactSoln(c, xn, yn, time); 
  }
}

/* ************************************************************************** */
void solverPlot(char *fileName, double *x, double *y, int nx, int ny, double *Q){
    FILE *fp = fopen(fileName, "w");
    if (fp == NULL) {
        printf("Error opening file\n");
        return;
    }

    fprintf(fp, "X,Y,Z,Q \n");
     for(int i=0; i<nx; i++){
      for(int j=0; j<ny; j++){
        const double xn = x[i + j*nx]; 
        const double yn = y[i + j*nx]; 
        fprintf(fp, "%.8f, %.8f,%.8f,%.8f\n", xn, yn, 0.0, Q[i + j*nx]);
      }
    }
}


/* ************************************************************************** */
double readInputFile(char *fileName, char* tag){
  FILE *fp = fopen(fileName, "r");
  if (fp == NULL) {
    printf("Error opening the input file\n");
    return -1;
  }

  int sk = 0; 
  double result; 
  char buffer[BUFSIZE];
  char fileTag[BUFSIZE]; 
  while(fgets(buffer, BUFSIZE, fp) != NULL){
    sscanf(buffer, "%s", fileTag);
    if(strstr(fileTag, tag)){
      fgets(buffer, BUFSIZE, fp);
      sscanf(buffer, "%lf", &result); 
      return result;
    }
    sk++;
  }

  if(sk==0){
    printf("could not find the tag: %s in the file %s\n", tag, fileName);
  }
}