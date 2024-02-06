/* This is a sample Advection solver in C 
The advection equation-> \partial q / \partial t - u \cdot \nabla q(x,y) = 0
The grid of NX by NX evenly spaced points are used for discretization.  
The first and last points in each direction are boundary points. 
Approximating the advection operator by 1st order finite difference. 
*/
# include <stdio.h>
# include <stdlib.h>
# include <time.h>
# include <stdbool.h>
# include <math.h>
# include "advection.h"

#define BUFSIZE 512
/* ************************************************************************** */
int main ( int argc, char *argv[] ){
  if(argc!=2){
    printf("Usage: ./levelSet input.dat\n");
    return -1;  
  }
  static int frame=0;

  // Create an advection solver
  solver_t advc; 
  // Create uniform rectangular (Cartesian) mesh
  advc.msh = createMesh(argv[1]); 
  // Create time stepper 
  tstep_t tstep = createTimeStepper(advc.msh.Nnodes); 
  // Create Initial Field
  initialCondition(&advc);

  // Read input file for time variables 
  tstep.tstart = readInputFile(argv[1], "TSART");
  tstep.tend   = readInputFile(argv[1], "TEND");
  tstep.dt     = readInputFile(argv[1], "DT");
  tstep.time = 0.0; 

  // adjust time step size 
  int Nsteps = ceil( (tstep.tend - tstep.tstart)/tstep.dt);
  tstep.dt = (tstep.tend - tstep.tstart)/Nsteps;

  // Read input file for OUTPUT FREQUENCY i.e. in every 1000 steps
  int Noutput = readInputFile(argv[1], "OUTPUT_FREQUENCY");


  // write the initial solution i.e. q at t = tstart
  {
    char fname[BUFSIZ];
    sprintf(fname, "test_%04d.csv", frame++);
    solverPlot(fname, &advc.msh, advc.q);
  }


  // ********************Time integration***************************************/
  // for every steps
  for(int step = 0; step<Nsteps; step++){
    // for every stage
    for(int stage=0; stage<tstep.Nstage; stage++){
      // Call integration function
      RhsQ(&advc, &tstep, stage); 
    }

    tstep.time = tstep.time+tstep.dt;

    if(step%Noutput == 0){
      char fname[BUFSIZ];
      sprintf(fname, "test_%04d.csv", frame++);
      solverPlot(fname, &advc.msh, advc.q);
    }
  }
  free(advc.msh.N2N);
  free(advc.msh.x);
  free(advc.msh.y);
}

/* ************************************************************************** */
/*
-> is used to access the members of an object which is not directly availble but a pointer to that object is available 
for example, 
struct Mesh {
    int NX, NY;
};
Mesh actualMesh;
Mesh* msh = &actualMesh;
actualMesh.NX = 10; // Access using dot because actualMesh is not a pointer.
msh->NX = 10; // Access using arrow because msh is a pointer to Mesh.
*/
void RhsQ(solver_t *solver, tstep_t *tstep, int stage){
mesh_t *msh = &solver->msh;
  for (int j = 0; j < msh->NY; j++) {
    for (int i = 0; i < msh->NX; i++) {
      double duq_dx = 0.0;
      double dvq_dy = 0.0;
      int index = j * msh->NX + i;

      // Using modular arithmetic for periodic boundary conditions
      int i_west = (i - 1 + msh->NX) % msh->NX;
      int i_east = (i + 1) % msh->NX;
      int j_south = (j - 1 + msh->NY) % msh->NY;
      int j_north = (j + 1) % msh->NY;

      // Compute indices for neighbors with periodic wrapping
      int index_west = j * msh->NX + i_west;
      int index_east = j * msh->NX + i_east;
      int index_south = j_south * msh->NX + i;
      int index_north = j_north * msh->NX + i;


      // Calculate duq_dx with upwind differencing
      if (solver->u[index] >= 0) {
        duq_dx = ((solver->u[index] * solver->q[index]) - (solver->u[index_west] * solver->q[index_west])) / (msh->x[index] - msh->x[index_west]);
      } else {
        duq_dx = ((solver->u[index_east] * solver->q[index_east]) - (solver->u[index] * solver->q[index])) / (msh->x[index_east] - msh->x[index]);
      }
      
      
      // Calculate dvq_dy with upwind differencing
      if (solver->v[index] >= 0) {
        dvq_dy = ((solver->v[index] * solver->q[index]) - (solver->v[index_south] * solver->q[index_south])) / (msh->y[index] - msh->y[index_south]) ;
      } else {
        dvq_dy = ((solver->v[index_north] * solver->q[index_north]) - (solver->v[index] * solver->q[index])) / (msh->y[index_north] - msh->y[index]); 
      }

      tstep->rhsq[index] = -(duq_dx + dvq_dy);
      /* 
       Time integration in 2 steps 
       Step 1: Update residual
       resq = rk4a(stage)* resq + dt*rhsq
       Step:2 Update solution and store
       q = q + rk4b(stage)*resq
      */

     // Store updated solutions in solver->q and solver->resq
      tstep->resq[index] = ((tstep->rk4a[stage]) * tstep->resq[index]) + (tstep->dt * tstep->rhsq[index]); 
      solver->q[index] = solver->q[index] + (tstep->rk4b[stage] * tstep->resq[index]);
    }
  } 
}
//  } 
//}

/* ************************************************************************** */
void initialCondition(solver_t *solver){
  mesh_t *msh = &(solver->msh); 
  #define pi   3.14159265358979323846
  solver->q = (double *)malloc(msh->Nnodes*sizeof(double)); 
  // solver->u = (double *)malloc(2*msh->Nnodes*sizeof(double));
  solver->u = (double *)malloc(msh->Nnodes*sizeof(double));
  solver->v = (double *)malloc(msh->Nnodes*sizeof(double));

  double xc = 0.5;
  double yc = 0.75;
  double r = 0.15;

  for(int j=0; j<msh->NY; j++){
    for(int i=0; i<msh->NX; i++){
      int index = j*msh->NX + i;
      double x = msh->x[index];
      double y = msh->y[index];

      solver->q[index] = sqrt(pow(x - xc, 2) + pow(y - yc, 2)) - r;
      solver->u[index] = sin(4 * pi * (x + 0.5)) * sin(4 * pi * (y + 0.5));
      solver->v[index] = cos(4 * pi * (x + 0.5)) * cos(4 * pi * (y + 0.5));
      // solver->u[index+msh->Nnodes] = cos(4 * pi * (x + 0.5)) * cos(4 * pi * (y + 0.5));
      // solver->v[index] = cos(4*pi*(x+0.5))   * cos(4*pi*(y+0.5));
    }
  }
}



/* ************************************************************************** */
// void createMesh(struct mesh *msh){
mesh_t createMesh(char* inputFile){

  mesh_t msh; 

  // Read required fields i.e. NX, NY, XMIN, XMAX, YMIN, YMAX
  msh.NX   = readInputFile(inputFile, "NX");
  msh.NY   = readInputFile(inputFile,"NY");
  msh.xmin = readInputFile(inputFile,"XMIN");
  msh.xmax = readInputFile(inputFile,"XMAX");
  msh.ymin = readInputFile(inputFile,"YMIN");
  msh.ymax = readInputFile(inputFile,"YMAX");

  msh.Nnodes = msh.NX*msh.NY;
  msh.x = (double *) malloc(msh.Nnodes*sizeof(double));
  msh.y = (double *) malloc(msh.Nnodes*sizeof(double));

  /*
  Compute Coordinates of the nodes
  */
  double step_x = (msh.xmax - msh.xmin)/(msh.NX-1);
  double step_y = (msh.ymax - msh.ymin)/(msh.NY-1);
  for(int j=0; j<msh.NY; j++){
    for(int i=0; i<msh.NX; i++){
      int index = (j*msh.NX)+i;
      msh.x[index]= msh.xmin + step_x * i;
      msh.y[index]= msh.ymin + step_y * j;
    }
  }

  // Create connectivity and periodic connectivity
  /* 
  for every node 4 connections east north west and south
  Note that periodic connections require specific treatment
  */
  msh.N2N = (int *)malloc(4*msh.Nnodes*sizeof(int));

  for(int j=0; j<msh.NY; j++){
    for(int i=0; i<msh.NX; i++){
      int index = (j*msh.NX)+i;
      int start_index = 4*index;
      
    msh.N2N[start_index]     = (j * msh.NX) + ((i + 1) % msh.NX);            // East
    msh.N2N[start_index + 1] = ((j + 1) % msh.NY) * msh.NX + i; // North
    msh.N2N[start_index + 2] = j * msh.NX + ((i - 1 + msh.NX) % msh.NX);   // West
    msh.N2N[start_index + 3] = ((j - 1 + msh.NY) % msh.NY) * msh.NX + i; // South
    }
  }
  return msh; 
}

/* ************************************************************************** */
void solverPlot(char *fileName, mesh_t *msh, double *Q){
    FILE *fp = fopen(fileName, "w");
    if (fp == NULL) {
        printf("Error opening file\n");
        return;
    }

    fprintf(fp, "X,Y,Z,Q \n");
    for(int n=0; n< msh->Nnodes; n++){
      fprintf(fp, "%.8f, %.8f,%.8f,%.8f\n", msh->x[n], msh->y[n], 0.0, Q[n]);
    } 
}

/* ************************************************************************** */
double readInputFile(char *fileName, char* tag){
  /* Author: Muhammad Ahmed 
    Date: 11/4/2023
    This Function reads the input.dat file*/
  // create a file pointer
  FILE *fp = fopen(fileName, "r");
  if (fp == NULL) {
    printf("Error opening the input file\n");
    return -1;
  }
  // define the buffer char array
  char buffer[BUFSIZE];
  // data array is used to store the tag read from the file for comparision
  char data[20] = {0};
  double val=0.0;
  bool foundtag = false;
  // This function reads the file until EOF BUT Breaks if TAG found before the EOF
  while(fgets(buffer,BUFSIZE,fp)!=NULL){
      int i= 1;
      if (buffer[0]=='['){
          while (i<19 && buffer[i]!=']' && buffer[i] != '\0' ) {
            data[i-1]=buffer[i];
            ++i;
          }
      data[i-1] = '\0';
      if (strcmp(data,tag)==0){
        foundtag = true;
        break;
      }
      memset(data,0,sizeof(data));
    }
  }
  // IF tag is found we need to read the next line to read its value
  if (foundtag==true && (fgets(buffer,BUFSIZE,fp)) != NULL) {
    // we should keep in mind that function returns float byt NY and NX values are integers
      if (strcmp(tag,"NY") ==0 ||  strcmp(tag,"NX") == 0){
          int int_val=0;
          sscanf(buffer,"%d",&int_val);
          val = int_val;
        }
      else{
          sscanf(buffer,"%lf",&val);
      }
  }
  // if no tag present 
  else {
        val= -1;
      }
  fclose(fp);
  return val;
}


/* ************************************************************************** */
// Time stepper clas RK(4-5)
// resq = rk4a(stage)* resq + dt*rhsq
//  q = q + rk4b(stage)*resq
tstep_t createTimeStepper(int Nnodes){
  tstep_t tstep; 
  tstep.Nstage = 5; 
  tstep.resq = (double *)calloc(Nnodes,sizeof(double)); 
  tstep.rhsq = (double *)calloc(Nnodes,sizeof(double));
  tstep.rk4a = (double *)malloc(tstep.Nstage*sizeof(double));
  tstep.rk4b = (double *)malloc(tstep.Nstage*sizeof(double));
  tstep.rk4c = (double *)malloc(tstep.Nstage*sizeof(double));

  tstep.rk4a[0] = 0.0; 
  tstep.rk4a[1] = -567301805773.0/1357537059087.0; 
  tstep.rk4a[2] = -2404267990393.0/2016746695238.0;
  tstep.rk4a[3] = -3550918686646.0/2091501179385.0;
  tstep.rk4a[4] = -1275806237668.0/842570457699.0;
        
  tstep.rk4b[0] = 1432997174477.0/9575080441755.0;
  tstep.rk4b[1] = 5161836677717.0/13612068292357.0; 
  tstep.rk4b[2] = 1720146321549.0/2090206949498.0;
  tstep.rk4b[3] = 3134564353537.0/4481467310338.0;
  tstep.rk4b[4] = 2277821191437.0/14882151754819.0;
             
  tstep.rk4c[0] = 0.0;
  tstep.rk4c[1] = 1432997174477.0/9575080441755.0;
  tstep.rk4c[2] = 2526269341429.0/6820363962896.0;
  tstep.rk4c[3] = 2006345519317.0/3224310063776.0;
  tstep.rk4c[4] = 2802321613138.0/2924317926251.0;
  return tstep; 
}