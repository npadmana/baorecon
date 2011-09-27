// Useful to run this with -info -mat_view_info for useful information
#include <iostream>
#include <cmath>

#include "Recon.h"

const double BoxSize=2000.0;
static char help[] = "A test program\n";

using namespace std;

int main(int argc, char *args[]) {
    
    PetscErrorCode ierr;
    ierr=PetscInitialize(&argc,&args,(char *) 0, help); CHKERRQ(ierr);
    {
      Particle pp;
      Vec grid, smooth, pot;
      PetscInt Ngrid;
      PetscTruth flg;
      PetscLogDouble t1, t2;
      int lo, hi;
      int rank;

      MPI_Comm_rank(PETSC_COMM_WORLD, &rank);
      PetscOptionsGetInt("-n", &Ngrid, &flg);
      if (!flg) Ngrid=64; // Testing, after all
      PetscPrintf(PETSC_COMM_WORLD, "Using Ngrid=%i\n", (int) Ngrid);
      
      // Allocate potential solver
      vector<double> origin(3, 0.0);
      PotentialSolve psolve(Ngrid, BoxSize);
      PetscGetTime(&t1);
      psolve.SetupOperator(RADIAL, 0.4, origin);
      PetscGetTime(&t2);
      PetscPrintf(PETSC_COMM_WORLD,"Elapsed time = %e\n", t2-t1);

    }


    // Only call this when everything is out of scope
    ierr=PetscFinalize(); CHKERRQ(ierr);

}
