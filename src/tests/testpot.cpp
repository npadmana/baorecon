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
      Vec grid, pot;
      PetscInt Ngrid;
      int rank;
      PetscBool flg;

      MPI_Comm_rank(PETSC_COMM_WORLD, &rank);
      PetscOptionsGetInt(NULL,"-n", &Ngrid, &flg);
      if (!flg) Ngrid=64; // Testing, after all
      PetscPrintf(PETSC_COMM_WORLD, "Using Ngrid=%i\n", (int) Ngrid);

    
      DensityGrid dg(Ngrid, BoxSize);
      grid = dg.Allocate();
      pp.TPMReadSerial("dm_1.0000.bin", BoxSize);
      pp.SlabDecompose(dg);

      // Test slab decomp
      bool good = dg.TestSlabDecompose(pp);
      if (good) 
         {PetscSynchronizedPrintf(PETSC_COMM_WORLD,"Slab decomposition succeeded on process %i\n",rank);}
      else 
         {PetscSynchronizedPrintf(PETSC_COMM_WORLD,"Slab decomposition FAILED on process %i\n",rank);}
      PetscSynchronizedFlush(PETSC_COMM_WORLD);
      if (!good) RAISE_ERR(99, "Slab decomposition failed");

      //CIC
      dg.CIC(grid, pp);
      double sum;
      VecSum(grid, &sum);
      PetscPrintf(PETSC_COMM_WORLD, "Sum = %f\n",sum); 
      VecNorm(grid, NORM_2, &sum);
      sum /= sqrt(pow((double) dg.Ng, 3));
      PetscPrintf(PETSC_COMM_WORLD, "stdev = %f\n",sum); 

      // Allocate potential solver
      PotentialSolve psolve(Ngrid, BoxSize);
      psolve.SetupOperator(REALPBC);
      pot = dg.Allocate();
      psolve.Solve(grid, pot);
      VecSum(pot, &sum);
      PetscPrintf(PETSC_COMM_WORLD, "Sum = %f\n",sum);

      // Test result
      Vec negdelta;
      Mat a = psolve.GetOperator();
      VecDuplicate(pot, &negdelta);
      MatMult(a,pot,negdelta);
      VecAXPY(negdelta,1.0,grid);
      VecNorm(negdelta, NORM_2, &sum);
      PetscPrintf(PETSC_COMM_WORLD, "norm = %f\n",sum); 
      MatDestroy(&a); _mydestroy(negdelta);


      // Cleanup
      _mydestroy(grid);
      _mydestroy(pot);

    }


    // Only call this when everything is out of scope
    ierr=PetscFinalize(); CHKERRQ(ierr);

}
