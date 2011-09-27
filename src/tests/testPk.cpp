#include <iostream>
#include <cmath>

#include "Recon.h"

static char help[] = "A test program\n";

using namespace std;

int main(int argc, char *args[]) {
    
    PetscErrorCode ierr;
    ierr=PetscInitialize(&argc,&args,(char *) 0, help); CHKERRQ(ierr);
    {
      Particle pp;
      Vec grid;
      PetscInt lo, hi;
      int rank;

      MPI_Comm_rank(PETSC_COMM_WORLD, &rank);
      DensityGrid dg(256, 2000.0);
      pp.TPMReadSerial("dm_1.0000.bin", 2000.0);
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
      grid = dg.Allocate();
      dg.CIC(grid, pp);
      double sum;
      VecSum(grid, &sum);
      PetscPrintf(PETSC_COMM_WORLD, "Sum = %f\n",sum); 
      VecNorm(grid, NORM_2, &sum);
      sum /= sqrt(pow((double) dg.Ng, 3));
      PetscPrintf(PETSC_COMM_WORLD, "stdev = %f\n",sum); 

      // Compute Pk(k)
      double k1, pk1, n1, pk2;
      PkStruct pkmean(0.001, 0.005, 100), pkdec(0.001, 0.005, 100);
      dg.PkCIC(grid, pkmean, pkdec);
      FILE *fp;
      PetscFOpen(PETSC_COMM_WORLD,"testpk.dat","w",&fp);
      for (int ii = pkmean.lo; ii < pkmean.hi; ++ii) {
        pk1 = pkmean(ii, k1, n1);
        pk2 = pkdec(ii);
        if (n1 > 0) PetscSynchronizedFPrintf(PETSC_COMM_WORLD, fp, "%6i %9.3f %15.8e %15.8e\n",ii,k1,pk1,pk2);
      }
      PetscSynchronizedFlush(PETSC_COMM_WORLD);
      PetscFClose(PETSC_COMM_WORLD, fp);

      VecDestroy(grid);
    }


    // Only call this when everything is out of scope
    ierr=PetscFinalize(); CHKERRQ(ierr);

}
