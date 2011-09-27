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
      PetscInt lo, hi;
      int rank;

      MPI_Comm_rank(PETSC_COMM_WORLD, &rank);
      DensityGrid dg(256, 1.0);
      pp.TPMReadSerial("dm_1.0000.bin");
      // Print out initial ownership ranges
      VecGetOwnershipRange(pp.px, &lo, &hi);
      PetscSynchronizedPrintf(PETSC_COMM_WORLD,"%10llu --> %10llu\n",lo, hi);
      PetscSynchronizedFlush(PETSC_COMM_WORLD);

      // Test slab decomp
      pp.SlabDecompose(dg);
      bool good = dg.TestSlabDecompose(pp);
      if (good) 
         {PetscSynchronizedPrintf(PETSC_COMM_WORLD,"Slab decomposition succeeded on process %i\n",rank);}
      else 
         {PetscSynchronizedPrintf(PETSC_COMM_WORLD,"Slab decomposition FAILED on process %i\n",rank);}
      PetscSynchronizedFlush(PETSC_COMM_WORLD);

      // Print out final ownership ranges
      VecGetOwnershipRange(pp.px, &lo, &hi);
      PetscSynchronizedPrintf(PETSC_COMM_WORLD,"%10llu --> %10llu\n",lo, hi);
      PetscSynchronizedFlush(PETSC_COMM_WORLD);

    }


    // Only call this when everything is out of scope
    ierr=PetscFinalize(); CHKERRQ(ierr);

}
