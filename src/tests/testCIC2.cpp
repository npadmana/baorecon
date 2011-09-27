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
      int lo, hi;
      int rank;

      MPI_Comm_rank(PETSC_COMM_WORLD, &rank);
      DensityGrid dg(256, 1.0);
      grid = dg.Allocate();
      pp.GridInitialize(256, 0.2,0.7,0.14);
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



      // Print out 
      dg.Pull(grid);
      dg.slab(lo, hi);
      int icount;
      FILE *fp;
      PetscFOpen(PETSC_COMM_WORLD,"testCIC.out","w", &fp);
      for (int ix=lo; ix < hi; ++ix) 
        for (int iy=0; iy < dg.Ng; ++iy) 
          for (int iz=0; iz < dg.Ng; ++iz) {
            icount = (ix*dg.Ng + iy)*dg.Ng + iz;
            if (icount < 10000) 
              PetscSynchronizedFPrintf(PETSC_COMM_WORLD,fp,"%6i %6i %6i %6i %15.8e\n",icount, ix,iy, iz, dg(ix, iy, iz));
            }
       PetscSynchronizedFlush(PETSC_COMM_WORLD);
       PetscFClose(PETSC_COMM_WORLD,fp);
       dg.Push(grid, INSERT_VALUES, false);

      // Cleanup
      _mydestroy(grid);
    }


    // Only call this when everything is out of scope
    ierr=PetscFinalize(); CHKERRQ(ierr);

}
