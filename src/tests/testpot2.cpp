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
      PetscBool flg;
      int lo, hi;
      int rank;

      MPI_Comm_rank(PETSC_COMM_WORLD, &rank);
      PetscOptionsGetInt(NULL, "-n", &Ngrid, &flg);
      if (!flg) Ngrid=64; // Testing, after all
      PetscPrintf(PETSC_COMM_WORLD, "Using Ngrid=%i\n", (int) Ngrid);
      
      
      DensityGrid dg(Ngrid, BoxSize);
      DensityGrid dg2(Ngrid, BoxSize);
      DensityGrid dg3(Ngrid, BoxSize);
      grid = dg.Allocate(); smooth = dg.Allocate();
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

      // Smooth
      VecCopy(grid, smooth);
      dg.GaussSmooth(smooth, 20.0);


      // Allocate potential solver
      PotentialSolve psolve(Ngrid, BoxSize);
      psolve.SetupOperator(REALPBC);
      pot = dg.Allocate();
      psolve.Solve(smooth, pot);
      VecSum(pot, &sum);
      PetscPrintf(PETSC_COMM_WORLD, "Sum = %f\n",sum);

      // Print out 
      dg.Pull(grid);
      dg2.Pull(smooth);
      dg3.Pull(pot);
      dg.slab(lo, hi);
      int icount;
      FILE *fp;
      PetscFOpen(PETSC_COMM_WORLD,"testCIC.out","w", &fp);
      for (int ix=lo; ix < hi; ++ix) 
        for (int iy=0; iy < dg.Ng; ++iy) 
          for (int iz=0; iz < dg.Ng; ++iz) {
            icount = (ix*dg.Ng + iy)*dg.Ng + iz;
            if (icount < 10000) 
              PetscSynchronizedFPrintf(PETSC_COMM_WORLD,fp,"%6i %6i %6i %6i %15.8e %15.8e %15.8e\n",icount, ix,iy, iz, 
                    dg(ix, iy, iz), dg2(ix, iy, iz), dg3(ix,iy,iz));
            }
       PetscSynchronizedFlush(PETSC_COMM_WORLD);
       PetscFClose(PETSC_COMM_WORLD,fp);

      // Cleanup
      _mydestroy(grid);
      _mydestroy(pot);
      _mydestroy(smooth);
    }


    // Only call this when everything is out of scope
    ierr=PetscFinalize(); CHKERRQ(ierr);

}
