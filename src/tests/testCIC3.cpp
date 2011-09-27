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
      Vec grid, grid2, W;
      int lo, hi;
      int rank;

      MPI_Comm_rank(PETSC_COMM_WORLD, &rank);
      DensityGrid dg(512, 2000.0);
      grid = dg.Allocate();
      pp.TPMReadSerial("dm_1.0000.bin", 2000.0);
      pp.SlabDecompose(dg);


      //CIC
      dg.CIC(grid, pp);
      // Generate the shell density grid
      Shell ss(0.0, 0.0, 0.0, -100.0, 6000.0);
      Delta del(512, 2000.0, 4, 0.75, ss);
      del.BuildDensityGrid(pp, grid2);
      //BuildDensityGrid(64, 2000.0, ss, pp, 4, 0.75, grid2, W);

      


      // Print out 
      DensityGrid dg2(64,2000.0); DensityGrid dgw(64, 2000.0);
      dg.Pull(grid); dg2.Pull(grid2);  dgw.Pull(del.W);
      dg.slab(lo, hi);
      int icount;
      FILE *fp;
      PetscFOpen(PETSC_COMM_WORLD,"testCIC3.out","w", &fp);
      for (int ix=lo; ix < hi; ++ix) 
        for (int iy=0; iy < dg.Ng; ++iy) 
          for (int iz=0; iz < dg.Ng; ++iz) {
            icount = (ix*dg.Ng + iy)*dg.Ng + iz;
            if (icount < 10000) 
              PetscSynchronizedFPrintf(PETSC_COMM_WORLD,fp,"%6i %6i %6i %6i %15.8e %15.8e %15.8e\n",icount, 
                  ix,iy, iz, dg(ix, iy, iz), dg2(ix, iy, iz), dgw(ix, iy, iz));
            }
       PetscSynchronizedFlush(PETSC_COMM_WORLD);
       PetscFClose(PETSC_COMM_WORLD,fp);

      // Cleanup
      _mydestroy(grid);
      _mydestroy(grid2);
      _mydestroy(W);
    }


    // Only call this when everything is out of scope
    ierr=PetscFinalize(); CHKERRQ(ierr);

}
