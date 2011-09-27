#include <iostream>
#include <cmath>

#include "Recon.h"

static char help[] = "A test program\n";

using namespace std;

int main(int argc, char *args[]) {
    
    PetscErrorCode ierr;
    ierr=PetscInitialize(&argc,&args,(char *) 0, help); CHKERRQ(ierr);
    {
      Vec grid;
      double *data;
      int lo, hi;
      int rank;
      PetscInt Ng;

      MPI_Comm_rank(PETSC_COMM_WORLD, &rank);
      DensityGrid dg(4, 1.0);
      grid = dg.Allocate();
      Ng = dg.Ng;
      dg.slab(lo, hi);
      dg.Pull(grid);
      for (int ix=lo; ix < hi; ++ix) 
        for (int iy=0; iy < Ng; ++iy) 
          for (int iz=0; iz < Ng; ++iz) 
            dg(ix, iy, iz) = (double) ((ix*Ng+iy)*Ng+iz);
      dg.Push(grid, INSERT_VALUES, false);
      dg.Pull(grid);
      for (int ix=lo-1; ix < hi+1; ++ix) 
        for (int iy=0; iy < Ng; ++iy) 
          for (int iz=0; iz < Ng; ++iz) 
            PetscSynchronizedPrintf(PETSC_COMM_WORLD, "%3i %3i %3i %5i\n",ix,iy,iz,(int) dg(ix,iy,iz));
      PetscSynchronizedPrintf(PETSC_COMM_WORLD,"---------\n");
      PetscSynchronizedFlush(PETSC_COMM_WORLD);


      PetscPrintf(PETSC_COMM_WORLD,"---------------------------------\n");
      VecGetArray(grid,&data);
      for (int ix = 0; ix < 96; ++ix) 
        PetscSynchronizedPrintf(PETSC_COMM_WORLD, "%4i %4i\n",ix, (int) data[ix]);
      PetscSynchronizedPrintf(PETSC_COMM_WORLD,"---------\n");
      PetscSynchronizedFlush(PETSC_COMM_WORLD);


      // Cleanup
      _mydestroy(grid);
    }


    // Only call this when everything is out of scope
    ierr=PetscFinalize(); CHKERRQ(ierr);

}
