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
      PetscInt  Ngrid;
      PetscBool flg;
      int rank;

      // Read in the data
      MPI_Comm_rank(PETSC_COMM_WORLD, &rank);
      PetscOptionsGetInt(NULL, "-n", &Ngrid, &flg);
      if (!flg) Ngrid=64; // Testing, after all
      PetscPrintf(PETSC_COMM_WORLD, "Using Ngrid=%i\n", (int) Ngrid);
      DensityGrid dg(Ngrid, 2000.0);
      pp.TPMReadSerial("dm_1.0000.bin", 2000.0);
      pp.SlabDecompose(dg);
      
      //CIC and compute the corrfunc the old fashioned way
      grid = dg.Allocate();
      dg.CIC(grid, pp);
      PkStruct Xi1(10.0, 3.0, 60);
      dg.XiFFT(grid, 2.0, Xi1);
      VecDestroy(&grid);


      // Generate the shell density grid
      Shell ss(0.0, 0.0, 0.0, 200.0, 1800.0);
      Delta del1(Ngrid, 2000.0, 4, 0.75, ss);
      del1.BuildDensityGrid(pp, grid);
      //BuildDensityGrid(Ngrid, 2000.0, ss, pp, 4, 0.75, grid, W);

      // Compute xi
      PkStruct Xi2(10.0,3.0,60);
      dg.XiFFT_W(grid, del1.W,2.0,Xi2);
      VecDestroy(&grid);


      // Print out results
      double _xi1, _r1, _xi2, n1;
      FILE *fp;
      PetscFOpen(PETSC_COMM_WORLD,"testxi_shell.dat","w",&fp);
      for (int ii = Xi1.lo; ii < Xi1.hi; ++ii) {
        _xi1 = Xi1(ii, _r1, n1);
        _xi2 = Xi2(ii);
        if (n1>0) PetscSynchronizedFPrintf(PETSC_COMM_WORLD, fp, "%6i %9.3f %15.8e %15.8e\n",ii,_r1,_xi1, _xi2);
      }
      PetscSynchronizedFlush(PETSC_COMM_WORLD);
      PetscFClose(PETSC_COMM_WORLD, fp);


    }


    // Only call this when everything is out of scope
    ierr=PetscFinalize(); CHKERRQ(ierr);

}
