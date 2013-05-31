#include <iostream>
#include <fstream>
#include <cmath>

#include "Recon.h"

static char help[] = "A test program\n";

using namespace std;

int main(int argc, char *args[]) {
    
    PetscErrorCode ierr;
    ierr=PetscInitialize(&argc,&args,(char *) 0, help); CHKERRQ(ierr);
    {
      Vec grid;
      PetscInt lo, hi;
      int rank, nk;
      PetscInt seed;
      PetscBool flg;
      vector<double> kvec, pkvec;
      double k1, pk1, pk2, n1;
      
      // Get seed for constrained realization
      PetscOptionsGetInt(NULL, "-seed", &seed, &flg);
      if (!flg) seed=200; 

      // MPI Rank
      MPI_Comm_rank(PETSC_COMM_WORLD, &rank);
      
      // Rank 0 job reads in pk function and broadcasts to everyone
      if (rank == 0) {
        ifstream fp("pk.dat");
        do {
          fp >> k1 >> pk1;
          if (fp) {kvec.push_back(k1); pkvec.push_back(pk1 + 1.e3);} // Add a shot-noise piece to this 
        } while (fp);
        fp.close();
        nk = kvec.size();
      }
      MPI_Bcast(&nk, 1, MPI_INT, 0, PETSC_COMM_WORLD);
      if (rank !=0) {kvec.resize(nk, 0.0); pkvec.resize(nk, 0.0);}
      MPI_Bcast(&kvec[0], nk, MPI_DOUBLE, 0, PETSC_COMM_WORLD);
      MPI_Bcast(&pkvec[0], nk, MPI_DOUBLE, 0, PETSC_COMM_WORLD);

      DensityGrid dg(256, 2000.0);

      //CIC
      grid = dg.Allocate();
      dg.FakeGauss(100, grid, kvec, pkvec);

      // Define a window function 
      Shell ss(0.0, 0.0, 0.0, 200.0, 1800.0);
      Delta del(256, 2000.0, 2, 0.75, ss);

      // Save the grid vector and apply the window function to it
      Vec grid_save;
      VecDuplicate(grid, &grid_save); 
      VecPointwiseMult(grid, del.W, grid); 
      VecCopy(grid, grid_save);
      double sum;
      VecSum(grid_save, &sum); PetscPrintf(PETSC_COMM_WORLD,"grid sum=%f\n",sum);
      VecNorm(grid_save, NORM_2, &sum);PetscPrintf(PETSC_COMM_WORLD,"grid norm = %f\n", sum);

      // Hoffman-Ribak
      del.HoffmanRibak(grid, kvec, pkvec, seed);

      // Test to see that the constraints have been satisfied
      VecAXPY(grid_save, -1.0, grid); 
      VecPointwiseMult(grid_save, del.W, grid_save);
      VecNorm(grid_save, NORM_2, &sum);
      PetscPrintf(PETSC_COMM_WORLD, "residual norm = %f\n", sum);

      

      // Compute Pk(k)
      PkStruct pkmean(0.001, 0.005, 100), pkdec(0.001, 0.005, 100);
      dg.PkCIC(grid, pkmean, pkdec);
      FILE *fp;
      PetscFOpen(PETSC_COMM_WORLD,"testfake.dat","w",&fp);
      for (int ii = pkmean.lo; ii < pkmean.hi; ++ii) {
        pk1 = pkmean(ii, k1, n1);
        if (n1 > 0) PetscSynchronizedFPrintf(PETSC_COMM_WORLD, fp, "%6i %9.3f %15.8e\n",ii,k1,pk1);
      }
      PetscSynchronizedFlush(PETSC_COMM_WORLD);
      PetscFClose(PETSC_COMM_WORLD, fp);

      VecDestroy(&grid); VecDestroy(&grid_save);
    }


    // Only call this when everything is out of scope
    ierr=PetscFinalize(); CHKERRQ(ierr);

}
