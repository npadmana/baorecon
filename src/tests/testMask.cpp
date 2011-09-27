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
      pp.TPMReadSerial("dm_1.0000.bin", 2000.0);

      VecGetOwnershipRange(pp.px, &lo, &hi);
      PetscSynchronizedPrintf(PETSC_COMM_WORLD,"%10llu --> %10llu\n",lo, hi);
      PetscSynchronizedFlush(PETSC_COMM_WORLD);

      // Count the number of particles not in the mask 
      int gc;
      Shell ss(0.0, 0.0, 0.0, 500.0, 1800.0);
      PetscPrintf(PETSC_COMM_WORLD, "Test... %i\n",(int) ss(480.0, 0.0, 0.0));
      PetscPrintf(PETSC_COMM_WORLD, "Test... %i\n",(int) ss(0.0, 1200.0, 0.0));
      PetscPrintf(PETSC_COMM_WORLD, "Test... %i\n",(int) ss(1000.0, 0.0, 1700.0));
      gc = pp.CountMask(ss);
      PetscPrintf(PETSC_COMM_WORLD,"In shell = %i\n",gc);

      // Trim 
      pp.TrimMask(ss);
      VecGetOwnershipRange(pp.px, &lo, &hi);
      PetscSynchronizedPrintf(PETSC_COMM_WORLD,"%10llu --> %10llu\n",lo, hi);
      PetscSynchronizedFlush(PETSC_COMM_WORLD);
      gc = pp.CountMask(ss);
      PetscPrintf(PETSC_COMM_WORLD,"In shell = %i\n",gc);
      PetscPrintf(PETSC_COMM_WORLD,"Testing npart..... %i\n",pp.npart);


    }


    // Only call this when everything is out of scope
    ierr=PetscFinalize(); CHKERRQ(ierr);

}
