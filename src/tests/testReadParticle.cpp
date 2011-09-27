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
      pp.TPMReadSerial("dm_1.0000.bin");

      VecGetOwnershipRange(pp.px, &lo, &hi);
      PetscSynchronizedPrintf(PETSC_COMM_WORLD,"%10llu --> %10llu\n",lo, hi);
      PetscSynchronizedFlush(PETSC_COMM_WORLD);

    }


    // Only call this when everything is out of scope
    ierr=PetscFinalize(); CHKERRQ(ierr);

}
