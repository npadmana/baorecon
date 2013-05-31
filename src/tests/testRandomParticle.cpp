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
      pp.RandomInit(10000000, 1.0, 2913);

      VecGetOwnershipRange(pp.px, &lo, &hi);
      PetscSynchronizedPrintf(PETSC_COMM_WORLD,"%10llu --> %10llu\n",lo, hi);
      PetscSynchronizedFlush(PETSC_COMM_WORLD);

      double sum, min, max;
      VecSum(pp.px, &sum); VecMin(pp.px, NULL, &min); VecMax(pp.px, NULL, &max);
      PetscPrintf(PETSC_COMM_WORLD,"px average= %10.6f, min=%10.6f, max-1=%10.6e\n",sum/pp.npart, min, max-1.0);
      VecSum(pp.py, &sum); VecMin(pp.py,NULL, &min); VecMax(pp.py,NULL, &max);
      PetscPrintf(PETSC_COMM_WORLD,"py average= %10.6f, min=%10.6f, max-1=%10.6e\n",sum/pp.npart, min, max-1.0);
      VecSum(pp.pz, &sum); VecMin(pp.pz,NULL, &min); VecMax(pp.pz,NULL, &max);
      PetscPrintf(PETSC_COMM_WORLD,"pz average= %10.6f, min=%10.6f, max-1=%10.6e\n",sum/pp.npart, min, max-1.0);

    }


    // Only call this when everything is out of scope
    ierr=PetscFinalize(); CHKERRQ(ierr);

}
