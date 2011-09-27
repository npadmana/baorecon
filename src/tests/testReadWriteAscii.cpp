#include <iostream>
#include <cmath>

#include "Recon.h"

static char help[] = "A test program\n";

using namespace std;

int main(int argc, char *args[]) {
    
    PetscErrorCode ierr;
    ierr=PetscInitialize(&argc,&args,(char *) 0, help); CHKERRQ(ierr);
    {
      Particle pp, pp1;
      int lo, hi;
      int rank;

      MPI_Comm_rank(PETSC_COMM_WORLD, &rank);
      pp.AsciiReadSerial("in.dat");
      pp.AsciiWriteSerial("out.dat");
      pp1.AsciiReadWeightedSerial("inweight.dat");
      pp1.AsciiWriteWeightedSerial("outweight.dat");
    }


    // Only call this when everything is out of scope
    ierr=PetscFinalize(); CHKERRQ(ierr);

}
