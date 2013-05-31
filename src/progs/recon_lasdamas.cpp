#include <iostream>
#include <cmath>
#include <sstream>
#include <iomanip>

#include "Recon.h"

static char help[] = "recon_realn -fn <TPMfile> -o <outfn> -L <boxsize> -N <grid=64> -bias <bias=1.0> -smooth <smooth=20.0> -rmin <rmin=10> -Nrbins <nbins=50> -dr <dr=3> -mask0 <mask Rmin=200.0> -mask1 <mask Rmax=1800.0> -maskthresh <mask thresh=0.75>\n";

using namespace std;


int main(int argc, char *args[])
// ---------------------------------------------------------------------------
// RS:  Las Damas reconstruction routine.  Uses the 50x randoms to build the
// survey mask.
// ---------------------------------------------------------------------------
{
    PetscErrorCode ierr;
    ierr=PetscInitialize(&argc,&args,(char *) 0, help); CHKERRQ(ierr);
    {
      /* ******************************
       * First we get the various options and print out useful information 
       * ********************************/
      PetscInt Ngrid, Nrbins;
      double boxsize, rmin,  dr, bias, smooth, mask0, mask1, maskthresh;
      char inpfn[200], inrfn[200], inmfn[200], outpfn[200], outrfn[200]; 
      PetscBool flg;
      ostringstream hdr;
      int rank; MPI_Comm_rank(PETSC_COMM_WORLD, &rank);

      PetscOptionsHasName(NULL, "-help", &flg);
      if (flg) exit(0);

      PetscOptionsGetString(NULL, "-fn", inpfn, 200, &flg);
      if (!flg) RAISE_ERR(99,"Specify inpfn");
      hdr << "# Galaxy file is " << inpfn << endl; 
      PetscOptionsGetString(NULL, "-fr", inrfn, 200, &flg);
      if (!flg) RAISE_ERR(99,"Specify inrfn");
      hdr << "# Randoms file is " << inrfn << endl; 
      PetscOptionsGetString(NULL, "-fm", inmfn, 200, &flg);
      if (!flg) RAISE_ERR(99,"Specify inmfn");
      hdr << "# Mask particles file is " << inmfn << endl; 
      PetscOptionsGetString(NULL, "-op", outpfn, 200, &flg);
      if (!flg) RAISE_ERR(99,"Specify outpfn");
      hdr << "# Shifted galaxy output file is " << outpfn << endl;
      PetscOptionsGetString(NULL, "-or", outrfn, 200, &flg);
      if (!flg) RAISE_ERR(99,"Specify outrfn");
      hdr << "# Shifted random output file is " << outpfn << endl;

      PetscOptionsGetInt(NULL, "-N", &Ngrid, &flg);
      if (!flg) Ngrid=64; 
      hdr << "# Ngrid=" << setw(5) << Ngrid << endl;
      PetscOptionsGetScalar(NULL, "-L", &boxsize, &flg);
      if (!flg) RAISE_ERR(99, "Set boxsize"); 
      hdr << "# boxsize=" << setw(8) << fixed << setprecision(2) << boxsize << endl;

      PetscOptionsGetScalar(NULL, "-bias", &bias, &flg);
      if (!flg) bias=1; 
      hdr << "# bias=" << setw(8) << setprecision(2) << bias << endl;
      PetscOptionsGetScalar(NULL, "-smooth", &smooth, &flg);
      if (!flg) smooth=20.0; 
      hdr << "# smooth=" << setw(8) << setprecision(2) << smooth << endl;

      PetscOptionsGetScalar(NULL, "-rmin", &rmin, &flg);
      if (!flg) rmin=10.0; 
      PetscOptionsGetScalar(NULL, "-dr", &dr, &flg);
      if (!flg) dr=3.0; 
      PetscOptionsGetInt(NULL, "-Nrbins", &Nrbins, &flg);
      if (!flg) Nrbins=50;
      hdr << "# " << setw(4) << Nrbins << " Xi bins from " << setw(8) << setprecision(2) << rmin 
        << " with spacing of " << dr << endl;
     
      // RS:  these are just dummy mask parameters
      mask0=200.0; mask1=1800.0; 
      Shell maskss (0.0, 0.0, 0.0, mask0, mask1);
      PetscOptionsGetScalar(NULL, "-maskthresh", &maskthresh, &flg);
      if (!flg) maskthresh=0.08; 
      hdr << "# Mask threshold = " << setprecision(2) << maskthresh
          << " particles per cell" << endl;

      // Print the header out
      PetscPrintf(PETSC_COMM_WORLD, (hdr.str()).c_str()); 

      /****************************************
       * Read in the particle data here and slab decompose
       ****************************************/

      // RS:  Read in all necessary lists of particles.  They will be
      // in the form of ASCII (x,y,z) lists in box coordinates.
      Particle pp, pr, pm;
      Vec grid;
      DensityGrid dg(Ngrid,boxsize);

      // ... Main "galaxies" list (pp)
      pp.AsciiReadSerial(inpfn,boxsize);
      PetscPrintf(PETSC_COMM_WORLD,"Read in %i galaxies...\n",pp.npart);
      pp.SlabDecompose(dg);
      bool good = dg.TestSlabDecompose(pp);
      if (good) PetscSynchronizedPrintf(PETSC_COMM_WORLD,
         "Slab decomposition succeeded on process %i\n",rank);
      else PetscSynchronizedPrintf(PETSC_COMM_WORLD,
         "Slab decomposition FAILED on process %i\n",rank);
      PetscSynchronizedFlush(PETSC_COMM_WORLD);
      if (!good) RAISE_ERR(99, "Slab decomposition failed");

      // ... Randoms list (pr)
      pr.AsciiReadSerial(inrfn,boxsize);
      PetscPrintf(PETSC_COMM_WORLD,"Read in %i randoms...\n",pr.npart);
      pr.SlabDecompose(dg);

      // ... Mask particle list (pm)
      pm.AsciiReadSerial(inmfn,boxsize);
      PetscPrintf(PETSC_COMM_WORLD,"Read in %i mask particles...\n",pm.npart);
      pm.SlabDecompose(dg);

      /*************************************************
       * Build density grid
       *************************************************/
      Delta del1(Ngrid, boxsize, 4, maskthresh, maskss, pm, smooth);
      del1.BuildDensityGrid(pp, grid);
      PetscPrintf(PETSC_COMM_WORLD, "Density grid computed.....\n");

      /************************************************
       * Now we solve for the potential 
       ************************************************/
      // Allocate potential solver
      Vec pot;
      PotentialSolve psolve(Ngrid, boxsize);
      pot = dg.Allocate();
      psolve.Solve(REALPBC, grid, pot, PETSC_NULL, bias);
      PetscPrintf(PETSC_COMM_WORLD,"Potential calculated....\n");
      
      /************************************************
       * Now we shift data and randoms
       ************************************************/ 
      // Generate random particles
      Vec dp, qx, qy, qz;

      // Shift the data
      dp = dg.Deriv(pot, 0); qx = dg.Interp3d(dp, pp); _mydestroy(dp);
      dp = dg.Deriv(pot, 1); qy = dg.Interp3d(dp, pp); _mydestroy(dp);
      dp = dg.Deriv(pot, 2); qz = dg.Interp3d(dp, pp); _mydestroy(dp);
      // Print some statistics
      double sum[3];
      VecSum(qx,&sum[0]); VecSum(qy, &sum[1]); VecSum(qz, &sum[2]);
      for (int ii=0; ii < 3; ++ii) sum[ii] /= pp.npart;
      PetscPrintf(PETSC_COMM_WORLD, "Mean x,y,z displacements on particles is : %10.4f,%10.4f,%10.4f\n",sum[0],sum[1],sum[2]);
      VecNorm(qx,NORM_2,&sum[0]); VecNorm(qy, NORM_2,&sum[1]); VecNorm(qz, NORM_2,&sum[2]);
      for (int ii=0; ii < 3; ++ii) sum[ii] /= sqrt(pp.npart);
      PetscPrintf(PETSC_COMM_WORLD, "RMS x,y,z displacements on particles is : %10.4f,%10.4f,%10.4f\n",sum[0],sum[1],sum[2]);
      VecAXPY(pp.px, -1.0, qx);
      VecAXPY(pp.py, -1.0, qy);
      VecAXPY(pp.pz, -1.0, qz);
      // Print the results.
      pp.AsciiWriteSerial(outpfn,boxsize);
      // Clean up
      _mydestroy(qx); _mydestroy(qy); _mydestroy(qz);

      // Now do the same for the randoms
      dp = dg.Deriv(pot, 0); qx = dg.Interp3d(dp, pr); _mydestroy(dp);
      dp = dg.Deriv(pot, 1); qy = dg.Interp3d(dp, pr); _mydestroy(dp);
      dp = dg.Deriv(pot, 2); qz = dg.Interp3d(dp, pr); _mydestroy(dp);
      VecSum(qx,&sum[0]); VecSum(qy, &sum[1]); VecSum(qz, &sum[2]);
      for (int ii=0; ii < 3; ++ii) sum[ii] /= pr.npart;
      PetscPrintf(PETSC_COMM_WORLD, "Mean x,y,z displacements on randoms is : %10.4f,%10.4f,%10.4f\n",sum[0],sum[1],sum[2]);
      VecNorm(qx,NORM_2,&sum[0]); VecNorm(qy, NORM_2,&sum[1]); VecNorm(qz, NORM_2,&sum[2]);
      for (int ii=0; ii < 3; ++ii) sum[ii] /= sqrt(pr.npart);
      PetscPrintf(PETSC_COMM_WORLD, "RMS x,y,z displacements on randoms is : %10.4f,%10.4f,%10.4f\n",sum[0],sum[1],sum[2]);
      VecAXPY(pr.px, -1.0, qx);
      VecAXPY(pr.py, -1.0, qy);
      VecAXPY(pr.pz, -1.0, qz);
      PetscPrintf(PETSC_COMM_WORLD,"Displacements calculated....\n");
      // Print the results.
      pr.AsciiWriteSerial(outrfn,boxsize);
      // Clean up
      _mydestroy(qx); _mydestroy(qy); _mydestroy(qz);
      
      // Clean up everything else
      _mydestroy(pot); 
      _mydestroy(grid);

    }


    // Only call this when everything is out of scope
    ierr=PetscFinalize(); CHKERRQ(ierr);

}
