#include <iostream>
#include <cmath>
#include <sstream>
#include <iomanip>

#include "Recon.h"

static char help[] = "recon_PBC_real -configfn <configuration file>\n";

using namespace std;

int main(int argc, char *args[]) {
    
    PetscErrorCode ierr;
    ierr=PetscInitialize(&argc,&args,(char *) 0, help); CHKERRQ(ierr);
  
    //  Get MPI rank
    int rank; MPI_Comm_rank(PETSC_COMM_WORLD, &rank);

    // Read in configuration file
    char configfn[200]; 
    PetscTruth flg; 

    // See if we need help
    PetscOptionsHasName("-help", &flg);
    if (flg) exit(0);
    PetscOptionsGetString("-configfn", configfn, 200, &flg);
    if (!flg) RAISE_ERR(99,"Specify configuration file");
    // Read in parameters
    PBCParams pars(string(configfn), "PBCreal"); 

    // Define the potential solver 
    PotentialSolve psolve(pars.Ngrid, pars.Lbox, pars.recon.maxit);
    psolve.SetupOperator(REALPBC);


    // Loop over files
    list<PBCParams::fn>::iterator files;
    for (files = pars.fnlist.begin(); files !=pars.fnlist.end(); ++files) 
    {
      /* ******************************
       * First we get the various options and print out useful information 
       * ********************************/
      ostringstream hdr;
      hdr << "# Input file is " << files->in << endl; 
      hdr << "# Output file is " << files->out << endl;
      hdr << "# Ngrid=" << setw(5) << pars.Ngrid << endl;
      hdr << "# boxsize=" << setw(8) << fixed << setprecision(2) << pars.Lbox << endl;
      hdr << "# bias=" << setw(8) << setprecision(2) << pars.recon.bias << endl;
      hdr << "# smooth=" << setw(8) << setprecision(2) << pars.recon.smooth << endl;
      hdr << "# " << setw(4) << pars.xi.Nbins << " Xi bins from " << setw(8) << setprecision(2) << pars.xi.rmin
        << " with spacing of " << pars.xi.dr << endl;
      hdr << "# " << "Correlation function smoothed with a smoothing scale of" << setw(8) << setprecision(2) 
        << pars.xi.smooth << endl;
      PetscPrintf(PETSC_COMM_WORLD, (hdr.str()).c_str()); 
      
      
      /****************************************
       * Read in the particle data here and slab decompose
       ****************************************/
      Particle pp;
      DensityGrid dg(pars.Ngrid, pars.Lbox);
      pp.TPMReadSerial(files->in.c_str(), pars.Lbox);
      PetscPrintf(PETSC_COMM_WORLD,"Read in %i particles.....\n",pp.npart);
      pp.SlabDecompose(dg);
      // Test slab decomp
      bool good = dg.TestSlabDecompose(pp);
      if (good) 
         {PetscSynchronizedPrintf(PETSC_COMM_WORLD,"Slab decomposition succeeded on process %i\n",rank);}
      else 
         {PetscSynchronizedPrintf(PETSC_COMM_WORLD,"Slab decomposition FAILED on process %i\n",rank);}
      PetscSynchronizedFlush(PETSC_COMM_WORLD);
      if (!good) RAISE_ERR(99, "Slab decomposition failed");


      /*************************************************
       * Now we start working on the grid
       * ***********************************************/
      Vec grid, gridr;
      PkStruct xi1(pars.xi.rmin, pars.xi.dr, pars.xi.Nbins);
      grid=dg.Allocate();
      //CIC
      dg.CIC(grid, pp);
      VecDuplicate(grid, &gridr); VecCopy(grid, gridr); //FFTs are destructive!
      dg.XiFFT(gridr, pars.xi.smooth, xi1);
      // Smooth
      dg.GaussSmooth(grid, pars.recon.smooth);
      PetscPrintf(PETSC_COMM_WORLD,"Initial correlation function computed....\n");



      /************************************************
       * Now we solve for the potential 
       ************************************************/
      // Allocate potential solver
      Vec pot;
      pot = dg.Allocate();
      if (psolve.Solve(grid, pot, pars.recon.bias)) {
        // If the potential calculation converged
        PetscPrintf(PETSC_COMM_WORLD,"Potential calculated....\n");
        
        /************************************************
         * Now we shift data and randoms
         ************************************************/
        // Generate random particles
        Vec dp, qx, qy, qz;
        Particle pr;
        pr.RandomInit(pp.npart*pars.recon.nrandomfac, pars.Lbox, 1931);
        pr.SlabDecompose(dg);
        
        // Compute derivatives at data positions and shift
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
        // Cleanup
        _mydestroy(qx); _mydestroy(qy); _mydestroy(qz);
        
        // Do the same for the randoms
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
        // Clean up
        _mydestroy(qx); _mydestroy(qy); _mydestroy(qz);
        
        
        // Shifted data and random grid
        pp.SlabDecompose(dg); pr.SlabDecompose(dg);
        dg.CIC(grid, pp); dg.CIC(gridr, pr);
        VecAXPY(grid, -1.0, gridr);
        // Correlation fn
        PkStruct xi2(pars.xi.rmin, pars.xi.dr, pars.xi.Nbins);
        dg.XiFFT(grid, pars.xi.smooth, xi2);
        
        
        // Outputs
        FILE *fp;
        double _rvec, _xi1, _xi2, _n1;
        PetscFOpen(PETSC_COMM_WORLD,files->out.c_str(),"w", &fp);
        PetscFPrintf(PETSC_COMM_WORLD, fp, (hdr.str()).c_str());
        for (int ii = xi1.lo; ii < xi1.hi; ++ii) {
          _xi1 = xi1(ii, _rvec, _n1);
          _xi2 = xi2(ii);
          if (_n1>0) PetscSynchronizedFPrintf(PETSC_COMM_WORLD, fp, "%6i %9.3f %15.8e  %15.8e\n",ii,_rvec,_xi1,_xi2);
        }
        PetscSynchronizedFlush(PETSC_COMM_WORLD);
        PetscFClose(PETSC_COMM_WORLD,fp);
      }
      // Cleanup
      _mydestroy(grid);_mydestroy(gridr); _mydestroy(pot);

    }


    // Only call this when everything is out of scope
    ierr=PetscFinalize(); CHKERRQ(ierr);

}
