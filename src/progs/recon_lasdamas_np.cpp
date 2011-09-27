#include <iostream>
#include <cmath>
#include <sstream>
#include <fstream>
#include <iomanip>

#include "Recon.h"

static char help[] = "recon_shell_real -configfn <configuration file>\n";

using namespace std;

int main(int argc, char *args[]) {
    
    PetscErrorCode ierr;
    ierr=PetscInitialize(&argc,&args,(char *) 0, help); CHKERRQ(ierr);
    { 
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
    LasDamasParams pars(string(configfn), "LDreal"); 

    /****************************
     * Define seeds
     ****************************/
    int const_seed=2147; int randompart_seed=8271;

    /******************************************
     * Define the mask and delta
     ******************************************/
    Mask3D dummymask;
    Particle pm;
    pm.AsciiReadSerial(pars.mask.fn.c_str(), pars.LDBoxEmbed);
    Delta del1(pars.Ngrid, pars.Lbox, 1, pars.mask.thresh, dummymask, pm, pars.recon.smooth);
   
    /********************************************
     * Read in pk prior 
     *******************************************/
    vector<double> kvec, pkvec;
    int nk;
    // Rank 0 job reads in pk function and broadcasts to everyone
    if (rank == 0) {
      double k1, pk1;
      ifstream fp(pars.pkprior.fn.c_str());
      do {
        fp >> k1 >> pk1;
        if (fp) {
          kvec.push_back(k1); 
          pkvec.push_back(pars.pkprior.bias*pk1 + pars.pkprior.noise);
        }  
      } while (fp);
      fp.close();
      nk = kvec.size();
    }
    MPI_Bcast(&nk, 1, MPI_INT, 0, PETSC_COMM_WORLD);
    if (rank !=0) {kvec.resize(nk, 0.0); pkvec.resize(nk, 0.0);}
    MPI_Bcast(&kvec[0], nk, MPI_DOUBLE, 0, PETSC_COMM_WORLD);
    MPI_Bcast(&pkvec[0], nk, MPI_DOUBLE, 0, PETSC_COMM_WORLD);
    PetscPrintf(PETSC_COMM_WORLD, "%d lines read in from %s...\n",nk, pars.pkprior.fn.c_str());

    // Loop over files
    list<LasDamasParams::fn>::iterator files;
    for (files = pars.fnlist.begin(); files !=pars.fnlist.end(); ++files) 
    {
      /* ******************************
       * First we get the various options and print out useful information 
       * ********************************/
      ostringstream hdr;
      hdr << "# Input data file is " << files->indata << endl; 
      hdr << "# Output data file is " << files->outdata << endl;
      hdr << "# Input random file is " << files->inrand << endl; 
      hdr << "# Output rand file is " << files->outrand << endl;
      hdr << "# Ngrid=" << setw(5) << pars.Ngrid << endl;
      hdr << "# boxsize=" << setw(8) << fixed << setprecision(2) << pars.Lbox << endl;
      hdr << "# Las Damas embed boxsize=" << setw(8) << fixed << setprecision(2) << pars.LDBoxEmbed << endl;
      hdr << "# bias=" << setw(8) << setprecision(2) << pars.recon.bias << endl;
      hdr << "# smooth=" << setw(8) << setprecision(2) << pars.recon.smooth << endl;
      hdr << "# Mask threshold =" << pars.mask.thresh << endl;
      hdr << "# Mask random file =" << pars.mask.fn << endl;
      PetscPrintf(PETSC_COMM_WORLD, (hdr.str()).c_str()); 
      
      
      /****************************************
       * Read in the particle data here and slab decompose
       ****************************************/
      Particle pp, pr;
      DensityGrid dg(pars.Ngrid, pars.Lbox);
      pp.AsciiReadSerial(files->indata.c_str(), pars.LDBoxEmbed);
      PetscPrintf(PETSC_COMM_WORLD,"Read in %i particles.....\n",pp.npart);
      pr.AsciiReadSerial(files->inrand.c_str(), pars.LDBoxEmbed);
      PetscPrintf(PETSC_COMM_WORLD,"Read in %i randoms.....\n",pr.npart);
      pp.SlabDecompose(dg);
      pr.SlabDecompose(dg);


      /*************************************************
       * Generate the density field and constrained realization
       *************************************************/
      Vec grid=PETSC_NULL;
      del1.BuildDensityGrid(pp, grid);
      PetscPrintf(PETSC_COMM_WORLD, "Density grid computed.....\n");
      del1.HoffmanRibak(grid, kvec, pkvec, const_seed); const_seed+=1;
      PetscPrintf(PETSC_COMM_WORLD, "Constrained realization computed.....\n");

      /************************************************
       * Now we solve for the potential 
       ************************************************/
      // Allocate potential solver
      Vec pot;
      PotentialSolve psolve(pars.Ngrid, pars.Lbox);
      pot = dg.Allocate();
      if (psolve.Solve(REALPBC, grid, pot, PETSC_NULL, pars.recon.bias)) {
        // If the potential calculation converged
        PetscPrintf(PETSC_COMM_WORLD,"Potential calculated....\n");
        
        /************************************************
         * Now we shift data and randoms
         ************************************************/
        // Generate random particles
        Vec dp, qx, qy, qz;
        
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
        
        // Write out files
        pp.AsciiWriteSerial(files->outdata.c_str(), pars.LDBoxEmbed);
        pr.AsciiWriteSerial(files->outrand.c_str(), pars.LDBoxEmbed);
      }
      // Cleanup
      _mydestroy(grid); _mydestroy(pot);

    }

    }
    // Only call this when everything is out of scope
    ierr=PetscFinalize(); CHKERRQ(ierr);

}
