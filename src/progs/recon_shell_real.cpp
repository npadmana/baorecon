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
    PetscBool flg; 

    // See if we need help
    PetscOptionsHasName(NULL, "-help", &flg);
    if (flg) exit(0);
    PetscOptionsGetString(NULL, "-configfn", configfn, 200, &flg);
    if (!flg) RAISE_ERR(99,"Specify configuration file");
    // Read in parameters
    ShellParams pars(string(configfn), "shellreal"); 

    /****************************
     * Define seeds
     ****************************/
    int const_seed=2147; int randompart_seed=8271;

    /******************************************
     * Define the mask and delta
     ******************************************/
    Shell maskss(pars.shell.xcen, pars.shell.ycen, pars.shell.zcen, pars.shell.rmin, pars.shell.rmax);
    Delta del1(pars.Ngrid, pars.Lbox, pars.shell.nover, pars.shell.thresh, maskss);
   
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

    // Define the potential solver 
    PotentialSolve psolve(pars.Ngrid, pars.Lbox, pars.recon.maxit);
    psolve.SetupOperator(REALPBC);

    // Loop over files
    list<ShellParams::fn>::iterator files;
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
      hdr << "# " << "Survey shell centered at the origin, rmin=" << setprecision(2) << pars.shell.rmin << ", rmax=" << pars.shell.rmax << endl;
      hdr << "# " << "Shell center at " << setprecision(3) << pars.shell.xcen << ", " << pars.shell.ycen << ", " << pars.shell.zcen << endl;
      hdr << "# Mask threshold =" << pars.shell.thresh << endl;
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
      Vec grid;
      PkStruct xi1(pars.xi.rmin, pars.xi.dr, pars.xi.Nbins);
      grid=dg.Allocate();
      //CIC
      dg.CIC(grid, pp);
      dg.XiFFT(grid, pars.xi.smooth, xi1);
      PetscPrintf(PETSC_COMM_WORLD,"Initial correlation function computed....\n");
      VecDestroy(&grid);

      /*************************************************
       * Generate the density field and constrained realization
       *************************************************/
      del1.BuildDensityGridSmooth(pars.recon.smooth, pp, grid);
      PetscPrintf(PETSC_COMM_WORLD, "Density grid computed.....\n");
      del1.HoffmanRibak(grid, kvec, pkvec, const_seed); const_seed+=1;
      PetscPrintf(PETSC_COMM_WORLD, "Constrained realization computed.....\n");

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
        pr.RandomInit(pp.npart*pars.recon.nrandomfac, pars.Lbox, randompart_seed); randompart_seed +=1;
        pr.SlabDecompose(dg);
        pr.TrimMask(maskss);
        
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
        Vec gridr=PETSC_NULL;
        del1.BuildDensityGrid(pp, grid);
        del1.BuildDensityGrid(pr, gridr);
        VecAXPY(grid, -1.0, gridr);
        // Correlation fn
        PkStruct xi2(pars.xi.rmin, pars.xi.dr, pars.xi.Nbins);
        dg.XiFFT_W(grid, del1.W, pars.xi.smooth, xi2);
        _mydestroy(gridr);
        
        
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
      _mydestroy(grid); _mydestroy(pot);

    }

    }
    // Only call this when everything is out of scope
    ierr=PetscFinalize(); CHKERRQ(ierr);

}
