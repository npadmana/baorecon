#include "Recon.h"


Delta::Delta(int _N, double _L, int _nover, double thresh,  Mask3D& _mask)
// RS 2010/07/06:  Comments!
{
  // Cache values
  N=_N; L=_L; nover=_nover; maskptr=&_mask; rmasksmooth = 0.0;

  // Set up the density
  // RS:  dg2 is some kind of local variable; dg1 is a class member.
  // W and _W are contents of a grid of the same size and dimensions
  // as dg1 and dg2.
  DensityGrid dg2(N, L);
  dg1.Init(N, L);
  _W = dg1.Allocate();
  W = dg1.Allocate(); VecSet(W, 0.0);

  // Now do the randoms
  // RS:  "Doing" the randoms in this context means "oversampling" each grid
  // cell into subcells, where the main cell is nover times as large as each
  // subcell and hence there are nover^3 subcells to a cell.
  nrand=0.0;
  double nexp, nthresh;
  nexp = pow((double) nover, 3);
  // RS:  thresh is the *fraction* of the main cell's volume to be included
  // inside the mask for it to be counted as not masked, so nthresh is the
  // corresponding number of subcells.
  nthresh = nexp*thresh;
  {
    double tmp = 1./nover, dx, dy, dz;
    VecSet(_W, 0.0); // Manually set to zero
    Vec W1; VecDuplicate(_W, &W1);
    // RS:  Loop over all subcell *offsets*.  The loop over actual grid cells
    // is implicit in the calls to methods like GridInitialize below.
    for (int ix = 0; ix < nover; ++ix) {
      PetscPrintf(PETSC_COMM_WORLD,"Starting random loop %i of %i....\n",ix,nover-1);
      dx = ix*tmp + tmp/2.0;
      for (int iy = 0; iy < nover; ++iy) {
        dy = iy*tmp + tmp/2.0;
        for (int iz = 0; iz < nover; ++iz) {
          dz = iz*tmp + tmp/2.0;
          Particle rr; // Definition here is important, since it ensures destruction, before recreating
          // RS:  Generate a bunch of particles uniformly spaced, and offset
          // from the low-(x,y,z) corner of each grid cell by (dx,dy,dz).
          rr.GridInitialize(N, dx, dy, dz, L);
          // RS:  Remove from the list any particles which were not allowed
          // by the mask.  Add to the total.
          rr.TrimMask(*maskptr); nrand += rr.npart;
          // rr.SlabDecompose(dg2); We don't need to slab decompose because nproc divides N by construction
          // RS:  CIC assign remaining particles to temporary grid vector W1.
          dg1.CIC(W1, rr, false); // No overdensity
          // RS:  Now add to _W the CIC contents of W1.
          VecAXPY(_W, 1.0, W1);
        }
      }
    }
  }
  // RS:  At this point, nrand equals the total number of oversampled subcells
  // found to be allowed by the masked region.  _W contains the number of
  // particles allowed by an oversampled mask at each grid location.
  PetscPrintf(PETSC_COMM_WORLD, "%f grid particles accumulated....\n", nrand);

  // Now set the mask
  // RS:  For each cell, if the contents of _W exceed nthresh, set W = 1.
  // Use dg1 and dg2 as wrappers to index the copies of _W and W.
  int lo, hi;
  dg1.slab(lo, hi);
  dg1.Pull(_W); dg2.Pull(W);
  for (int ix=lo; ix < hi; ++ix)
    for (int iy=0; iy < N; ++iy) 
      for (int iz=0; iz < N; ++iz) 
        if (dg1(ix, iy,iz) > nthresh) {
            dg2(ix, iy,iz) = 1.0;
        } else {
          dg1(ix, iy, iz) = 0.0; // Good to explicitly set this to zero as well
        }
  dg1.Push(_W, INSERT_VALUES, false);
  dg2.Push(W, INSERT_VALUES, false);
  double tmp;
  VecSum(W, &tmp);
  PetscPrintf(PETSC_COMM_WORLD,"Nonzero grid points in W : %f\n",tmp);
  VecSum(_W, &nrand);
  PetscPrintf(PETSC_COMM_WORLD,"Sum of background density vector : %f\n",nrand);

}

Delta::Delta(int _N, double _L, int _nover, double thresh,  Mask3D& _mask,
   Particle& pr, double smooth)
// RS 2010/07/06:  Version of Delta constructor that takes as input a set
// of random particles to define where the mask is or is not.  This is in
// contrast to the regular grid of "randoms" set up to compute the average
// density in the normal Delta constructor above.
{
  // We're not going to use the Mask3D& for anything, but I make the user
  // pass one in anyway, because Delta can't manage the memory for it without
  // creating the risk either of a memory leak or a seg fault.
  N=_N; L=_L; nover=_nover; maskptr=&_mask; rmasksmooth=smooth;

  // W is the mask, _W is the average density of randoms.
  // dg1 and dg2 are DensityGrid wrappers for these vectors.
  DensityGrid dg2(N, L);
  dg1.Init(N, L);
  _W = dg1.Allocate();
  W = dg1.Allocate(); VecSet(W, 0.0);

  pr.SlabDecompose(dg1);
  dg1.CIC(_W, pr, false);           // Assign randoms to _W ("no overdensity")
  dg1.GaussSmooth(_W, smooth);      // Smooth by smoothing scale
  PetscPrintf(PETSC_COMM_WORLD, "%d grid particles accumulated....\n", pr.npart);

  // RS:  For each cell, if the contents of _W exceed thresh, set W = 1.
  // Use dg1 and dg2 as wrappers to index the copies of _W and W.
  int lo, hi;
  dg1.slab(lo, hi);
  dg1.Pull(_W); dg2.Pull(W);
  for (int ix=lo; ix < hi; ++ix)
    for (int iy=0; iy < N; ++iy) 
      for (int iz=0; iz < N; ++iz) 
        if (dg1(ix,iy,iz) > thresh) {
          dg2(ix,iy,iz) = 1.0;
        } else {
          dg1(ix, iy, iz) = 0.0; // Good to set this explicitly to zero as well
        }
  dg1.Push(_W, INSERT_VALUES, false);
  dg2.Push(W, INSERT_VALUES, false);
  double tmp;
  VecSum(W, &tmp);
  PetscPrintf(PETSC_COMM_WORLD,"Nonzero grid points in W : %f\n",tmp);
  VecSum(_W, &nrand);
  PetscPrintf(PETSC_COMM_WORLD,"Sum of background density vector : %f\n",nrand);
}

Delta::~Delta() {
  _mydestroy(_W);
  _mydestroy(W);
}


void Delta::ZeroMean(Vec &delta) {

  // Compute mean
  double sum, npts, norm;
  VecPointwiseMult(delta, W, delta);
  VecSum(delta, &sum);
  VecSum(W, &npts);
  norm = sum/npts;

  // Subtract mean
  VecAXPY(delta, -norm, W);
}


void Delta::BuildDensityGrid(Particle& pp, Vec& delta)
// RS 2010/07/06:  Commented!
{
  _mydestroy(delta); // Clean up delta 
  DensityGrid dg2(N, L);
  DensityGrid dg3(N, L);
  delta = dg1.Allocate();

  bool rmask = (rmasksmooth > 1e-5);   // RS:  boolean shorthand

  // First the particles
  // RS 2010/07/06:  CIC assign the list of particles to delta.
  // If we're using a mask made from smoothed randoms, don't bother trimming,
  // because the mask will just be a "dummy mask".
  if (!rmask) pp.TrimMask(*maskptr);
  pp.SlabDecompose(dg1);
  dg1.CIC(delta,pp, false);
  PetscPrintf(PETSC_COMM_WORLD, "Particles assigned to the grid.....\n");

  // RS 2010/07/06:  If rmasksmooth > 0, smooth CIC-assigned particles before
  // computing overdensity.  Otherwise we'll get unpleasant edge effects.
  if (rmask) dg1.GaussSmooth(delta, rmasksmooth);

  // Normalization factor
  // RS:  This equals the *average* number (per allowed cell) of "randoms"
  // compared to the number of particles from which we're building delta.
  VecPointwiseMult(delta, W);
  double fac;
  VecSum(delta, &fac);
  fac = nrand/fac;

  // Now construct delta
  int lo, hi;
  dg1.slab(lo, hi);
  // RS:  Set up the wrappers to allow us to index delta by (x,y,z).
  dg1.Pull(delta); dg2.Pull(W); dg3.Pull(_W);
  for (int ix=lo; ix < hi; ++ix)
    for (int iy=0; iy < N; ++iy) 
      for (int iz=0; iz < N; ++iz) 
        if (dg2(ix, iy,iz) > 1.e-5) {
          // RS:  In each cell location, if W > 0, divide by _W/fac.
          // Since _W is the average density of regular-grid "randoms" in
          // each cell (accounting for edge effects), _W/fac corresponds to
          // the average density of pp particles.  So W/(_W/fac) = W/_W*fac,
          // i.e., the density contrast, which is what we want.
          dg1(ix, iy, iz) = (dg1(ix, iy, iz)/dg3(ix, iy, iz))*fac - 1.0;
        } else {
          // RS:  If W = 0 (to within epsilon), set delta to zero.
          dg1(ix, iy, iz) = 0.0;
        }
  // RS:  Deallocate the memory we just set up...  and we'll be done.
  dg1.Push(delta, INSERT_VALUES, false);
  dg2.Push(W, INSERT_VALUES, false);
  dg3.Push(_W, INSERT_VALUES, false);

  VecSum(delta, &fac);
  PetscPrintf(PETSC_COMM_WORLD, "Sum (delta).....%e\n", fac);

}

void Delta::BuildDensityGridSmooth(double smooth, Particle& pp, Vec& delta) 
{
  BuildDensityGrid(pp, delta);
  if (rmasksmooth > 1.e-5) PetscPrintf(PETSC_COMM_WORLD, "WARNING!!! POSSIBLE MULTIPLE SMOOTHING DETECTED\n");
  dg1.GaussSmooth(delta, smooth);
}


void BuildDensityGrid(int N, double L, Mask3D& mask, Particle& pp, int nover, double thresh, Vec& delta, Vec& W, double smooth) {
/* Parameters :
 *   INPUT ---
 *     N     :  grid size
 *     L     :  box size
 *     mask  :  A 3D positive mask (assumed to be 1 or 0)
 *     nover :  Oversampling rate for defining the mean density (we use the GridInitialize routine)
 *              This determines the displacement. 
 *              For a periodic box, nover means there will be nover**3 particles per grid.
 *     thresh:  Fractional threshold below which to remove grid point from survey
 *     smooth:  This is an optional parameter -- if set to > 0, it will smooth the density and mask
 *              fields before computing the overdensity.
 *   OUTPUT ---
 *     delta :  What do think this is???
 *     W     :  Window function, 1 in survey, 0 elsewhere. 
 *   Note that the output vectors are not cleared, so you had better make sure you don't leak memory.
 */

  // Allocate two density grids
  DensityGrid dg1(N,L), dg2(N,L);
  delta = dg1.Allocate();
  W = dg2.Allocate(); 


  // First the particles
  pp.TrimMask(mask);
  pp.SlabDecompose(dg1);
  dg2.CIC(delta,pp, false);
  PetscPrintf(PETSC_COMM_WORLD, "Particles assigned to the grid.....\n");


  // Now do the randoms
  double nrand=0.0, nexp, nthresh;
  nexp = pow((double) nover, 3);
  nthresh = nexp*thresh;
  {
    double tmp = 1./nover, dx, dy, dz;
    VecSet(W, 0.0); // Manually set to zero
    Vec W1; VecDuplicate(W, &W1);
    for (int ix = 0; ix < nover; ++ix) {
      PetscPrintf(PETSC_COMM_WORLD,"Starting random loop %i of %i....\n",ix,nover-1);
      dx = ix*tmp + tmp/2.0;
      for (int iy = 0; iy < nover; ++iy) {
        dy = iy*tmp + tmp/2.0;
        for (int iz = 0; iz < nover; ++iz) {
          dz = iz*tmp + tmp/2.0;
          Particle rr; // Definition here is important, since it ensures destruction, before recreating
          rr.GridInitialize(N, dx, dy, dz, L);
          rr.TrimMask(mask); nrand += rr.npart;
          // rr.SlabDecompose(dg2); We don't need to slab decompose because nproc divides N by construction
          dg2.CIC(W1, rr, false); // No overdensity
          VecAXPY(W, 1.0, W1);
        }
      }
    }
  }
  PetscPrintf(PETSC_COMM_WORLD, "%f grid particles accumulated....\n", nrand);
  PetscPrintf(PETSC_COMM_WORLD, "Expected number of particles=%f, threshold=%f\n", nexp, nthresh);

  // Smooth if desired
  if (smooth > 1.e-5) {
    dg1.GaussSmooth(delta, smooth);
    dg2.GaussSmooth(W, smooth);
  }


  // Now build up delta
  int lo, hi;
  nrand /= pp.npart;
  dg1.slab(lo, hi);
  dg1.Pull(delta); dg2.Pull(W);
  for (int ix=lo; ix < hi; ++ix)
    for (int iy=0; iy < N; ++iy) 
      for (int iz=0; iz < N; ++iz) {
        if (dg2(ix, iy,iz) > nthresh) {
          dg1(ix, iy,iz) =  (dg1(ix,iy,iz)/dg2(ix, iy, iz)) * nrand - 1.0;
          dg2(ix, iy,iz) = 1.0;
        } else {
          dg1(ix, iy, iz) = 0.0;
          dg2(ix, iy, iz) = 0.0;
        }
  }
  dg1.Push(delta, INSERT_VALUES, false);
  dg2.Push(W, INSERT_VALUES, false);
  VecSum(W, &nrand);
  PetscPrintf(PETSC_COMM_WORLD,"Nonzero grid points in W : %f\n",nrand);

}



void Delta::HoffmanRibak(Vec& c, vector<double>& kprior, vector<double>& pkprior, 
                         int seed, double tol, int dorandom) {

  // Set up preconditioner
  vector<double> precondpk(pkprior);
  for (int ii=0; ii<pkprior.size(); ++ii) precondpk[ii] = 1./pkprior[ii];

  // Allocate space for a random Gaussian field
  Vec f; f = dg1.Allocate();

  // Define the RHS vector
  // Start by generating a Gaussian density field
  // We only do this if we are generating a random Gaussian field
  if (dorandom > 0) {
    dg1.FakeGauss(seed, f, kprior, pkprior);
    VecAXPY(c, -1.0, f);
  }
  VecPointwiseMult(c, W, c);

  // Now set up for CG solution
  Vec rk, pk, Axk, zk; 
  double bnorm, rnorm, rnorm1, rz, rz1;
  VecDuplicate(c, &pk); VecDuplicate(c, &rk); VecDuplicate(c, &Axk); VecDuplicate(c, &zk);

  // Compute initial residual and norms
  VecNorm(c, NORM_2, &bnorm);
  VecCopy(c, rk); rnorm = bnorm; 
  VecCopy(rk, zk); dg1.kConvolve(zk, kprior, precondpk); VecPointwiseMult(zk, W, zk);
  VecCopy(zk, pk);
  VecSet(c, 0.0);

  int iter = 0;
  double alphak, betak, tmp1, relnorm;
  relnorm = sqrt(rnorm/bnorm);
  while(relnorm > tol) {
    // CG iteration
    PetscPrintf(PETSC_COMM_WORLD, "Starting iteration %4d .... rnorm = %10.5e, frac = %10.5e\n",iter,rnorm, relnorm);
    
    // Compute alpha_k
    VecCopy(pk, Axk); dg1.kConvolve(Axk, kprior, pkprior); VecPointwiseMult(Axk, W, Axk);
    rz = VecDot(rk, zk);
    tmp1 = VecDot(pk, Axk);
    alphak = rz/tmp1;
    
    // Update x and r
    rnorm1 = rnorm; rz1 = rz;
    VecAXPY(c, alphak, pk);
    VecAXPY(rk, -alphak, Axk);
    VecNorm(rk, NORM_2, &rnorm);
    VecCopy(rk, zk); dg1.kConvolve(zk, kprior, precondpk); VecPointwiseMult(zk, W, zk);
    rz = VecDot(rk, zk);

    // Compute beta_k
    betak = rz/rz1;
    VecAYPX(pk, betak, zk);

    // Update iteration count
    iter+=1; 
    relnorm = sqrt(rnorm/bnorm);
  }
  PetscPrintf(PETSC_COMM_WORLD, "Converged after %4d iterations .... rnorm = %10.5e, frac = %10.5e\n",iter,rnorm, relnorm);

  // Finish up generating the realization 
  dg1.kConvolve(c, kprior, pkprior);
  // If we are generating a random field, instead of the simple WF
  if (dorandom > 0) 
    VecAXPY(c, 1.0, f);  

  // Clean up the CG vectors
  VecDestroy(rk); VecDestroy(f); VecDestroy(pk); VecDestroy(Axk); VecDestroy(zk);

}
