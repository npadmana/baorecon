#ifndef RECON_H_
#define RECON_H_ 

#include <vector>
#include <cmath>
#include <iostream>
#include <string>
#include <list>

// Header file for a Parallel particle class
// Uses PETSc for distributed vectors
#include "petscsys.h"
#include "petscvec.h"
#include "petscmat.h"
#include "petscpc.h"
#include "petscksp.h"
#include "petsctime.h"

//GSL includes
#include <gsl/gsl_rng.h>
#include <gsl/gsl_randist.h>
#include <gsl/gsl_interp.h>
#include <gsl/gsl_spline.h>
#include <gsl/gsl_histogram.h>

// MPI FFTW
#include "drfftw_mpi.h"

//XML support
#define TIXML_USE_TICPP
#include "ticpp/ticpp.h"

#define RAISE_ERR(n,s)              {PetscError(__LINE__,__FUNCT__,__FILE__,__SDIR__,n,1,s); MPI_Abort(PETSC_COMM_WORLD, n);}

#ifndef M_PI
#define M_PI 3.141592653589793238462643
#endif

//Basic default parameters
//These can be overridden in the config file (or even worse, through PETSC)
#define DEFAULT_RTOL 1.e-4
#define DEFAULT_ATOL 1.e-2
#define DEFAULT_MAXIT 2000



using namespace std;

/* Miscellaneous routines go here */
double periodic(double x, double L=1.0);
void _mydestroy(Vec &v);
void VecHist(const Vec& x, int nbins, double xmin, double xmax, vector<double>& hh);

// Forward declarations
class Mask3D; 
class Particle ;
class DensityGrid ;


/* A simple class to store all the parameters we need
 * All the parameters are public, so can be accessed by all routines.
 * The parameters are assumed to be in an XML file, an example of which 
 * is checked into the progs directory
 */
class Params {
  public :
    double Lbox;
    int Ngrid;

    // Recon parameters 
    struct reconst {
      double bias, smooth, fval;
      int maxit, nrandomfac; // note nrandomfac is not used in some codes, but we still read it in
      double rtol, atol; // Tolerances for the converge of the "Poisson" solver. 
      vector<double> origin;
    };
    reconst recon;

    //Code to read in parameters
    Params(string configfn);
};



class PBCParams : public Params {
  public :
    //Xi params
    struct Xist {
      double rmin, dr, smooth;
      int Nbins;
    };
    Xist xi;

    struct fn {
      string in, out;
    };
    list<fn> fnlist;

    //Code to read in parameters
    PBCParams(string configfn, string fntype);
};


class ShellParams : public PBCParams {
  public :
    // Power spectrum priors
    // The prior is assumed to be bias*Pk0 + noise
    struct Pkpriorst {
      string fn;
      double bias, noise;
    };
    Pkpriorst pkprior; 
    
    //Shell params
    struct Shellst {
      double xcen, ycen, zcen, rmin, rmax, thresh;
      int nover;
    };
    Shellst shell;

    //Code to read in parameters
    ShellParams(string configfn, string fntype);
};


class LasDamasParams : public Params {
  public :
    // Power spectrum priors
    // The prior is assumed to be bias**2*Pk0 + noise
    struct Pkpriorst {
      string fn;
      double bias, noise;
    };
    Pkpriorst pkprior; 
    
    //Shell params
    struct Maskst {
      double thresh;
      string fn;
    };
    Maskst mask;


    struct fn {
      string indata, outdata;
      string inrand, outrand;
    };
    list<fn> fnlist;

    //Code to read in parameters
    LasDamasParams(string configfn, string fntype);
};


// Simple class for collecting Pk/Xi data
class PkStruct {
  private :
    double *_kvec, *_pkvec, *_nmodes;
  public :
    Vec kvec, pkvec, nmodes; 
    double kmin, dk;
    int Nbins;
    PetscInt lo, hi;

    PkStruct(double _kmin, double _dk, int Nbins);
    ~PkStruct();
    void accum(double k1, double pk1, double fac=1.0);
    void finalize();
    double operator()(PetscInt ii, double& k1, double& N1); 
    double operator()(PetscInt ii); 
};

// This is a dummy class to be inherited from later
// Mask is assumed to be the positive mask
class Mask3D {
  public :
    virtual bool operator()(double x, double y, double z) {return true;};
};

class Shell : public Mask3D {
  private :
    double _x0,_y0,_z0;
    double _rmin, _rmax;
  public :
    Shell(double x0, double y0, double z0, double rmin, double rmax);
    virtual bool operator()(double x, double y, double z);
};

/* A class to store Particles */
class Particle {

  public :
    int npart; // Note 32-bit integer here for particle number
    Vec px,py,pz; // PETSc vectors to store the particle positions
    Vec pw; // PETSc vector to store particle weights

    // Constructors and destructors 
    Particle(const Particle& p);
    Particle() {npart=0; px=py=pz=pw=PETSC_NULL;};
    ~Particle();

    // IO routines
    // Reader for TPM files
    //    zspace -- shifts to redshift space
    //    pplel -- plane parallel redshift space distortions. 
    //             If true, we shift along the z-axis.
    //             if not, we shift in a radial direction.     
    //             The origin is assumed to be at 0,0,0
    //             Note that since the box is periodic, this will result in artefacts, unless you
    //             trim the box. Which we leave up to you.
    int TPMReadSerial(const char *infilename, double L=1.0, bool zspace=false, bool pplel=true);
    // ASCII reading, writing options
    int AsciiReadSerial(const char *infilename);
    int AsciiWriteSerial(const char *outfilename);
    // ASCII reading, writing options for weights
    int AsciiReadWeightedSerial(const char *infilename);
    int AsciiWriteWeightedSerial(const char *outfilename);
    // Initialize particles onto a grid, with a possible displacement
    //    dx, dy, dz --- in cell units
    void GridInitialize(int N, double dx=0.0, double dy=0.0, double dz=0.0, double L=1.0);
    void RandomInit(int _npart, double L, int seed);

    // Reorganize
    void SlabDecompose(const DensityGrid &g);

    // Mask handling routines
    // Count the number of objects that fall into the positive mask
    int CountMask(Mask3D& mask);
    /* TrimMask(Mask3D& mask) 
     *    Trim to those particles for whom mask() is true.
     *    Note that this can result in a very unbalanced array, if some processors have 
     *    no particles. We don't attempt a rebalancing, since very often, this operation
     *    will be followed by a SlabDecompose step.
     *
     *    This routine is *not* optimized for speed or memory, but who cares!!
     */
    void TrimMask(Mask3D& mask);


    /* RadialDisplace(Vec qx, Vec qy, Vec qz, vector<double> &origin, double factor)
     *   Shift the particles in the radial direction (defined by origin) 
     *   The exact shift applied is 
     *       q.p/|r|
     */
    void RadialDisplace(Vec qx, Vec qy, Vec qz, vector<double> &origin, double factor);
};


/* A light weight wrapper for a PETSc vector to treat it as a density grid */
class DensityGrid {

  private :
    // FFTW private data
    rfftwnd_mpi_plan plan, iplan; // FFTW plans
    int local_nx, local_x_start, local_ny_after_transpose, local_y_start_after_transpose, total_local_size;
    int nghost;
    PetscInt lstart, lend;

    // Status bools 
    bool _config; // Are we in configuration space

    // Local data
    Vec _grid;
    double *_ldata; 

    // Private functions
    void _cleanFFTWplan(rfftwnd_mpi_plan &p);

  public :
    PetscInt Ng, Ng2; 
    double L; // Size of the grid
    

    // Constructors
    DensityGrid() {plan=NULL; iplan=NULL;};
    DensityGrid(int N, double _L);
    ~DensityGrid();
    void Init(int N, double _L);
    // Properly allocate a vector compatible with this class
    Vec Allocate(); 


    // Update operations -- scatter data into the local array (Pull) and globally (Push)
    // Note : ghost values are updated with INSERT_VALUES for the Pull step and 
    // ADD_VALUES for the Push step
    // IMPORTANT ::: ANY ACCESS OF THE LOCAL DATA *must* be preceded by a Pull, to 
    // ensure that the ghost values are correctly updated
    // The Pull routine also sets the local vector, so that the () operator can be used.
    // Push deallocates this.
    void Push(Vec v, InsertMode iora, bool update);
    void Pull(Vec v, bool config=true);

    // Misc functions 
    /* slab : 
     *   Returns the slab decomposition. config sets whether we are in configuration or Fourier space.
     */
    void slab(int &lo, int &hi, bool config=true);
    void size(PetscInt &local, PetscInt  &global) { local=total_local_size; global=Ng*Ng*Ng2;};
    /* ScaleShift :
     *    v = v*a + b -- in place
     *    USE ONLY IN CONFIGURATION SPACE. This skips the extra padding.
     */
    void ScaleShift(Vec v, double a, double b);
    void PrintCells (Vec v, double smooth);     // RS:  prints contents of grid
    void ZeroPad(Vec v, double val=0.0);
    double kval(int ix, int iy, int iz);
    double rval(int ix, int iy, int iz); // rval is the distance from the 0,0,0. We wrap around.
    double CICKern(int ix, int iy, int iz); 

    /* (ix, iy, iz, ic=0) 
     *      Get/Set the ix, iy, iz element of the array -- global ordering
     *      This will raise an exception if you try to set a non-local value (except see below).
     *      The ic operator is useful for setting the complex terms after an FFT 
     *      (use ic=1). 
     *      
     *      In real space :
     *        Note that iy and iz must always be between 0, Ng-1. However, if you want to access
     *        the ghosted region, you should use ix +/- 1. 
     *        Eg. if you are on processor 0, and want to access the processor N-1 ghost, use ix=-1
     *        DO NOT USE ix=Ng-1. 
     *        Similarly for processor N-1, use ix=Ng, not ix=0. 
     *
     *      You *MUST* Pull() before starting, and Push() after completion, otherwise, there isn't a vector
     *      loaded.
     */
    double& operator()(int ix, int iy, int iz, int ic=0);
    double& operator()(vector<int> ii, int ic=0) {return (*this)(ii[0], ii[1], ii[2], ic);};


    // FFT functions
    // The same function does the forward and reverse transform depending on config.
    // Note that the value of config is changed by the routine
    void FFT(Vec v, bool &config);
    void GaussSmooth(Vec v, double R);
    /* kConvolve(Vec v, std::vector &k, std::vector &fk)
     *     Convolve the vector v with a function f. This routine assumes you will pass in the fourier transform
     *       of f as a spline-able pair of vectors specifying k and fk).
     *
     *       For wavenumbers < min(k), or wavenumbers > max(k), the routine assumes the fourier transform is zero.
     *       One should be careful that this does not induce significant ringing -- the simplest solution is to 
     *       simply make sure the k-range covered is adequate.
     */
    void kConvolve(Vec v, vector<double> &kvec, vector<double> &fkvec); 
    /* XiFFT(v, Rsmooth, Xi) :
     *     Compute the correlation function of v via FFTs. 
     *     The binning is defined by rmin, dr, and Nbin
     *     The correlation function can be optionally smoothed by a Gaussian of width Rsmooth -- this improves
     *       numerical behaviour.
     *     rvec, Xi are PETSc vectors containing the average r and Xi points
     */
    void XiFFT(Vec v, double Rsmooth, PkStruct& Xi);
    void XiFFT_W(Vec v, Vec W1, double Rsmooth, PkStruct& Xi);
    /* PkCIC(v, Pkmean, Pkdec) 
     *   Compute the power spectrum of a CIC assigned field in v.
     *   NOTE : v is destroyed.
     *      v    -- CIC assigned field 
     *      Pkmean, Pkdec -- mean and deconvolved power spectrum.
     */
    void PkCIC(Vec v, PkStruct& pkmean, PkStruct& pkdec);
    /* FakeGauss(int seed, Vec v, vector<double> &kvec, vector<double> &Pkvec)
     *     Fill the vector v with a Gaussian random realization with power spectrum specified by Pkvec.
     *     NOTE :: Overwrites v
     */
    void FakeGauss(int seed, Vec v, vector<double> &kvec, vector<double>& Pkvec);

    
    /* CIC(Vec v, const Particle &pp, bool overdense, bool zero) 
     *   Takes a list of particles and CIC assigns them to a grid
     *   This routine assumes that the particles have been slab-decomposed onto the 
     *   processors, and will fail spectacularly if not.
     *
     *   If overdense is set : computes the overdensity
     *   Note that this zeros grid before beginning
     *
     *   This routine is a modified version of code from Richard Scalzo.
     */
    void CIC(Vec v, const Particle &pp, bool overdense=true);
    // Helper routine -- note this is part of grid, not particle
    bool TestSlabDecompose(const Particle &pp);
    /* Vec out = Interp3d(Vec v, const Particle &pp) 
     *    Interpolate the grid onto the particle positions
     *
     *    Again, this requires the particles to be slab decomposed, so that this is done 
     *    locally.
     */
    Vec Interp3d(Vec v, const Particle &pp); 


    /* Vec dv = Deriv(Vec v, int dim) 
     *    Computes the derivative in the dim'th direction.
     *    The derivatives are computed using a central differencing algorithm, wrapping
     *    around at the boundaries.
     *
     *    The output vector is defined identically to the input vector, so can be used with the 
     *    same grid definitions.
     */
    Vec Deriv(Vec v, int dim);

};



//-------------------------------------------------------------------------------------------
//
// Density grid helper routines 
// 
// ------------------------------------------------------------------------------------------
class Delta {
  private :
    DensityGrid dg1;
    Vec _W;
    double nrand;
    double rmasksmooth;

  public :
    int N, nover; 
    double L;
    Mask3D *maskptr;
    Vec W;

    // Constructors and destructors
    Delta(int _N, double _L, int _nover, double thresh, Mask3D& _mask);
    Delta(int _N, double _L, int _nover, double thresh, Mask3D& _mask,
          Particle& pr, double smooth);
    ~Delta();
    /* ZeroMean : 
     *    Subtract the mean density. 
     *    Note that this first multiplies the grid by W, then computes the mean and subtracts it.
     */
    void ZeroMean(Vec &delta);
    /* BuildDensityGrid :
     *    Construct delta for pp using the masked field.
     *    Note that pp is altered by this routine -- trimmed to the mask.
     *
     *    IMPORTANT : You *MUST* construct a window before calling this routine.
     */
    void BuildDensityGrid(Particle& pp, Vec& delta);
    void BuildDensityGridSmooth(double smooth, Particle& pp, Vec& delta);
    /* HoffmanRibak(f, c, kprior, pkprior) :
     *    Implements the Hoffman-Ribak algorithm to generare a constrained realization.
     *
     *     f : The output constrained realization
     *     c : The constraint vector -- the region it is defined is specified by W
     *     kprior, pkprior : The power spectrum assumed to fill in the region
     *
     *     c is destroyed, and f must be allocated.
     */     
    void HoffmanRibak(Vec& c, vector<double>& kprior, vector<double>& pkprior, int seed, double tol=1.e-5);
};


void BuildDensityGrid(int N, double L, Mask3D& mask, Particle& pp, int nover, double thresh, 
                      Vec& delta, Vec& W, double smooth=0.0);
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









//-------------------------------------------------------------------------------------------
//
// Potential Solvers
// 
// ------------------------------------------------------------------------------------------

// Operators
//PetscErrorCode _TESTING_RealPBC(Mat a, Vec x, Vec y);


enum PotentialOp { REALPBC, CARTPBC, RADIAL};
/* REALPBC -- Real Space, Periodic Boundary Conditions
   CARTPBC  -- Cartesian coordinates, periodic boundary conditions, with z-space in the z-direction 
   RADIAL -- Radial wide angle redshift space distortions
*/

class PotentialSolve {

  private :
    DensityGrid _dg1; // We don't really need the full power of DensityGrid, but what the heck
    Mat a; // Apologies for the truly atrocious name, but this is the discrete matrix operator
    MatNullSpace mnull; // The null space
    KSP solver;

  public :
    PetscInt Ng;
    double L, dx, rtol, atol;
    int maxit;

    // Constructors
    PotentialSolve(int N, double _L, 
      int _maxit=DEFAULT_MAXIT, double _rtol=DEFAULT_RTOL, double _atol=DEFAULT_ATOL); 
    ~PotentialSolve();

    // Build matrix operators below
    // These are public, but not really meant for use....
    /* _Real :
     *    The real space operator 
     */
    void _BuildCARTPBC(double fval=0.0);
    void _BuildRADIAL(double fval, vector<double>& origin);

    // Set up the operator, this allows us to reuse the same
    void SetupOperator(PotentialOp optype, double fval=0.0, vector<double> origin=vector<double>(3));

    // Access the matrix
    Mat GetOperator() {return a;};

    // Solver
    // The origin is assumed to be in box coordinates
    bool Solve(Vec delta, Vec pot, double bias=1.0);
};




#endif /* RECON_H_ */
