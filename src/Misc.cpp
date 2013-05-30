#include "Recon.h"


// Periodic
double periodic(double x, double L) {
  double tmp;

  tmp = x/L;
  if (tmp>=1.0) tmp = tmp-floor(tmp);
  if (tmp< 0.0) tmp = 1.0 -(-tmp-floor(-tmp));

  return tmp*L;
}

// Safe vector cleanup
void _mydestroy(Vec &v) {
  PetscBool flg;
  VecValid(v, &flg);
  if (flg) VecDestroy(&v);
}


Shell::Shell(double x, double y, double z, double rmin, double rmax) {
  _x0 = x; _y0 = y; _z0 = z;
  _rmin = rmin; _rmax = rmax;
}

bool Shell::operator()(double x, double y, double z) {

  double dx, dy, dz, rr;
  dx = x-_x0; dy = y-_y0; dz= z-_z0;

  rr = sqrt(dx*dx + dy*dy + dz*dz);
  if ((rr >= _rmin) && (rr < _rmax)) {return true;} else {return false;}
}

PkStruct::PkStruct(double _kmin, double _dk, int _Nbins) {
  kmin = _kmin; dk = _dk; Nbins = _Nbins;
  VecCreateMPI(PETSC_COMM_WORLD, PETSC_DECIDE, Nbins, &kvec);
  VecDuplicate(kvec, &pkvec);
  VecDuplicate(kvec, &nmodes);
  VecSet(kvec,0.0);
  VecSet(pkvec, 0.0);
  VecSet(nmodes, 0.0);

  VecGetOwnershipRange(kvec, &lo, &hi);
}

PkStruct::~PkStruct() {
  _mydestroy(kvec); _mydestroy(pkvec); _mydestroy(nmodes);
}

void PkStruct::accum(double k1, double pk1, double fac) {
  int ibin;

  if (k1 >= kmin) {
    ibin = (int) ((k1 - kmin)/dk);
    if (ibin < Nbins) {
      VecSetValue(kvec, ibin, k1*fac, ADD_VALUES);
      VecSetValue(pkvec, ibin, pk1*fac, ADD_VALUES);
      VecSetValue(nmodes, ibin, fac, ADD_VALUES);
    }
  }
}

void PkStruct::finalize() {

  VecAssemblyBegin(pkvec); VecAssemblyEnd(pkvec);
  VecAssemblyBegin(nmodes); VecAssemblyEnd(nmodes);
  VecAssemblyBegin(kvec); VecAssemblyEnd(kvec);

  //Normalize and return 
  VecShift(nmodes, 1.e-20);
  VecPointwiseDivide(pkvec, nmodes);
  VecPointwiseDivide(kvec, nmodes);

  VecGetArray(kvec, &_kvec);
  VecGetArray(pkvec, &_pkvec);
  VecGetArray(nmodes, &_nmodes);
}

double PkStruct::operator()(PetscInt ii, double& k1, double& N1) {
  if ((ii < lo)||(ii>=hi)) RAISE_ERR(99, "Out of range!");
  k1 = _kvec[ii-lo];
  N1 = _nmodes[ii-lo];
  return _pkvec[ii-lo];
}
double PkStruct::operator()(PetscInt ii) {
  if ((ii < lo)||(ii>=hi)) RAISE_ERR(99, "Out of range!");
  return _pkvec[ii-lo];
}


/* Histogram a PETSC vector 
 *
 * x is the vector
 * nbins -- number of bins
 * xmin, xmax -- histogram xmin, xmax -- assume uniform bins
 * hh -- output vector -- assumed to be defined.
 */
void VecHist(const Vec& x, int nbins, double xmin, double xmax, vector<double>& hh) {
  gsl_histogram *h1; 
  double *_x, x1;
  PetscInt lo, hi;
  vector<double> tmp(nbins);

  // Set up the histogram struct
  h1 = gsl_histogram_alloc(nbins);
  gsl_histogram_set_ranges_uniform(h1, xmin, xmax);

  // Get the array
  VecGetOwnershipRange(x, &lo, &hi);
  hi -= lo;
  VecGetArray(x, &_x);
  for (PetscInt ii=0; ii < hi; ++ii) {
    x1 = _x[ii];
    if (x1 < xmin) x1 = xmin;
    if (x1 >= xmax) x1 = xmax - 1.e-10;
    gsl_histogram_increment(h1, x1);
  }
  VecRestoreArray(x, &_x);
  
  // Fill the temporary output vector
  for (int ii =0; ii<nbins; ++ii) 
    tmp[ii] = gsl_histogram_get(h1, ii);

  // MPI Allreduce
  MPI_Allreduce(&tmp[0], &hh[0], nbins, MPI_DOUBLE, MPI_SUM, PETSC_COMM_WORLD); 

  // Clean up
  gsl_histogram_free(h1);
}


