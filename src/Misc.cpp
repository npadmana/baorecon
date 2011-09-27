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
  PetscTruth flg;
  VecValid(v, &flg);
  if (flg) VecDestroy(v);
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



