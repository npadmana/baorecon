#include "Recon.h"

using namespace std;

void DensityGrid::_cleanFFTWplan(rfftwnd_mpi_plan &p) {
  if (p) rfftwnd_mpi_destroy_plan(p);
}


DensityGrid::~DensityGrid() {
  _cleanFFTWplan(plan);
  _cleanFFTWplan(iplan);
}


void DensityGrid::slab(int &lo, int &hi, bool config) {
  if (config) {
    lo = local_x_start;
    hi = local_x_start + local_nx;
  } else {
    lo = local_y_start_after_transpose;
    hi = local_y_start_after_transpose + local_ny_after_transpose;
  }
}

DensityGrid::DensityGrid(int N, double _L) {
  plan=NULL; iplan=NULL;
  (*this).Init(N, _L);
}


void DensityGrid::Init(int N, double _L) {
  int rank, size;
  PetscInt ntot;
  
  // Get MPI rank and size
  MPI_Comm_rank(PETSC_COMM_WORLD, &rank);
  MPI_Comm_size(PETSC_COMM_WORLD, &size);

  // Make sure that size divides N --- this makes everything simpler
  if ((N%size)!=0) RAISE_ERR(99,"Number of processors should divide N");
  if ((N%2)!=0) RAISE_ERR(99, "Really?? Odd numbered grid??");

  // Set basic parameters
  L = _L; Ng = N; Ng2 = Ng+2;

  // Generate FFTW plans
  /* create the forward and backward plans: */
  _cleanFFTWplan(plan); _cleanFFTWplan(iplan);
  plan = rfftw3d_mpi_create_plan(PETSC_COMM_WORLD,
                                    Ng, Ng, Ng,
                                    FFTW_REAL_TO_COMPLEX,
                                    FFTW_ESTIMATE);
  iplan = rfftw3d_mpi_create_plan(PETSC_COMM_WORLD,
  /* dim.'s of REAL data --> */  Ng, Ng, Ng,
                                 FFTW_COMPLEX_TO_REAL,
                                 FFTW_ESTIMATE);
  rfftwnd_mpi_local_sizes(plan, &local_nx, &local_x_start,
                      &local_ny_after_transpose,
                      &local_y_start_after_transpose,
                      &total_local_size);
  // We assume total local size = (N/size)*N*(N+2) -- fail if not true
  ntot = Ng*Ng*Ng2;
  lstart = (ntot/size) * rank;
  lend = lstart + (ntot/size);
  nghost = 2*Ng*Ng2;
  if ((lend-lstart)!=total_local_size) RAISE_ERR(99,"Local sizes not expected");
  if ((Ng/size)!=local_nx) RAISE_ERR(99,"local_x not expected");
  
}
  
Vec DensityGrid::Allocate() {
  PetscInt ntot, nghost1, tmp;
  vector<PetscInt> ighost;
  Vec grid;

  ntot = Ng*Ng*Ng2;
  nghost1 = Ng*Ng2;
  // Set ghost indices
  for (int ii=0; ii < nghost1; ii++) {
    tmp = (PetscInt) ((lend+ii)%ntot); 
    ighost.push_back(tmp);
  }
  for (int ii=nghost1; ii > 0; --ii) {
    tmp = (PetscInt) ((lstart-ii+ntot)%ntot); 
    ighost.push_back(tmp);
  }

  // Now set up the vector and zero it
  VecCreateGhost(PETSC_COMM_WORLD,total_local_size, ntot, nghost, &ighost[0], &grid);
  VecSet(grid, 0.0);
  return grid; 
}





void DensityGrid::Pull(Vec v, bool config) {
    if (config) {
      VecGhostUpdateBegin(v,INSERT_VALUES,SCATTER_FORWARD);
      VecGhostUpdateEnd(v,INSERT_VALUES,SCATTER_FORWARD);
      VecGhostGetLocalForm(v, &_grid);
      VecGetArray(_grid, &_ldata);
    } else {
      VecGetArray(v, &_ldata);
    }
    _config=config;
}


void DensityGrid::Push(Vec v, InsertMode iora, bool update) {
  if (_config) {
    VecRestoreArray(_grid, &_ldata);
    VecGhostRestoreLocalForm(v, &_grid);
    if (update) {
      VecGhostUpdateBegin(v, iora,SCATTER_REVERSE);
      VecGhostUpdateEnd(v, iora, SCATTER_REVERSE);
    }
  } else {
    VecRestoreArray(v, &_ldata);
  }

}






double& DensityGrid::operator()(int ix, int iy, int iz, int ic) {
  PetscInt pos;

  if (_config) {
    // We are in configuration space -- allow single point ghosting
    if ((iy > (Ng)) || (iy < -1)) RAISE_ERR(99, "Out of bounds in iy");
    if ((iz > (Ng)) || (iz < -1)) RAISE_ERR(99, "Out of bounds in iz");
    iy = (iy+Ng)%Ng;
    iz = (iz+Ng)%Ng;
    if ((ix > (local_x_start+local_nx)) || (ix < (local_x_start-1))) {cout << "ix:" << ix << " "  << local_x_start << " "  << local_nx << endl;RAISE_ERR(99, "Out of bounds in ix")} 
    pos = (ix*Ng + iy) * Ng2 + iz - lstart;
    if (pos < 0) pos = total_local_size + nghost + pos;
    if (pos >= (total_local_size+nghost)) RAISE_ERR(99,"pos out of range");
  } else {
    // We are in Fourier space
    // Data are transposed!
    if ((ix > (Ng-1)) || (ix < 0)) RAISE_ERR(99, "Out of bounds in ix");
    if (   (iy >= (local_y_start_after_transpose+local_ny_after_transpose)) ||
           (iy < local_y_start_after_transpose)) 
                   RAISE_ERR(99, "Out of bounds in iy");
    if ((iz > (Ng/2+1)) || (iz < 0)) RAISE_ERR(99, "Out of bounds in iz");
    pos = (((iy-local_y_start_after_transpose)*Ng+ix) * (Ng/2 + 1) + iz)*2 + ic; 
  }

  return _ldata[pos];
}


void DensityGrid::ZeroPad(Vec v, double val) {
  double *data;
  VecGetArray(v, &data); // No worries about ghosts here
  // Zero out the padded region
  for (int ix =0; ix < local_nx; ++ix) 
    for (int iy=0; iy < Ng; ++iy) 
      for (int iz=Ng; iz < Ng2; ++iz) 
        data[(ix*Ng + iy) * Ng2 + iz] = val;
  VecRestoreArray(v, &data);
}

double DensityGrid::kval(int ix, int iy, int iz) {
  double ii[3];
  ii[0] = (double) ((ix > Ng/2) ? ix - Ng : ix),
  ii[1] = (double) ((iy > Ng/2) ? iy - Ng : iy),
  ii[2] = (double) ((iz > Ng/2) ? iz - Ng : iz);
  double kmod = sqrt (ii[0]*ii[0] + ii[1]*ii[1] + ii[2]*ii[2]);

  return kmod * (2.0*M_PI/L);
}


double DensityGrid::rval(int ix, int iy, int iz) {
  double ii[3];
  ii[0] = (double) ((ix > Ng/2) ? ix - Ng : ix),
  ii[1] = (double) ((iy > Ng/2) ? iy - Ng : iy),
  ii[2] = (double) ((iz > Ng/2) ? iz - Ng : iz);
  double rmod = sqrt (ii[0]*ii[0] + ii[1]*ii[1] + ii[2]*ii[2]);

  return rmod * (L/Ng);
}
  


void DensityGrid::FFT(Vec v, bool &config) {

  double *data, *work;

  // Get the data
  VecGetArray(v, &data);
  work = new double[total_local_size];

  if (config) {
    // We are in configuration space
    rfftwnd_mpi(plan, 1, data, work, FFTW_TRANSPOSED_ORDER);
    config=false;
  } else {
    // We are in Fourier space
    rfftwnd_mpi(iplan, 1, data, work, FFTW_TRANSPOSED_ORDER);
    config = true;
  }

  VecRestoreArray(v, &data);
  delete[] work;

  if (!config) {
    // Normalization : normalize the r2k transform 
    VecScale(v, 1./pow((double)Ng,3));
  } else {
    ZeroPad(v); // Zero pad
  }

}

void DensityGrid::ScaleShift(Vec v, double a, double b) {

  (*this).Pull(v);
  for (int ix = local_x_start; ix < local_x_start+local_nx; ++ix) 
    for (int iy = 0; iy < Ng; ++iy) 
      for (int iz = 0; iz < Ng; ++iz) 
        (*this)(ix, iy, iz) = (*this)(ix, iy, iz)*a + b;
  (*this).Push(v, INSERT_VALUES, false);
}

void DensityGrid::PrintCells (Vec v,double smooth) {       // RS:  hackery

  int di = (int)(smooth*Ng/L) + 1;
  (*this).Pull(v);
  for (int ix = local_x_start; ix < local_x_start+local_nx; ix += di) 
  {
    for (int iy = 0; iy < Ng; iy += di) 
      for (int iz = 0; iz < Ng; iz += di)
        if ((*this)(ix,iy,iz) > 1e-3)
        PetscSynchronizedPrintf (PETSC_COMM_WORLD, "%4d %4d %4d %8.2g\n",
          ix, iy, iz, (*this)(ix, iy, iz));
    PetscSynchronizedFlush(PETSC_COMM_WORLD);
  }
  (*this).Push(v, INSERT_VALUES, false);
}

void DensityGrid::CIC(Vec v, const Particle &pp, bool overdense) {
  // variables to store particle positions
  double *_px, *_py, *_pz, *_pw;
  PetscInt lo, hi, nlocal;
  
  int ii[3], ix, iy,iz; 
  double dx, dy, dz, x0, y0, z0, w0;
  double rho_mean;

  // Mean density
  rho_mean = (double) pp.npart/pow((double) Ng, 3);

  // ZERO grid
  VecSet(v,0.0);

  // Make the grid available for updates
  (*this).Pull(v);

  // Access the local particles
  VecGetOwnershipRange(pp.px, &lo, &hi);
  nlocal = hi-lo;
  VecGetArray(pp.px, &_px); 
  VecGetArray(pp.py, &_py);
  VecGetArray(pp.pz, &_pz);
  VecGetArray(pp.pw, &_pw);

  for (PetscInt ip=0; ip < nlocal; ++ip) {
      x0 = periodic(_px[ip]/L); // Scaled periodic versions of coordinates
      y0 = periodic(_py[ip]/L);
      z0 = periodic(_pz[ip]/L);
      w0 = _pw[ip];
      ix = (int)(Ng*x0); ii[0] = (ix+1); dx = Ng*x0-ix;
      iy = (int)(Ng*y0); ii[1] = (iy+1); dy = Ng*y0-iy;
      iz = (int)(Ng*z0); ii[2] = (iz+1); dz = Ng*z0-iz;

      // Do the interpolation.  Go ahead and use the indexing operator
      // for the density field.
      (*this)(ix,   iy   ,iz   ) += (1.0-dx)*(1.0-dy)*(1.0-dz)*w0;
      (*this)(ii[0],iy   ,iz   ) +=      dx *(1.0-dy)*(1.0-dz)*w0;
      (*this)(ix   ,ii[1],iz   ) += (1.0-dx)*     dy *(1.0-dz)*w0;
      (*this)(ix   ,iy   ,ii[2]) += (1.0-dx)*(1.0-dy)*     dz *w0;
      (*this)(ii[0],ii[1],iz   ) +=      dx *     dy *(1.0-dz)*w0;
      (*this)(ii[0],iy   ,ii[2]) +=      dx *(1.0-dy)*     dz *w0;
      (*this)(ix   ,ii[1],ii[2]) += (1.0-dx)*     dy *     dz *w0;
      (*this)(ii[0],ii[1],ii[2]) +=      dx *     dy *     dz *w0;
  }
  VecRestoreArray(pp.px, &_px);
  VecRestoreArray(pp.py, &_py);
  VecRestoreArray(pp.pz, &_pz);
  VecRestoreArray(pp.pw, &_pw);


  // Push out updates 
  (*this).Push(v, ADD_VALUES, true);


  // Normalize
  if (overdense) (*this).ScaleShift(v, 1./rho_mean, -1.0);

}

bool DensityGrid::TestSlabDecompose(const Particle &pp) {
  // variables to store particle positions
  int rank; 
  double *_px;
  PetscInt lo, hi, nlocal;
  int ix; 
  double x0;
  bool retval;
  MPI_Comm_rank(PETSC_COMM_WORLD, &rank);

  // Access the local particles
  VecGetOwnershipRange(pp.px, &lo, &hi);
  nlocal = hi-lo;
  VecGetArray(pp.px, &_px); 

  retval = true;
  for (PetscInt ip=0; ip < nlocal; ++ip) {
      x0 = periodic(_px[ip]/L); // Scaled periodic versions of coordinates
      ix = (int)(Ng*x0);
      if ((ix < local_x_start) || (ix >= (local_x_start+local_nx))) {
	cout << rank << " " << local_x_start << " " << local_nx << " " << _px[ip] << " " << ix << endl;
	return false;
      }
  }
  VecRestoreArray(pp.px, &_px);

  return retval;
}


void DensityGrid::GaussSmooth(Vec v, double R) {
  bool config=true;
  double gauss, k;
  // Note that this routine assumes we are starting in real space
  FFT(v, config);
  if (config) RAISE_ERR(99, "Assertion failed : should be in F-space");
  Pull(v, false);
  
  // Loop over k -- note we are transposed.
  for (int iy = local_y_start_after_transpose; iy < local_y_start_after_transpose+local_ny_after_transpose; ++iy)
    for (int ix=0; ix < Ng; ++ix) 
      for (int iz=0; iz < (Ng/2+1); ++iz) {
        k = kval(ix, iy, iz);
        gauss = exp(-(k*k*R*R)/2.0);
        (*this)(ix,iy,iz,0) *= gauss;
        (*this)(ix,iy,iz,1) *= gauss;
  }
  Push(v, INSERT_VALUES, false);
  FFT(v, config);
}


void DensityGrid::kConvolve(Vec v, vector<double> &kvec, vector<double> &fkvec) {
// This routine mostly follows the GaussSmooth case
  bool config=true;
  double fval,k;
  // Note we assume we start in configuration space
  FFT(v, config);
  if (config) RAISE_ERR(99, "Assertion failed : should be in F-space");
  Pull(v, false);

  // Set up the spline class
  gsl_interp_accel *acc = gsl_interp_accel_alloc();
  gsl_spline *interp = gsl_spline_alloc(gsl_interp_akima, kvec.size());
  gsl_spline_init(interp, &kvec[0], &fkvec[0], kvec.size());
  double kmin, kmax; 
  kmin = kvec[0]; // Vector must be sorted
  kmax = kvec[kvec.size()-1];


  // Loop over k -- note we are transposed.
  for (int iy = local_y_start_after_transpose; iy < local_y_start_after_transpose+local_ny_after_transpose; ++iy)
    for (int ix=0; ix < Ng; ++ix) 
      for (int iz=0; iz < (Ng/2+1); ++iz) {
        k = kval(ix, iy, iz);
        if ((k > kmin) && (k < kmax)) {fval = gsl_spline_eval(interp,k, acc);} else {fval = 0.0;}
        (*this)(ix,iy,iz,0) *= fval;
        (*this)(ix,iy,iz,1) *= fval;
  }
  Push(v, INSERT_VALUES, false);
  FFT(v, config);
  gsl_interp_accel_free(acc);
  gsl_spline_free(interp);
}



// RS 2010/06/22:  Commented as I found appropriate.
void DensityGrid::XiFFT(Vec v, double Rsmooth, PkStruct& Xi) {
  bool config=true;
  double c1, c2;

  // Compute the correlation function on the grid 
  FFT(v, config);
  if (config) RAISE_ERR(99, "Assertion failed : should be in F-space");
  Pull(v, false);
  
  // Loop over k -- note we are transposed.
  double k, gauss;
  for (int iy = local_y_start_after_transpose; iy < local_y_start_after_transpose+local_ny_after_transpose; ++iy)
    for (int ix=0; ix < Ng; ++ix) 
      for (int iz=0; iz < (Ng/2+1); ++iz) {
        k = kval(ix, iy, iz);
        gauss = exp(-(k*k*Rsmooth*Rsmooth));
        c1 = (*this)(ix, iy, iz,0); c2 = (*this)(ix,iy,iz,1);
        (*this)(ix,iy,iz,0) = ((c1*c1) + (c2*c2))*gauss;
        (*this)(ix,iy,iz,1) = 0.0;
  }
  Push(v, INSERT_VALUES, false);
  FFT(v, config);
  Pull(v);

  // Now loop over all points counting up modes
  for (int ix = local_x_start; ix < local_x_start+local_nx; ++ix) 
    for (int iy = 0; iy < Ng; ++iy)
      for (int iz = 0; iz < Ng; ++iz) 
        Xi.accum(rval(ix,iy,iz), (*this)(ix,iy,iz));
  Push(v, INSERT_VALUES, false);
  Xi.finalize();

}

void DensityGrid::XiFFT_W(Vec v, Vec W1, double Rsmooth, PkStruct& Xi) {
  bool config=true;
  double c1, c2;


  {
    // Compute the correlation function on the grid -- first for v
    FFT(v, config);
    if (config) RAISE_ERR(99, "Assertion failed : should be in F-space");
    Pull(v, false);
    
    // Loop over k -- note we are transposed.
    double k, gauss;
    for (int iy = local_y_start_after_transpose; iy < local_y_start_after_transpose+local_ny_after_transpose; ++iy)
      for (int ix=0; ix < Ng; ++ix) 
        for (int iz=0; iz < (Ng/2+1); ++iz) {
          k = kval(ix, iy, iz);
          gauss = exp(-(k*k*Rsmooth*Rsmooth));
          c1 = (*this)(ix, iy, iz,0); c2 = (*this)(ix,iy,iz,1);
          (*this)(ix,iy,iz,0) = ((c1*c1) + (c2*c2))*gauss;
          (*this)(ix,iy,iz,1) = 0.0;
    }
    Push(v, INSERT_VALUES, false);
    FFT(v, config);
  }

  // Compute the correlation function on the grid -- now for W
  Vec W;
  VecDuplicate(W1, &W);
  VecCopy(W1, W);
  {
    config=true;
    FFT(W, config);
    if (config) RAISE_ERR(99, "Assertion failed : should be in F-space");
    Pull(W, false);
    
    // Loop over k -- note we are transposed.
    for (int iy = local_y_start_after_transpose; iy < local_y_start_after_transpose+local_ny_after_transpose; ++iy)
      for (int ix=0; ix < Ng; ++ix) 
        for (int iz=0; iz < (Ng/2+1); ++iz) {
          c1 = (*this)(ix, iy, iz,0); c2 = (*this)(ix,iy,iz,1);
          (*this)(ix,iy,iz,0) = ((c1*c1) + (c2*c2));
          (*this)(ix,iy,iz,1) = 0.0;
    }
    Push(W, INSERT_VALUES, false);
    FFT(W, config);
  }

  // Now loop over all points counting up modes
  Pull(v);
  DensityGrid tmp(Ng, L); tmp.Pull(W);
  for (int ix = local_x_start; ix < local_x_start+local_nx; ++ix) 
    for (int iy = 0; iy < Ng; ++iy)
      for (int iz = 0; iz < Ng; ++iz) 
        Xi.accum(rval(ix,iy,iz),(*this)(ix, iy, iz)/(tmp(ix,iy,iz)+1.e-20));
  Xi.finalize();    
  Push(v, INSERT_VALUES, false);
  tmp.Push(W, INSERT_VALUES, false);
  _mydestroy(W);

}


Vec DensityGrid::Deriv(Vec v, int dim) {
  // Allocate dv
  Vec dv;
  VecDuplicate(v, &dv);
  VecSet(dv,0.0);
  DensityGrid dg2(Ng, L); // To address dv

  // Actually compute the derivative
  Pull(v);
  dg2.Pull(dv);

  vector<int> ii(3), ip(3), im(3);
  for (ii[0]=local_x_start; ii[0] < local_x_start+local_nx; ++ii[0])
    for (ii[1]=0; ii[1] < Ng; ++ii[1]) 
      for (ii[2]=0; ii[2] < Ng; ++ii[2]) {
        ip = ii; im = ii;
        ip[dim]++; im[dim]--;
        dg2(ii) = (*this)(ip) - (*this)(im);
      }
  dg2.Push(dv, INSERT_VALUES, false);
  Push(v,INSERT_VALUES, false);
  double dx = L/Ng;
  VecScale(dv, 1./(2*dx));

  return dv;
}


Vec DensityGrid::Interp3d(Vec v, const Particle& pp) {
  // variables to store particle positions
  double *_px, *_py, *_pz, *_out;
  PetscInt lo, hi, nlocal; 
  int ix, iy, iz, ii[3]; 
  double dx, dy, dz, x0, y0, z0, val;
  Vec out;
    

  // Set output vector
  VecDuplicate(pp.px, &out);

  // Make the grid available for updates
  (*this).Pull(v);

  // Access the local particles
  VecGetOwnershipRange(pp.px, &lo, &hi);
  nlocal = hi-lo;
  VecGetArray(pp.px, &_px); 
  VecGetArray(pp.py, &_py);
  VecGetArray(pp.pz, &_pz);
  VecGetArray(out, &_out);

  for (PetscInt ip=0; ip < nlocal; ++ip) {
      x0 = periodic(_px[ip]/L); // Scaled periodic versions of coordinates
      y0 = periodic(_py[ip]/L);
      z0 = periodic(_pz[ip]/L);
      ix = (int)(Ng*x0); ii[0] = (ix+1); dx = Ng*x0-ix; // Note that we don't need to mod with nx
      iy = (int)(Ng*y0); ii[1] = (iy+1); dy = Ng*y0-iy;
      iz = (int)(Ng*z0); ii[2] = (iz+1); dz = Ng*z0-iz;

      // Do the interpolation.  Go ahead and use the indexing operator
      // for the density field.
      // Note that this is the opposite of the CIC operation.
      val = 
      (*this)(ix,   iy   ,iz   ) * (1.0-dx)*(1.0-dy)*(1.0-dz) +
      (*this)(ii[0],iy   ,iz   ) *      dx *(1.0-dy)*(1.0-dz) +
      (*this)(ix   ,ii[1],iz   ) * (1.0-dx)*     dy *(1.0-dz) +
      (*this)(ix   ,iy   ,ii[2]) * (1.0-dx)*(1.0-dy)*     dz  +
      (*this)(ii[0],ii[1],iz   ) *      dx *     dy *(1.0-dz) +
      (*this)(ii[0],iy   ,ii[2]) *      dx *(1.0-dy)*     dz  +
      (*this)(ix   ,ii[1],ii[2]) * (1.0-dx)*     dy *     dz  +
      (*this)(ii[0],ii[1],ii[2]) *      dx *     dy *     dz ;
      _out[ip] = val;
  }
  VecRestoreArray(pp.px, &_px);
  VecRestoreArray(pp.py, &_py);
  VecRestoreArray(pp.pz, &_pz);
  VecRestoreArray(out, &_out);


  // Cleanup
  Push(v, INSERT_VALUES, false);
  return out;
}


void DensityGrid::FakeGauss(int seed, Vec v, vector<double> &kvec, vector<double>& Pkvec) {
  // We build a Gaussian random field as follows :
  //   -- Fill the vector with white noise
  //   -- FFT
  //   -- Scale each of the Fourier components by sqrt(P(k))
  //   -- FFT back
  //   -- Normalize
  
  // MPI rank
  int rank;
  MPI_Comm_rank(PETSC_COMM_WORLD, &rank);

  // Initialize v
  VecSet(v, 0.0);
  
  // Fill with random values
  int lo, hi;
  slab(lo, hi);
  gsl_rng *ran = gsl_rng_alloc(gsl_rng_mt19937);
  gsl_rng_set(ran, seed + 100*rank);
  Pull(v);
  for (int ix=lo; ix < hi; ++ix) 
    for (int iy=0; iy < Ng; ++iy) 
      for (int iz=0; iz < Ng; ++iz)
        (*this)(ix, iy, iz) = gsl_ran_gaussian_ziggurat(ran, 1.0);
  Push(v, INSERT_VALUES, false);
  gsl_rng_free(ran);

  bool config=true;
  double fval,k;
  // Note we assume we start in configuration space
  FFT(v, config);
  if (config) RAISE_ERR(99, "Assertion failed : should be in F-space");
  Pull(v, false);

  // Set up the spline class
  gsl_interp_accel *acc = gsl_interp_accel_alloc();
  gsl_spline *interp = gsl_spline_alloc(gsl_interp_akima, kvec.size());
  gsl_spline_init(interp, &kvec[0], &Pkvec[0], kvec.size());
  double kmin, kmax; 
  kmin = kvec[0]; // Vector must be sorted
  kmax = kvec[kvec.size()-1];


  // Loop over k -- note we are transposed.
  for (int iy = local_y_start_after_transpose; iy < local_y_start_after_transpose+local_ny_after_transpose; ++iy)
    for (int ix=0; ix < Ng; ++ix) 
      for (int iz=0; iz < (Ng/2+1); ++iz) {
        k = kval(ix, iy, iz);
        if ((k > kmin) && (k < kmax)) {fval = sqrt(gsl_spline_eval(interp, k, acc));} else {fval = 0.0;}
        (*this)(ix,iy,iz,0) *= fval;
        (*this)(ix,iy,iz,1) *= fval;
  }
  gsl_interp_accel_free(acc);
  gsl_spline_free(interp);
  Push(v, INSERT_VALUES, false);
  FFT(v, config);

  // Clean up and normalize
  ZeroPad(v);
  fval = pow(sqrt(Ng/L),3);
  VecScale(v, fval);
}


double DensityGrid::CICKern(int ix, int iy, int iz) {
  double kmin, rcell, kx2, ky2, kz2;
  double kern;

  kmin = 2.0*M_PI/L;
  rcell = L/Ng/2.0;
  // C++ passes by value
  if (ix > (Ng/2)) ix = ix - Ng;
  if (iy > (Ng/2)) iy = iy - Ng;
  // We don't need one for z since this is the folded direction.
  kx2 = kmin*kmin*ix*ix;
  ky2 = kmin*kmin*iy*iy;
  kz2 = kmin*kmin*iz*iz;
  kern = 1.0;
  if (kx2 > 0) kern *= sin(sqrt(kx2)*rcell)/(sqrt(kx2)*rcell);
  if (ky2 > 0) kern *= sin(sqrt(ky2)*rcell)/(sqrt(ky2)*rcell);
  if (kz2 > 0) kern *= sin(sqrt(kz2)*rcell)/(sqrt(kz2)*rcell);
  
  return (kern*kern);
}


void DensityGrid::PkCIC(Vec v, PkStruct& pkmean, PkStruct& pkdec) {
  // FFT 
  bool config=true;
  FFT(v, config);
  Pull(v, false);

  // Now compute the power spectrum
  double k1, pk1, kern, fac;
  for (int iy = local_y_start_after_transpose; iy < local_y_start_after_transpose+local_ny_after_transpose; ++iy)
    for (int ix=0; ix < Ng; ++ix) 
      for (int iz=0; iz < (Ng/2+1); ++iz) {
        if ((iz==0)||(iz==(Ng/2))) {fac=1.0;} else {fac=2.0;}
        k1 = kval(ix, iy, iz);
        kern = CICKern(ix, iy, iz);
        pk1 = pow((*this)(ix,iy,iz,0), 2) + pow((*this)(ix,iy,iz,1),2);
        pkmean.accum(k1, pk1, fac);
        pkdec.accum(k1, pk1/(kern*kern), fac);
  }
  Push(v, INSERT_VALUES, false);
  
  // Clean up
  pkmean.finalize();
  pkdec.finalize();
  VecScale(pkmean.pkvec, L*L*L);
  VecScale(pkdec.pkvec, L*L*L);
}

