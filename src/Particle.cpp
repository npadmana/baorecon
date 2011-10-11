#include "Recon.h"
#include <list>

using namespace std;


// Destructor
Particle::~Particle() {
  _mydestroy(px);
  _mydestroy(py);
  _mydestroy(pz);
  _mydestroy(pw);
};

// Copy constructor
Particle::Particle(const Particle& p) {
  npart = p.npart;
  VecDuplicate(p.px, &px);
  VecDuplicate(p.px, &py);
  VecDuplicate(p.px, &pz);
  VecDuplicate(p.pw, &pw);
  VecCopy(p.px, px);
  VecCopy(p.py, py);
  VecCopy(p.pz, pz);
  VecCopy(p.pw, pw);

}



// IO routines
int Particle::TPMReadSerial(const char *infilename, double L, bool zspace, bool pplel) {
  
   // Header structure
   struct FileHeader
   {
      int   npart;		// Total number of particles
      int   nsph;		// Number of gas particles
      int   nstar;		// Number of star particles
      float aa;			// Scale factor
      float softlen;		// Gravitational softening
   } header;

   // Declarations yanked from Martin's code
   int   j,n,nget,ngot;
   int    eflag,hsize;
   int rank;
   float  *ppos = NULL, *pvel = NULL;
   FILE   *fpr = NULL, *fpv = NULL;

   const int Ndim = 3;		// particles live in 3 dimensions
   const int BlkSiz = 262144;	// for reading in files

   // Get MPI rank 
   MPI_Comm_rank(PETSC_COMM_WORLD,&rank);

   // Rank-0 process will read in header information and bcast npart
   if (rank==0) {

      // Can we open the file?
      if ((fpr = fopen(infilename,"r")) == NULL) RAISE_ERR(99, "Failed opening file...");

      // Read endian flag.
      ngot = fread (&eflag,sizeof(int),1,fpr);
      if (ngot != 1) RAISE_ERR(99, "Can't read file");
      if (eflag != 1) RAISE_ERR(99,"Endian flag is not 1");

      // Read header size.
      ngot = fread(&hsize,sizeof(int),1,fpr);
      if (ngot != 1) RAISE_ERR(99,"Error reading file header");
      if (hsize != sizeof(struct FileHeader)) RAISE_ERR(99, "Incorrect header size");

      // Read and unpack header.
      ngot = fread(&header,sizeof(struct FileHeader),1,fpr);
      if (ngot != 1) RAISE_ERR(99, "Unable to read header");
      npart = header.npart;
   }
   MPI_Bcast(&npart, 1, MPI_INT, 0, PETSC_COMM_WORLD);
   
   // Allocate vectors
   VecCreateMPI(PETSC_COMM_WORLD, PETSC_DECIDE,npart,&px);
   VecDuplicate(px, &py); 
   VecDuplicate(px, &pz); 
   VecDuplicate(px, &pw); 


   // Continue to read
   if (rank==0) {

      // Initialize a second pointer "fpv" to the same file.  Position it
      // past the header and a good chunk of the file.
      // We are apparently reading in the positions of the particles first,
      // then the velocities.
      if ((fpv = fopen(infilename,"r")) == NULL) RAISE_ERR(99,"Failed to reopen file");
      if (fseek(fpv,2*sizeof(int)+sizeof(struct FileHeader)+
          header.npart*Ndim*sizeof(float),SEEK_CUR) != 0 )
         RAISE_ERR(99,"Failed to seek in file");
      // Allocate storage for the particles we are about to read in.
      // If we haven't succeeded by now, there's not much point.
      ppos = new float[Ndim*BlkSiz];
      pvel = new float[Ndim*BlkSiz];
   }

   // Now read the particles in.
   n = 0;
   double _p[3], _v[3], rr;
   while (n < npart)
   {
     if (rank==0) {
         if (header.npart - n > BlkSiz) nget = BlkSiz;
         else nget = header.npart-n;

         // Read a bunch of positions
         ngot = fread(ppos,sizeof(float),Ndim*nget,fpr);
         if (ngot != Ndim*nget) RAISE_ERR(99,"Error reading positions");

         // Read a bunch of velocities
         ngot = fread(pvel,sizeof(float),Ndim*nget,fpv);
         if (ngot != Ndim*nget) RAISE_ERR(99,"Error reading velocities");

         // Now assign these to particles.
         for (j=0; j<nget; j++)
         {  
            // Pull out values
            for (int idim = 0; idim < 3; ++idim) {
              _p[idim] = periodic(ppos[Ndim*j+idim]);
              _v[idim] = pvel[Ndim*j+idim];
            }

            // Simple redshift space conversion
            if (zspace) {
              // Two cases
              if (pplel) {
                // Plane parallel
                _p[2] += _v[2];
              } else {
                // Radial
                rr=0.0;
                for (int ii=0; ii < 3; ++ii) rr += pow(_p[ii],2);
                rr = sqrt(rr);
                for (int ii=0; ii < 3; ++ii) _p[ii] += _v[ii] * (_p[ii]/rr);
              }
            }

            // Load the values
            VecSetValue(px, n, periodic(_p[0])*L,INSERT_VALUES);
            VecSetValue(py, n, periodic(_p[1])*L,INSERT_VALUES);
            VecSetValue(pz, n, periodic(_p[2])*L,INSERT_VALUES);

            n+=1;
         }
     }

     // Synchronize every block
     VecAssemblyBegin(px); VecAssemblyEnd(px);
     VecAssemblyBegin(py); VecAssemblyEnd(py);
     VecAssemblyBegin(pz); VecAssemblyEnd(pz);
     MPI_Bcast(&n,1,MPI_INT,0, PETSC_COMM_WORLD);
     VecSet(pw, 1.0); // TPM files have no weights, so we set all weights to 1
   }

   

   if (rank==0) {
    // Close the files, de-allocate the memory and shut down
    if (fpr  != NULL) fclose(fpr);
    if (fpv  != NULL) fclose(fpv);
    if (ppos != NULL) delete[] ppos;
    if (pvel != NULL) delete[] pvel;
   }

   return 0;
}

int Particle::AsciiReadWeightedSerial(const char *infilename)
// ---------------------------------------------------------------------------
// RS 2010/06/29:  ASCII file reader.  Reads a three-column format,
// box-coordinates (x,y,z).  For randoms and/or rotated Las Damas format.
// ---------------------------------------------------------------------------
{
   FILE *fpr;
   int rank;
   char buf[256];
   vector<double> tmpx, tmpy, tmpz, tmpw;
   vector<PetscInt> idx;

   // Get MPI rank 
   MPI_Comm_rank(PETSC_COMM_WORLD,&rank);
   
   // Rank-0 process will read in header information and bcast npart
   PetscPrintf (PETSC_COMM_WORLD, "Opening file...\n");
   if (rank==0)
   {
      // Can we open the file?
      if ((fpr = fopen(infilename,"r")) == NULL)
         RAISE_ERR(99, "Failed opening file...");
      npart = 0;
   }

   // Read the file.
   PetscPrintf (PETSC_COMM_WORLD, "Reading from file...\n");
   if (rank==0) for(;;)
   {
      fgets (buf, 255, fpr);
      if (feof(fpr)) break;
      if (buf[0] == '#') continue;

      double x, y, z, w;
      sscanf (buf, "%lf %lf %lf %lf", &x, &y, &z, &w);
      tmpx.push_back(x); tmpy.push_back(y); tmpz.push_back(z); tmpw.push_back(w); idx.push_back(npart); npart++;
   }

   // Number of particles = however many were read in.
   MPI_Bcast(&npart, 1, MPI_INT, 0, PETSC_COMM_WORLD);
   
   // Allocate vectors
   PetscPrintf (PETSC_COMM_WORLD, "Allocating vectors...\n");
   VecCreateMPI(PETSC_COMM_WORLD,PETSC_DECIDE,npart,&px);
   VecDuplicate(px, &py); 
   VecDuplicate(px, &pz); 
   VecDuplicate(px, &pw);

   // Fill the vectors by looping through the lists.
   PetscPrintf (PETSC_COMM_WORLD, "Filling vectors...\n");
   if (rank==0)
   {
      VecSetValues(px, npart, &idx[0], &tmpx[0], INSERT_VALUES);
      VecSetValues(py, npart, &idx[0], &tmpy[0], INSERT_VALUES);
      VecSetValues(pz, npart, &idx[0], &tmpz[0], INSERT_VALUES);
      VecSetValues(pw, npart, &idx[0], &tmpw[0], INSERT_VALUES);
   }

   // Sync up the vectors in the different processes.
   PetscPrintf (PETSC_COMM_WORLD, "Assembling vectors...\n");
   VecAssemblyBegin(px); VecAssemblyEnd(px);
   VecAssemblyBegin(py); VecAssemblyEnd(py);
   VecAssemblyBegin(pz); VecAssemblyEnd(pz);
   VecAssemblyBegin(pw); VecAssemblyEnd(pw);
   

   // Close the files, de-allocate the memory and shut down.
   PetscPrintf (PETSC_COMM_WORLD, "Closing file & finishing.\n");
   if (rank==0 && fpr != NULL) fclose(fpr);
   return 0;
}

int Particle::AsciiReadSerial(const char *infilename)
// ---------------------------------------------------------------------------
// RS 2010/06/29:  ASCII file reader.  Reads a three-column format,
// box-coordinates (x,y,z).  For randoms and/or rotated Las Damas format.
// ---------------------------------------------------------------------------
{
   FILE *fpr;
   int rank;
   char buf[256];
   vector<double> tmpx, tmpy, tmpz;
   vector<PetscInt> idx;

   // Get MPI rank 
   MPI_Comm_rank(PETSC_COMM_WORLD,&rank);
   
   // Rank-0 process will read in header information and bcast npart
   PetscPrintf (PETSC_COMM_WORLD, "Opening file...\n");
   if (rank==0)
   {
      // Can we open the file?
      if ((fpr = fopen(infilename,"r")) == NULL)
         RAISE_ERR(99, "Failed opening file...");
      npart = 0;
   }

   // Read the file.
   PetscPrintf (PETSC_COMM_WORLD, "Reading from file...\n");
   if (rank==0) for(;;)
   {
      fgets (buf, 255, fpr);
      if (feof(fpr)) break;
      if (buf[0] == '#') continue;

      double x, y, z;
      sscanf (buf, "%lf %lf %lf", &x, &y, &z);
      tmpx.push_back(x); tmpy.push_back(y); tmpz.push_back(z); idx.push_back(npart); npart++;
   }

   // Number of particles = however many were read in.
   MPI_Bcast(&npart, 1, MPI_INT, 0, PETSC_COMM_WORLD);
   
   // Allocate vectors
   PetscPrintf (PETSC_COMM_WORLD, "Allocating vectors...\n");
   VecCreateMPI(PETSC_COMM_WORLD,PETSC_DECIDE,npart,&px);
   VecDuplicate(px, &py); 
   VecDuplicate(px, &pz); 
   VecDuplicate(px, &pw);

   // Fill the vectors by looping through the lists.
   PetscPrintf (PETSC_COMM_WORLD, "Filling vectors...\n");
   if (rank==0)
   {
      VecSetValues(px, npart, &idx[0], &tmpx[0], INSERT_VALUES);
      VecSetValues(py, npart, &idx[0], &tmpy[0], INSERT_VALUES);
      VecSetValues(pz, npart, &idx[0], &tmpz[0], INSERT_VALUES);
   }

   // Sync up the vectors in the different processes.
   PetscPrintf (PETSC_COMM_WORLD, "Assembling vectors...\n");
   VecAssemblyBegin(px); VecAssemblyEnd(px);
   VecAssemblyBegin(py); VecAssemblyEnd(py);
   VecAssemblyBegin(pz); VecAssemblyEnd(pz);
   
   // This is for unweighted files
   VecSet(pw, 1.0);


   // Close the files, de-allocate the memory and shut down.
   PetscPrintf (PETSC_COMM_WORLD, "Closing file & finishing.\n");
   if (rank==0 && fpr != NULL) fclose(fpr);
   return 0;
}

int Particle::AsciiWriteSerial(const char *outfilename)
// ---------------------------------------------------------------------------
// RS 2010/06/29:  Prints out particles *to* an ASCII file.  We need this
// to bridge the gap between the particle-displacement step (done w/MPI+KSP)
// and the correlation function calculation (done w/pair-counting).
// NP 2011/08/18: Updated the code to have each processor directly write. This 
// avoids potential memory overruns with large files.
// ---------------------------------------------------------------------------
{  
   // Get rank
   int rank, size;
   MPI_Comm_rank(PETSC_COMM_WORLD, &rank);
   MPI_Comm_size(PETSC_COMM_WORLD, &size);
	
   // Print out 
   double *_px, *_py, *_pz, *_pw;
   PetscInt lo, hi;
   FILE *fp; 
  

   VecGetOwnershipRange(px, &lo, &hi);
   VecGetArray(px, &_px); VecGetArray(py, &_py); VecGetArray(pz, &_pz); VecGetArray(pw, &_pw);
   for (int iproc = 0; iproc < size; ++iproc) {
     if (iproc == rank) {

       /* The rank 0 process should open a new file. Everyone else 
        * should append to the file. The first step is not essential, but
        * you WILL get into trouble, if you are not careful */
       if (rank == 0) {
        fp = fopen(outfilename, "w");
       } else {
        fp = fopen(outfilename, "a");
       }
       if (!fp) RAISE_ERR(99, "Unable to open file....\n");
       for (int ii = lo; ii < hi; ++ii) fprintf(fp, "%20.10e %20.10e %20.10e\n",
                _px[ii-lo],_py[ii-lo],_pz[ii-lo]);
       fflush(fp);
       fclose(fp);
     }
     MPI_Barrier(PETSC_COMM_WORLD);
   }
   VecRestoreArray(px,&_px); VecRestoreArray(py,&_py); VecRestoreArray(pz,&_pz); VecRestoreArray(pw, &_pw);
}


int Particle::AsciiWriteWeightedSerial(const char *outfilename)
// ---------------------------------------------------------------------------
// RS 2010/06/29:  Prints out particles *to* an ASCII file.  We need this
// to bridge the gap between the particle-displacement step (done w/MPI+KSP)
// and the correlation function calculation (done w/pair-counting).
// NP 2011/08/18: Updated the code to have each processor directly write. This 
// avoids potential memory overruns with large files.
// ---------------------------------------------------------------------------
{  
   // Get rank
   int rank, size;
   MPI_Comm_rank(PETSC_COMM_WORLD, &rank);
   MPI_Comm_size(PETSC_COMM_WORLD, &size);
	
   // Print out 
   double *_px, *_py, *_pz, *_pw;
   PetscInt lo, hi;
   FILE *fp; 
  

   VecGetOwnershipRange(px, &lo, &hi);
   VecGetArray(px, &_px); VecGetArray(py, &_py); VecGetArray(pz, &_pz); VecGetArray(pw, &_pw);
   for (int iproc = 0; iproc < size; ++iproc) {
     if (iproc == rank) {
       /* The rank 0 process should open a new file. Everyone else 
        * should append to the file. The first step is not essential, but
        * you WILL get into trouble, if you are not careful */
       if (rank == 0) {
        fp = fopen(outfilename, "w");
       } else {
        fp = fopen(outfilename, "a");
       }
       if (!fp) RAISE_ERR(99, "Unable to open file....\n");

       for (int ii = lo; ii < hi; ++ii) fprintf(fp, "%20.10e %20.10e %20.10e %20.10e\n",
                _px[ii-lo],_py[ii-lo],_pz[ii-lo], _pw[ii-lo]);
       fflush(fp);
       fclose(fp);
     }
     MPI_Barrier(PETSC_COMM_WORLD);
   }
   VecRestoreArray(px,&_px); VecRestoreArray(py,&_py); VecRestoreArray(pz,&_pz); VecRestoreArray(pw, &_pw);
}


void Particle::GridInitialize(int N, double dx, double dy, double dz, double L) {
  int rank, size, ix, iy, iz;
  PetscInt lo, hi, tmp;

  MPI_Comm_rank(PETSC_COMM_WORLD, &rank);
  npart = N*N*N;

  // Allocate vectors
  VecCreateMPI(PETSC_COMM_WORLD, PETSC_DECIDE,npart,&px);
  VecDuplicate(px, &py); 
  VecDuplicate(px, &pz);
  VecDuplicate(px, &pw);

  
  VecGetOwnershipRange(px, &lo, &hi);
  double cell = L/N;
  for (PetscInt ipart=lo; ipart < hi; ++ipart) {
    iz = ipart%N; tmp = (PetscInt) ipart/N;
    iy = tmp%N; ix = (PetscInt) tmp/N;
    VecSetValue(px, ipart, (ix + dx)*cell, INSERT_VALUES);
    VecSetValue(py, ipart, (iy + dy)*cell, INSERT_VALUES);
    VecSetValue(pz, ipart, (iz + dz)*cell, INSERT_VALUES);
  }

  // Synchronize every block
  VecAssemblyBegin(px); VecAssemblyEnd(px);
  VecAssemblyBegin(py); VecAssemblyEnd(py);
  VecAssemblyBegin(pz); VecAssemblyEnd(pz);
  VecSet(pw, 1.0); // This never needs weights...
}


void Particle::RandomInit(int _npart, double L, int seed) {
  int ipart, rank;
  PetscInt lo, hi;
  double *_pp;

  npart = _npart;

  // MPI rank
  MPI_Comm_rank(PETSC_COMM_WORLD,&rank);

  // Allocate vectors
  VecCreateMPI(PETSC_COMM_WORLD, PETSC_DECIDE,npart,&px);
  VecDuplicate(px, &py); 
  VecDuplicate(px, &pz);
  VecDuplicate(px, &pw); VecSet(pw, 1.0);

  // Sizes
  VecGetOwnershipRange(px, &lo, &hi);
  ipart = hi-lo;

  // Initialize the random number generator
  gsl_rng *ran = gsl_rng_alloc(gsl_rng_mt19937);
  gsl_rng_set(ran, seed + 100*rank);

  // Each dimension separately -- bulky because of the way we defined stuff
  VecGetArray(px, &_pp);
  for (int ii =0; ii < ipart; ++ii) _pp[ii] = periodic(gsl_rng_uniform(ran))*L;
  VecRestoreArray(px, &_pp);

  VecGetArray(py, &_pp);
  for (int ii =0; ii < ipart; ++ii) _pp[ii] = periodic(gsl_rng_uniform(ran))*L;
  VecRestoreArray(py, &_pp);

  VecGetArray(pz, &_pp);
  for (int ii =0; ii < ipart; ++ii) _pp[ii] = periodic(gsl_rng_uniform(ran))*L;
  VecRestoreArray(pz, &_pp);

  // Free the random number generator
  gsl_rng_free(ran);
}

// RS 2010/06/22:  commented below, as I found useful.
void Particle::SlabDecompose(const DensityGrid& g) {
  int rank, size, proc, Ngproc, Ng, rtot;
  vector<int> narr, narr1;
  vector<PetscInt> idx,icount, icount_check;
  PetscInt nlocal, lo, hi;
  double *_px;
  double L;

  Ng = (int) g.Ng;
  L = g.L;

  MPI_Comm_rank(PETSC_COMM_WORLD, &rank);
  // RS:  size = number of processors/processes being used.
  MPI_Comm_size(PETSC_COMM_WORLD, &size);
  narr.resize(size); icount.resize(size); narr1.resize(size*size); icount_check.resize(size);
  Ngproc = Ng/size;
  for (int ii=0; ii < size; ++ii) narr[ii] = 0;

  // Get the local range
  VecGetOwnershipRange(px, &lo, &hi);
  nlocal = hi-lo;
  VecGetArray(px, &_px);
  for (PetscInt ii=0; ii < nlocal; ++ii) {
    // RS:  compute which process this particle belongs to
    proc = ((int) (periodic(_px[ii]/L)*Ng))/Ngproc;
    // RS:  narr = the number of particles being handled by this process
    narr[proc]++;
  }
  VecRestoreArray(px, &_px);
  MPI_Allgather(&narr[0], size, MPI_INT, &narr1[0], size, MPI_INT, PETSC_COMM_WORLD);

  // Sum
  // RS:  not yet sure what's going on here
  for (int ii=0; ii < size; ++ii) {narr[ii] = 0; icount[ii] = 0;}
  for (int ii=0; ii < size; ++ii) { 
    for (int jj=0; jj < size; ++jj) 
      narr[ii] += narr1[ii+jj*size];
    for (int jj=0; jj < rank;++jj) 
      icount[ii] += narr1[ii+jj*size];
  }
  rtot = 0;
  for (int ii=0; ii < size; ++ii) { 
    icount[ii] += rtot; 
    rtot += narr[ii];
    icount_check[ii] = icount[ii] + narr1[ii+rank*size];
  }
  if (rtot != npart) RAISE_ERR(99, "Assertion failed : rtot != npart");

  // USEFUL DEBUGGING
  //  for (int ii=0; ii < size; ++ii) {
  //    PetscSynchronizedPrintf(PETSC_COMM_WORLD,"%i %i %llu %llu\n",rank*size+ii,narr[ii], icount[ii], icount_check[ii]);
  //  }
  //  PetscSynchronizedFlush(PETSC_COMM_WORLD);

  
  // Set up a scatter set
  // RS:  This has something to do with how all the grid locations are
  // indexed within the array.  The mappings between global indices and
  // processor-local indices are called index sets or scatter sets.
  IS is1, is2;
  VecGetArray(px, &_px);
  // RS:  idx[] is a vector<int> we set up above.  Set each element of the
  // vector to correspond to the processor on which each particle lives.
  // (This correspondence would abstract well into a helper function,
  //  since a very similar block appears above.)
  for (PetscInt ii=0; ii < nlocal; ++ii) {
    proc = ((int) (periodic(_px[ii]/L)*Ng))/Ngproc;
    idx.push_back(icount[proc]);
    icount[proc]++;
  }

  // Assertion
  for (int ii=0; ii < size; ++ii) 
    if (icount[ii] != icount_check[ii]) RAISE_ERR(99,"Assertion failed : icount != icount_check");

  VecRestoreArray(px, &_px);
  ISCreateStride(PETSC_COMM_WORLD, nlocal, lo, 1, &is1);
  ISCreateGeneralWithArray(PETSC_COMM_WORLD, nlocal, &idx[0], &is2);


  // RS:  Rearrange the particles amongst the processes according to our
  // scheme, i.e., so that each processor has all particles in a slab dx.
  // Create a new temporary vector to scatter to 
  Vec tmp;
  VecCreateMPI(narr[rank], PETSC_DETERMINE, &tmp);
  VecScatter vs = VecScatterCreate(px, is1, tmp, is2);
  // Scatter px
  VecScatterBegin(vs, px, tmp, INSERT_VALUES, SCATTER_FORWARD);
  VecScatterEnd(vs, px, tmp, INSERT_VALUES, SCATTER_FORWARD);
  VecDestroy(px); VecDuplicate(tmp, &px); VecCopy(tmp, px); 
  // py
  VecScatterBegin(vs, py, tmp, INSERT_VALUES, SCATTER_FORWARD);
  VecScatterEnd(vs, py, tmp, INSERT_VALUES, SCATTER_FORWARD);
  VecDestroy(py); VecDuplicate(tmp, &py); VecCopy(tmp, py); 
  // pz
  VecScatterBegin(vs, pz, tmp, INSERT_VALUES, SCATTER_FORWARD);
  VecScatterEnd(vs, pz, tmp, INSERT_VALUES, SCATTER_FORWARD);
  VecDestroy(pz); VecDuplicate(tmp, &pz); VecCopy(tmp, pz); 
  // pw
  VecScatterBegin(vs, pw, tmp, INSERT_VALUES, SCATTER_FORWARD);
  VecScatterEnd(vs, pw, tmp, INSERT_VALUES, SCATTER_FORWARD);
  VecDestroy(pw); VecDuplicate(tmp, &pw); VecCopy(tmp, pw); 

  // Clean up
  VecDestroy(tmp);
  ISDestroy(is1);
  ISDestroy(is2);

}



int Particle::CountMask(Mask3D& mask) {
  PetscInt lo, hi, np;
  double *_px, *_py, *_pz;
  int lc=0, gc=0; // local and global counts
  
  VecGetOwnershipRange(px, &lo, &hi);
  np = hi-lo;
  
  VecGetArray(px, &_px);
  VecGetArray(py, &_py);
  VecGetArray(pz, &_pz);
  for (int ii=0; ii < np; ++ii) lc +=  mask(_px[ii], _py[ii], _pz[ii]);
  VecRestoreArray(px, &_px);
  VecRestoreArray(py, &_py);
  VecRestoreArray(pz, &_pz);

  MPI_Allreduce(&lc, &gc, 1, MPI_INT, MPI_SUM, PETSC_COMM_WORLD);
  return gc;
}


void Particle::TrimMask(Mask3D& mask)
// RS 2010/07/06:  Commented!
{
  PetscInt lo, hi, np;
  double *_px, *_py, *_pz, *_pw;
  double *_px1, *_py1, *_pz1, *_pw1;
  int lc=0, lc1=0;
  Vec px1, py1, pz1, pw1;

   
  VecGetOwnershipRange(px, &lo, &hi);
  np = hi-lo;
  
  // Count the number of local entries
  VecGetArray(px, &_px);
  VecGetArray(py, &_py);
  VecGetArray(pz, &_pz);
  VecGetArray(pw, &_pw);
  // RS:  Specifically, look at the position of each particle and see
  // whether the mask is 1 at that position.  If so, count it as "in".
  for (int ii=0; ii < np; ++ii) lc +=  mask(_px[ii], _py[ii], _pz[ii]);

  // Allocate new vectors
  // RS:  ...to hold the number of particles we counted above as being
  // "inside" the mask.
  VecCreateMPI(PETSC_COMM_WORLD, lc, PETSC_DETERMINE, &px1);
  VecDuplicate(px1, &py1);
  VecDuplicate(px1, &pz1);
  VecDuplicate(px1, &pw1);

  // Get ready
  VecGetArray(px1, &_px1);
  VecGetArray(py1, &_py1);
  VecGetArray(pz1, &_pz1);
  VecGetArray(pw1, &_pw1);

  // RS:  Now just copy the particle data from the old arrays to the new ones
  // for each particle not cut out by the mask.
  for (int ii=0; ii < np; ++ii) 
    if (mask(_px[ii], _py[ii], _pz[ii])) {
      _px1[lc1] = _px[ii]; _py1[lc1] = _py[ii]; _pz1[lc1] = _pz[ii]; _pw1[lc1] = _pw[ii];
      lc1++;
    }

  // Assertion check
  if (lc1 != lc) RAISE_ERR(99, "lc1 != lc in TrimMask");

  
  // We will restore here
  VecRestoreArray(px, &_px);
  VecRestoreArray(py, &_py);
  VecRestoreArray(pz, &_pz);
  VecRestoreArray(pw, &_pw);
  VecRestoreArray(px1, &_px1);
  VecRestoreArray(py1, &_py1);
  VecRestoreArray(pz1, &_pz1);
  VecRestoreArray(pw1, &_pw1);

  // And now we swap
  // RS:  Overwrite the class member copies of the particle position vectors
  // with the local ones we just constructed.
  VecDestroy(px); VecDuplicate(px1, &px); VecCopy(px1, px);
  VecDestroy(py); VecDuplicate(py1, &py); VecCopy(py1, py);
  VecDestroy(pz); VecDuplicate(pz1, &pz); VecCopy(pz1, pz);
  VecDestroy(pw); VecDuplicate(pw1, &pw); VecCopy(pw1, pw);

  // Clean up
  VecDestroy(px1);
  VecDestroy(py1);
  VecDestroy(pz1);
  VecDestroy(pw1);

  // Set npart
  VecGetSize(px, &np);
  npart = np;
}


void Particle::RadialDisplace(Vec qx, Vec qy, Vec qz, vector<double> &origin, double factor) 
{
  // Local arrays
  double *_px, *_py, *_pz, *_qx, *_qy, *_qz;
  PetscInt lo, hi, np;
  double rr, x, y, z;

  // Get the ranges of objects
  VecGetOwnershipRange(px, &lo, &hi);
  np = hi-lo;

  // Localize arrays
  VecGetArray(px, &_px);
  VecGetArray(py, &_py);
  VecGetArray(pz, &_pz);
  VecGetArray(qx, &_qx);
  VecGetArray(qy, &_qy);
  VecGetArray(qz, &_qz);


  for (int ii=0; ii< np; ++ii) {
    /* This is ugly */
    x = _px[ii] - origin[0];
    y = _py[ii] - origin[1];
    z = _pz[ii] - origin[2];
    rr = sqrt(x*x + y*y + z*z) + 1.e-10; // For the case that a particle exactly is on the origin

    /* Now shift */
    _px[ii] += (factor*_qx[ii]*x)/rr;
    _py[ii] += (factor*_qy[ii]*y)/rr;
    _pz[ii] += (factor*_qz[ii]*z)/rr;
  }


  // Delocalize arrays
  VecRestoreArray(px, &_px);
  VecRestoreArray(py, &_py);
  VecRestoreArray(pz, &_pz);
  VecRestoreArray(qx, &_qx);
  VecRestoreArray(qy, &_qy);
  VecRestoreArray(qz, &_qz);
}
