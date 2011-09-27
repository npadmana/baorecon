#include "Recon.h"

PotentialSolve::PotentialSolve(int N, double _L, int _maxit, double _rtol, double _atol) {
  Ng = N;
  L = _L;
  dx = L/Ng;
  maxit = _maxit;
  rtol = _rtol;
  atol = _atol;
  _dg1.Init(N, L);
  a = PETSC_NULL;
  mnull = PETSC_NULL;
  solver = PETSC_NULL;
}

PotentialSolve::~PotentialSolve() {
  if (a != PETSC_NULL) MatDestroy(a); 
  if (mnull != PETSC_NULL) MatNullSpaceDestroy(mnull);
  if (solver != PETSC_NULL) KSPDestroy(solver);
}

void PotentialSolve::_BuildCARTPBC(double fval) {
  PetscInt local, global, pos1, pos2, Ng2;
  int lo, hi, ixp, ixm, iyp, iym, izp, izm;

  // Collect rows and push once at a time 
  PetscInt idxm[1], idxn[7];
  PetscScalar val[7];

  // Get sizes
  _dg1.size(local, global);
  // Allocate maximum values of diagonal and off-diagonal memory
  // This is a little wasteful, but it simplifies the amount you need to think
  MatCreateMPIAIJ(PETSC_COMM_WORLD, local, local, global, global,7, PETSC_NULL, 2, PETSC_NULL, &a);
  _dg1.slab(lo, hi);
  Ng2 = Ng+2;

  // Now fill in the matrix -- do this in the maximally dumb way. Easier to debug.
  for (int ix=lo; ix < hi; ++ix) {
    ixp = (ix+1)%Ng; ixm = (ix-1+Ng)%Ng;
    for (int iy=0; iy < Ng; ++iy) {
      iyp = (iy+1)%Ng; iym = (iy-1+Ng)%Ng;
      for (int iz=0; iz < Ng; ++iz) { 
        izp = (iz+1)%Ng; izm = (iz-1+Ng)%Ng;
        pos1 = (ix*Ng+iy)*Ng2 + iz ; idxn[0]=pos1; val[0] = (-6.0-2.0*fval)/(dx*dx);
        pos2 = (ixp*Ng+iy)*Ng2 + iz; idxn[1]=pos2; val[1] = 1.0/(dx*dx);
        pos2 = (ixm*Ng+iy)*Ng2 + iz; idxn[2]=pos2; val[2] = 1.0/(dx*dx);
        pos2 = (ix*Ng+iyp)*Ng2 + iz; idxn[3]=pos2; val[3] = 1.0/(dx*dx);
        pos2 = (ix*Ng+iym)*Ng2 + iz; idxn[4]=pos2; val[4] = 1.0/(dx*dx);
        pos2 = (ix*Ng+iy)*Ng2 + izp; idxn[5]=pos2; val[5] = (1.0+fval)/(dx*dx);
        pos2 = (ix*Ng+iy)*Ng2 + izm; idxn[6]=pos2; val[6] = (1.0+fval)/(dx*dx);
        idxm[0] = pos1;
        MatSetValues(a, 1, idxm, 7, idxn, val, INSERT_VALUES);
      }
      // Special case the padded region to ensure that the const function is still in the NULL space
      pos1 = (ix*Ng+iy)*Ng2 + Ng; pos2 = pos1+1;
      MatSetValue(a, pos1, pos1, 1.0/(dx*dx), INSERT_VALUES);
      MatSetValue(a, pos2, pos2, 1.0/(dx*dx), INSERT_VALUES);
      MatSetValue(a, pos1, pos2, -1.0/(dx*dx), INSERT_VALUES);
      MatSetValue(a, pos2, pos1, -1.0/(dx*dx), INSERT_VALUES);
    }
  }
  MatAssemblyBegin(a,MAT_FINAL_ASSEMBLY);
  MatAssemblyEnd(a,MAT_FINAL_ASSEMBLY);
}


void PotentialSolve::_BuildRADIAL(double fval, vector<double>& origin) {
  PetscInt local, global, pos1, pos2, Ng2;
  int lo, hi, ixp, ixm, iyp, iym, izp, izm;

  double x, y, z, rr2, rsd, dx2;
  dx2 = dx*dx;
  // Collect rows and push once at a time 
  PetscInt idxm[1], idxn[19];
  PetscScalar val[19];

  // Get sizes
  _dg1.size(local, global);
  MatCreateMPIAIJ(PETSC_COMM_WORLD, local, local, global, global,19, PETSC_NULL, 10, PETSC_NULL, &a);
  _dg1.slab(lo, hi);
  Ng2 = Ng+2;

  // Now fill in the matrix -- do this in the maximally dumb way. Easier to debug.
  for (int ix=lo; ix < hi; ++ix) {
    ixp = (ix+1)%Ng; ixm = (ix-1+Ng)%Ng;
    for (int iy=0; iy < Ng; ++iy) {
      iyp = (iy+1)%Ng; iym = (iy-1+Ng)%Ng;
      for (int iz=0; iz < Ng; ++iz) { 
        izp = (iz+1)%Ng; izm = (iz-1+Ng)%Ng;

        // Start by doing the coordinate geometry and setting xx,yy,zz
        x = double(ix)*dx - origin[0];
        y = double(iy)*dx - origin[1];
        z = double(iz)*dx - origin[2];
        rr2 = x*x + y*y + z*z + 1.e-5; // If the origin coincides with a grid cell


        // Now set the various matrix terms. There are 19 terms, and some organizing principles.
        // See python/spherical_v2.py and python/spherical_v2.tex for the redshift space terms
        // As always, do the maximally dumb way, for ease of debugging

        // (0,0,0) - unchanged from the normal case
        pos1 = (ix*Ng+iy)*Ng2 + iz;  idxm[0] = pos1; idxn[0] = pos1; val[0] = (-6.0-2.0*fval)/(dx*dx);

        // (1,0,0)
        rsd = (x*x)/(dx2*rr2) + x/(dx*rr2);
        idxn[1] = (ixp*Ng+iy)*Ng2 + iz; val[1]=1.0/(dx2) + fval*rsd;
        // (-1,0,0)
        rsd = (x*x)/(dx2*rr2) - x/(dx*rr2);
        idxn[2] = (ixm*Ng+iy)*Ng2 + iz; val[2]=1.0/(dx2) + fval*rsd;

        // (0,1,0)
        rsd = (y*y)/(dx2*rr2) + y/(dx*rr2);
        idxn[3]= (ix*Ng+iyp)*Ng2 + iz; val[3]=1.0/(dx2) + fval*rsd;
        // (0,-1,0)
        rsd = (y*y)/(dx2*rr2) - y/(dx*rr2);
        idxn[4]= (ix*Ng+iym)*Ng2 + iz; val[4]=1.0/(dx2) + fval*rsd;

        // (0,0,1)
        rsd = (z*z)/(dx2*rr2) + z/(dx*rr2);
        idxn[5] = (ix*Ng+iy)*Ng2 + izp; val[5]=1.0/(dx2) + fval*rsd;
        // (0,0,-1)
        rsd = (z*z)/(dx2*rr2) - z/(dx*rr2);
        idxn[6] = (ix*Ng+iy)*Ng2 + izm; val[6]=1.0/(dx2) + fval*rsd;

        // (-1,-1,0), (1,1,0), (-1,1,0), (1,-1,0)
        rsd = (x*y)/(2*dx2*rr2);
        idxn[ 7] = (ixm*Ng+iym)*Ng2 + iz; val[ 7] =  fval*rsd;
        idxn[ 8] = (ixp*Ng+iyp)*Ng2 + iz; val[ 8] =  fval*rsd;
        idxn[ 9] = (ixm*Ng+iyp)*Ng2 + iz; val[ 9] = -fval*rsd;
        idxn[10] = (ixp*Ng+iym)*Ng2 + iz; val[10] = -fval*rsd;

        // (-1,0,-1), (1,0,1), (-1,0,1), (1,0,-1)
        rsd = (x*z)/(2*dx2*rr2);
        idxn[11] = (ixm*Ng+iy)*Ng2 + izm; val[11] =  fval*rsd;
        idxn[12] = (ixp*Ng+iy)*Ng2 + izp; val[12] =  fval*rsd;
        idxn[13] = (ixm*Ng+iy)*Ng2 + izp; val[13] = -fval*rsd;
        idxn[14] = (ixp*Ng+iy)*Ng2 + izm; val[14] = -fval*rsd;

        // (0,-1,-1), (0,1,1), (0,-1,1), (0,1,-1)
        rsd = (y*z)/(2*dx2*rr2);
        idxn[15] = (ix*Ng+iym)*Ng2 + izm; val[15] =  fval*rsd;
        idxn[16] = (ix*Ng+iyp)*Ng2 + izp; val[16] =  fval*rsd;
        idxn[17] = (ix*Ng+iym)*Ng2 + izp; val[17] = -fval*rsd;
        idxn[18] = (ix*Ng+iyp)*Ng2 + izm; val[18] = -fval*rsd;
        MatSetValues(a, 1, idxm, 19, idxn, val, INSERT_VALUES);
      }
      // Special case the padded region to ensure that the const function is still in the NULL space
      pos1 = (ix*Ng+iy)*Ng2 + Ng; pos2 = pos1+1;
      MatSetValue(a, pos1, pos1, 1.0/(dx*dx), INSERT_VALUES);
      MatSetValue(a, pos2, pos2, 1.0/(dx*dx), INSERT_VALUES);
      MatSetValue(a, pos1, pos2, -1.0/(dx*dx), INSERT_VALUES);
      MatSetValue(a, pos2, pos1, -1.0/(dx*dx), INSERT_VALUES);
    }
  }
  MatAssemblyBegin(a,MAT_FINAL_ASSEMBLY);
  MatAssemblyEnd(a,MAT_FINAL_ASSEMBLY);
}


void PotentialSolve::SetupOperator(PotentialOp optype, double fval, vector<double> origin) {

  //Set the matrix
  switch (optype) {
    case REALPBC :
      _BuildCARTPBC();
      MatNullSpaceCreate(PETSC_COMM_WORLD, PETSC_TRUE, 0, PETSC_NULL, &mnull);
      break;
    case CARTPBC :
      _BuildCARTPBC(fval);
      MatNullSpaceCreate(PETSC_COMM_WORLD, PETSC_TRUE, 0, PETSC_NULL, &mnull);
      break;
    case RADIAL :
      _BuildRADIAL(fval, origin);
      MatNullSpaceCreate(PETSC_COMM_WORLD, PETSC_TRUE, 0, PETSC_NULL, &mnull);
      break;
    default :
      RAISE_ERR(99, "Unknown operator type");
  }

  // Sanity check
  PetscTruth isNull;
  MatNullSpaceTest(mnull, a, &isNull);
  if (isNull != PETSC_TRUE) {
     PetscPrintf(PETSC_COMM_WORLD, "Warning -- Null space is not truly a null space!\n");
  }
  PetscPrintf(PETSC_COMM_WORLD, "Completed operator initialization\n");

  // Initialize the solver
  KSPCreate(PETSC_COMM_WORLD,&solver);
  KSPSetOperators(solver, a, a,SAME_PRECONDITIONER);
  KSPSetTolerances(solver, rtol, atol, PETSC_DEFAULT, maxit);
  
  // Set some defaults 
  KSPSetType(solver,KSPLGMRES);
  KSPGMRESSetRestart(solver, 10);
  // Possible for user to override
  KSPSetFromOptions(solver);
  // Set up the null space
  KSPSetNullSpace(solver, mnull);
  PetscPrintf(PETSC_COMM_WORLD, "Solver setup\n");

}

bool PotentialSolve::Solve(Vec delta, Vec pot, double bias) {
  PetscInt its;
  KSPConvergedReason reason;
  bool retval;
  
  // Delta to -Delta
  VecScale(delta, -1.0/bias);

  // Actually solve
  KSPSolve(solver, delta, pot);
  KSPGetConvergedReason(solver,&reason);
  if (reason<0) {
    PetscPrintf(PETSC_COMM_WORLD,"Diverged : %d.\n",reason);
    retval=false;
  } else {
    KSPGetIterationNumber(solver,&its);
    PetscPrintf(PETSC_COMM_WORLD,"\nConvergence in %d iterations.\n",(int)its);
    retval=true;
  }


  // Remove any values that might have crept in here
  _dg1.ZeroPad(pot);
  
  // Clean up
  VecScale(delta, -1.0*bias);

  return retval;
}
