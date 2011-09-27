#include <iostream>
#include <cmath>
#include <sstream>
#include <iomanip>

#include "Recon.h"

using namespace std;
static char help[] = "A test program\n";

int main(int argc, char *args[]) {
    PetscErrorCode ierr;
    ierr=PetscInitialize(&argc,&args,(char *) 0, help); CHKERRQ(ierr);
    PBCParams pars(string("Az_template.xml"), "PBCreal"); 

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
      hdr << "# Origin =" << pars.recon.origin[0] << " " << pars.recon.origin[1] << " " << pars.recon.origin[2] << endl;

      cout << hdr.str() << endl;
    }
    // Only call this when everything is out of scope
    ierr=PetscFinalize(); CHKERRQ(ierr);

    return 0;
}
 
