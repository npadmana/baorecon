#include "Recon.h"

using namespace std;

Params::Params(string configfn) {

  try {
    ticpp::Document doc(configfn);
    doc.LoadFile();

    ticpp::Element *root = doc.FirstChildElement();
    
    // Now read in all the parameters
    root->FirstChildElement("LBox")->GetText(&Lbox);
    root->FirstChildElement("Ngrid")->GetText(&Ngrid);
    // Recon
    root->FirstChildElement("ReconParams")->FirstChildElement("bias")->GetText(&recon.bias);
    root->FirstChildElement("ReconParams")->FirstChildElement("smooth")->GetText(&recon.smooth);
    root->FirstChildElement("ReconParams")->FirstChildElement("NRandomFac")->GetText(&recon.nrandomfac);

    // The fields below are optional
    ticpp::Element *elt;
    recon.beta = 0.0;
    elt = root->FirstChildElement("ReconParams")->FirstChildElement("beta", false);
    if (elt) elt->GetTextOrDefault(&recon.beta, 0.0);
    recon.fval = recon.beta * recon.bias;
    recon.planeparallel = 0;
    elt = root->FirstChildElement("ReconParams")->FirstChildElement("planeparallel");
    if (elt) elt->GetTextOrDefault(&recon.planeparallel, 0);
    

    // Tolerances, max iterations
    elt = root->FirstChildElement("ReconParams")->FirstChildElement("MaxIterations", false);
    if (elt) elt->GetTextOrDefault(&recon.maxit, DEFAULT_MAXIT);
    elt = root->FirstChildElement("ReconParams")->FirstChildElement("rtol", false);
    if (elt) elt->GetTextOrDefault(&recon.rtol, DEFAULT_RTOL);
    elt = root->FirstChildElement("ReconParams")->FirstChildElement("atol", false);
    if (elt) elt->GetTextOrDefault(&recon.atol, DEFAULT_ATOL);

    // Origin
    double val, scale;
    recon.origin = vector<double>(3,0.0);
    // X
    elt = root->FirstChildElement("ReconParams")->FirstChildElement("originx",false);
    if (elt) {
        elt->GetAttributeOrDefault("scale",&scale,1.0);
        elt->GetTextOrDefault(&val,0.0);
        recon.origin[0] = val*scale;
    }
    // Y
    elt = root->FirstChildElement("ReconParams")->FirstChildElement("originy",false);
    if (elt) {
        elt->GetAttributeOrDefault("scale",&scale,1.0);
        elt->GetTextOrDefault(&val,0.0);
        recon.origin[1] = val*scale;
    }
    // Z
    elt = root->FirstChildElement("ReconParams")->FirstChildElement("originz",false);
    if (elt) {
        elt->GetAttributeOrDefault("scale",&scale,1.0);
        elt->GetTextOrDefault(&val,0.0);
        recon.origin[2] = val*scale;
    }


  } 
  catch( ticpp::Exception& ex )
  {
    cout << "XML error :" << ex.what() << endl;
    RAISE_ERR(99, "An error occurred reading the parameter file");
  }
  

}


PBCParams::PBCParams(string configfn, string fntype): Params(configfn) {

  try {
    ticpp::Document doc(configfn);
    doc.LoadFile();

    ticpp::Element *root = doc.FirstChildElement();
    
    // Xi Params
    root->FirstChildElement("XiParams")->FirstChildElement("rmin")->GetText(&xi.rmin);
    root->FirstChildElement("XiParams")->FirstChildElement("dr")->GetText(&xi.dr);
    root->FirstChildElement("XiParams")->FirstChildElement("smooth")->GetText(&xi.smooth);
    root->FirstChildElement("XiParams")->FirstChildElement("Nbins")->GetText(&xi.Nbins);

    // Now read in the file list
    fn fn1;
    ticpp::Iterator< ticpp::Element > iter(fntype);
    for (iter = iter.begin(root); iter != iter.end(); ++iter) {
      fn1.in = iter->FirstChildElement("input")->GetText();
      fn1.out = iter->FirstChildElement("output")->GetText();
      fnlist.push_back(fn1);
    }
  } 
  catch( ticpp::Exception& ex )
  {
    cout << "XML error :" << ex.what() << endl;
    RAISE_ERR(99, "An error occurred reading the parameter file");
  }
  

}

ShellParams::ShellParams(string configfn, string fntype) : PBCParams(configfn, fntype) {

  try {
    ticpp::Document doc(configfn);
    doc.LoadFile();

    ticpp::Element *root = doc.FirstChildElement();
    
    // Now read in all the parameters
    // Shell params
    //root->FirstChildElement("ShellParams")->FirstChildElement("xcen")->GetText(&shell.xcen);
    //root->FirstChildElement("ShellParams")->FirstChildElement("ycen")->GetText(&shell.ycen);
    //root->FirstChildElement("ShellParams")->FirstChildElement("zcen")->GetText(&shell.zcen);
    shell.xcen = shell.ycen = shell.zcen = 0.0; // The shell should always be at zero 
    root->FirstChildElement("ShellParams")->FirstChildElement("rmin")->GetText(&shell.rmin);
    root->FirstChildElement("ShellParams")->FirstChildElement("rmax")->GetText(&shell.rmax);
    root->FirstChildElement("ShellParams")->FirstChildElement("threshold")->GetText(&shell.thresh);
    root->FirstChildElement("ShellParams")->FirstChildElement("nover")->GetText(&shell.nover);
    // Pk prior  
    root->FirstChildElement("PkPrior")->FirstChildElement("bias")->GetText(&pkprior.bias);
    root->FirstChildElement("PkPrior")->FirstChildElement("noise")->GetText(&pkprior.noise);
    pkprior.fn = root->FirstChildElement("PkPrior")->FirstChildElement("fn")->GetText();

  } 
  catch( ticpp::Exception& ex )
  {
    cout << "XML error :" << ex.what() << endl;
    RAISE_ERR(99, "An error occurred reading the parameter file");
  }
  

}

LasDamasParams::LasDamasParams(string configfn, string fntype) : Params(configfn) {

  try {
    ticpp::Document doc(configfn);
    doc.LoadFile();

    ticpp::Element *root = doc.FirstChildElement();
    
    // Now read in all the parameters
    // Pk prior  
    root->FirstChildElement("PkPrior")->FirstChildElement("bias")->GetText(&pkprior.bias);
    root->FirstChildElement("PkPrior")->FirstChildElement("noise")->GetText(&pkprior.noise);
    pkprior.fn = root->FirstChildElement("PkPrior")->FirstChildElement("fn")->GetText();
    // This is optional 
    ticpp::Element *elt;
    pkprior.dorandom = 1;
    elt = root->FirstChildElement("PkPrior")->FirstChildElement("dorandom");
    if (elt) elt->GetTextOrDefault(&pkprior.dorandom, 0);


    // Las Damas parameters
    mask.fn = root->FirstChildElement("Mask")->FirstChildElement("randomfn")->GetText();
    root->FirstChildElement("Mask")->FirstChildElement("threshold")->GetText(&mask.thresh);

    // Now read in the file list
    fn fn1;
    ticpp::Iterator< ticpp::Element > iter(fntype);
    for (iter = iter.begin(root); iter != iter.end(); ++iter) {
      fn1.indata = iter->FirstChildElement("indata")->GetText();
      fn1.outdata = iter->FirstChildElement("outdata")->GetText();
      fn1.inrand = iter->FirstChildElement("inrand")->GetText();
      fn1.outrand = iter->FirstChildElement("outrand")->GetText();
      fnlist.push_back(fn1);
    }
  } 
  catch( ticpp::Exception& ex )
  {
    cout << "XML error :" << ex.what() << endl;
    RAISE_ERR(99, "An error occurred reading the parameter file");
  }
  

}
