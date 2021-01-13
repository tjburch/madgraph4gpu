//==========================================================================
// This file has been automatically generated for C++ Standalone by
// MadGraph5_aMC@NLO v. 2.8.2, 2020-10-30
// By the MadGraph5_aMC@NLO Development Team
// Visit launchpad.net/madgraph5 and amcatnlo.web.cern.ch
//==========================================================================

#ifdef CL_SYCL_LANGUAGE_VERSION
#include <CL/sycl.hpp>
#endif
#include "../../src/HelAmps_sm.h"

#ifndef MG5_Sigma_sm_gg_ttxgg_H
#define MG5_Sigma_sm_gg_ttxgg_H

#include <complex>
#include <vector>
#include <cassert>
#include <iostream>

#include "mgOnGpuConfig.h"
#include "mgOnGpuTypes.h"


#include "Parameters_sm.h"

//--------------------------------------------------------------------------

#ifdef CL_SYCL_LANGUAGE_VERSION

#define checkCuda(code) {assertCuda(code, __FILE__, __LINE__);}

inline void assertCuda(int code, const char *file, int line, bool abort = true)
{
}

#endif
//--------------------------------------------------------------------------

#ifdef CL_SYCL_LANGUAGE_VERSION
namespace gProc
#else
namespace Proc
#endif
{

//==========================================================================
// A class for calculating the matrix elements for
// Process: g g > t t~ g g WEIGHTED<=4 @1
//--------------------------------------------------------------------------

class CPPProcess
{
  public:

    CPPProcess(int numiterations, int gpublocks, int gputhreads, 
    bool verbose = false, bool debug = false); 

    ~CPPProcess(); 

    // Initialize process.
    virtual void initProc(std::string param_card_name); 


    virtual int code() const {return 1;}

    const std::vector<fptype> &getMasses() const; 

    void setInitial(int inid1, int inid2) 
    {
      id1 = inid1; 
      id2 = inid2; 
    }

    int getDim() const {return dim;}

    int getNIOParticles() const {return nexternal;}


    // Constants for array limits
    static const int ninitial = mgOnGpu::npari; 
    static const int nexternal = mgOnGpu::npar; 
    // static const int nprocesses = 1;

  private:
    int m_numiterations; 
    // gpu variables
    int gpu_nblocks; 
    int gpu_nthreads; 
    int dim;  // gpu_nblocks * gpu_nthreads;

    // print verbose info
    bool m_verbose; 

    // print debug info
    bool m_debug; 

    static const int nwavefuncs = 6; 
    static const int namplitudes = 159; 
    static const int ncomb = 64; 
    static const int wrows = 63; 
    // static const int nioparticles = 6;

    cxtype** amp; 


    // Pointer to the model parameters
    Parameters_sm * pars; 

    // vector with external particle masses
    std::vector<fptype> mME; 

    // Initial particle ids
    int id1, id2; 

}; 



//--------------------------------------------------------------------------
#ifdef CL_SYCL_LANGUAGE_VERSION

void sigmaKin_getGoodHel(const fptype * allmomenta,  // input: momenta as AOSOA[npagM][npar][4][neppM] with nevt=npagM*neppM
bool * isGoodHel,
sycl::nd_item<3> item_ct1,
sycl::accessor<int, 2, sycl::access::mode::read_write> cHel,
fptype *cIPC,
fptype *cIPD);  // output: isGoodHel[ncomb] - device array
#endif

//--------------------------------------------------------------------------

#ifdef CL_SYCL_LANGUAGE_VERSION
 void sigmaKin_setGoodHel(const bool * isGoodHel, int* cNGoodHel, int* cGoodHel);  // input: isGoodHel[ncomb] - host array
#endif

//--------------------------------------------------------------------------

 void sigmaKin(
    const fptype *allmomenta, // input: momenta as AOSOA[npagM][npar][4][neppM]
                              // with nevt=npagM*neppM
    fptype *allMEs, sycl::nd_item<3> item_ct1,
    sycl::accessor<int, 2, sycl::access::mode::read_write> cHel,
    int *cNGoodHel,
    int *cGoodHel // output: allMEs[nevt], final |M|^2 averaged over all
                  // helicities
#ifndef CL_SYCL_LANGUAGE_VERSION
    ,
    const int
        nevt // input: #events (for cuda: nevt == ndim == gpublocks*gputhreads)
#endif
); 

//--------------------------------------------------------------------------
}

#endif // MG5_Sigma_sm_gg_ttxgg_H
