//========================================================================================
// Athena++ problem generator: vertical slab at fixed R with stellar irradiation
// 2D Cartesian (X1 = z, X2 = R*phi (length units)).
// - X1: z in [0, zmax], reflecting at z=0, outflow at z=zmax
// - X2: phi-direction in R*phi units in [0, 2*pi*R], non-periodic (outflow) BCs
// - Radiation: user-set opacity; ray-tracing from zmax downwards; F = (r_*/R)^2 sigma T_*^4 exp(-tau/mu)
//========================================================================================

// C++ headers
#include <algorithm>
#include <cmath>
#include <cstring>
#include <iostream>

// Athena++ headers
#include "../athena.hpp"
#include "../athena_arrays.hpp"
#include "../bvals/bvals.hpp"
#include "../coordinates/coordinates.hpp"
#include "../eos/eos.hpp"
#include "../field/field.hpp"
#include "../globals.hpp"
#include "../hydro/hydro.hpp"
#include "../mesh/mesh.hpp"
#include "../parameter_input.hpp"
#include "../nr_radiation/radiation.hpp"
#include "../nr_radiation/integrators/rad_integrators.hpp"

namespace {
// parameters
Real rho0, zscale, r_star, t_star, kappa_star, kappa_a;
Real rhounit, lunit, tunit, tfloor, tceiling;
Real grazing_mu; // grazing angle (cosine)
Real gm0, p0_over_r0, dfloor, pfloor;
}

// forward declarations (global scope) matching the definitions below
void DiskRphiOpacity(MeshBlock *pmb, AthenaArray<Real> &prim);
void StellarHeatingRphi(MeshBlock *pmb, const Real time, const Real dt,
                        const AthenaArray<Real> &prim, const AthenaArray<Real> &prim_scalar,
                        const AthenaArray<Real> &bcc, AthenaArray<Real> &cons,
                        AthenaArray<Real> &cons_scalar);

//----------------------------------------------------------------------------------------
void Mesh::InitUserMeshData(ParameterInput *pin) {
  // read simple problem parameters
  rho0   = pin->GetOrAddReal("problem", "rho0", 1.0);

  // radiation / irradiation parameters
  r_star     = pin->GetOrAddReal("radiation","r_star", 1.0e-4);
  t_star     = pin->GetOrAddReal("radiation","t_star", 1.0);
  kappa_star = pin->GetOrAddReal("radiation","kappa_star", 1.0);
  kappa_a    = pin->GetOrAddReal("radiation","kappa_a", 0.0);

  rhounit = pin->GetOrAddReal("radiation","density_unit", 1.0);
  lunit   = pin->GetOrAddReal("radiation","length_unit", 1.0);
  tunit   = pin->GetOrAddReal("radiation","T_unit", 1.0);
  tfloor  = pin->GetOrAddReal("radiation","tfloor", 1e-12);
  tceiling= pin->GetOrAddReal("radiation","tceiling", 1e6);

  grazing_mu = pin->GetOrAddReal("radiation","grazing_mu", 0.1);

  // gravitational parameter and pressure-over-rho normalization
  gm0 = pin->GetOrAddReal("problem","GM0", 1.0);
  p0_over_r0 = pin->GetOrAddReal("problem","p0_over_r0", 0.0025);

  dfloor = pin->GetOrAddReal("hydro","dfloor", 1e-12);
  pfloor = pin->GetOrAddReal("hydro","pfloor", 1e-20);

  // register explicit source (stellar heating) if radiation is enabled
  if (NR_RADIATION_ENABLED || IM_RADIATION_ENABLED) {
    EnrollUserExplicitSourceFunction(StellarHeatingRphi);
  }
}

//----------------------------------------------------------------------------------------
void MeshBlock::InitUserMeshBlockData(ParameterInput *pin) {
  if (NR_RADIATION_ENABLED || IM_RADIATION_ENABLED) {
    // enroll opacity function with radiation module
    pnrrad->EnrollOpacityFunction(DiskRphiOpacity);
  }
}

//----------------------------------------------------------------------------------------
void MeshBlock::ProblemGenerator(ParameterInput *pin) {
  // Construct a vertically stratified density (exponential). Use PoverR and set
  // azimuthal (x2) velocity to Keplerian at fixed radius R_disk.

  AthenaArray<Real> ir_cm;
  Real *ir_lab;

  ir_cm.NewAthenaArray(pnrrad->n_fre_ang);
  if(NR_RADIATION_ENABLED || IM_RADIATION_ENABLED){
    ir_cm.NewAthenaArray(pnrrad->n_fre_ang);
  }

  for (int k=ks; k<=ke; ++k) {
    for (int j=js; j<=je; ++j) {
      for (int i=is; i<=ie; ++i) {
        Real z = pcoord->x1v(i); // X1 is vertical
        Real dens = rho0 * std::exp(-0.5*SQR(z)/p0_over_r0);
        dens = std::max(dens, dfloor);
        phydro->u(IDN,k,j,i) = dens;
        phydro->u(IM1,k,j,i) = 0.0;
        // azimuthal momentum = rho * v_phi (Keplerian)
        phydro->u(IM2,k,j,i) = 0.0;
        phydro->u(IM3,k,j,i) = 0.0;
        if (i == is && j == js && k == ks) {
          std::cout << phydro->u(IM2,k,j,i)/phydro->u(IDN,k,j,i) << std::endl;
          std::cout << peos->GetGamma() << std::endl;
        }
        Real press = p0_over_r0 * dens;
        press = std::max(press, pfloor);
        Real eint = press / (peos->GetGamma() - 1.0);
        
        // add kinetic energy
        Real ekin = 0.5*(SQR(phydro->u(IM1,k,j,i)) + SQR(phydro->u(IM2,k,j,i)) + SQR(phydro->u(IM3,k,j,i))) / phydro->u(IDN,k,j,i);
        phydro->u(IEN,k,j,i) = eint + ekin;

        if(NR_RADIATION_ENABLED || IM_RADIATION_ENABLED){
          Real gast = p0_over_r0;
          for(int n=0; n<pnrrad->n_fre_ang; ++n)
            ir_cm(n) = gast * gast * gast * gast;

          Real *mux = &(pnrrad->mu(0,k,j,i,0));
          Real *muy = &(pnrrad->mu(1,k,j,i,0));
          Real *muz = &(pnrrad->mu(2,k,j,i,0));

          ir_lab = &(pnrrad->ir(k,j,i,0));
          pnrrad->pradintegrator->ComToLab(0,0,0,mux,muy,muz,ir_cm,ir_lab);
   
        }// End Rad
      }
    }
  }
}

//----------------------------------------------------------------------------------------
void DiskRphiOpacity(MeshBlock *pmb, AthenaArray<Real> &prim) {
  // simple gray opacity: user-specified kappa_a (per mass) and kappa_star for stellar
  NRRadiation *prad = pmb->pnrrad;
  int il = pmb->is; int jl = pmb->js; int kl = pmb->ks;
  int iu = pmb->ie; int ju = pmb->je; int ku = pmb->ke;
  il -= NGHOST;
  iu += NGHOST;
  if(ju > jl){
    jl -= NGHOST;
    ju += NGHOST;
  }
  if(ku > kl){
    kl -= NGHOST;
    ku += NGHOST;
  }
  // convert kappa unit from cgs to code unit
  for (int k=kl; k<=ku; ++k) {
    for (int j=jl; j<=ju; ++j) {
      for (int i=il; i<=iu; ++i) {
        for (int ifr=0; ifr<prad->nfreq; ++ifr){
          Real rho = prim(IDN,k,j,i);
          prad->sigma_s(k,j,i,ifr) = 0.0; // use kappa_a as total absorption for simplicity
          prad->sigma_a(k,j,i,ifr) = kappa_a * rho * rhounit * lunit;
          prad->sigma_p(k,j,i,ifr) = kappa_a * rho * rhounit * lunit;
          prad->sigma_pe(k,j,i,ifr) = kappa_a * rho * rhounit * lunit;
        }
      }
    }
  }
}

//----------------------------------------------------------------------------------------
void StellarHeatingRphi(MeshBlock *pmb, const Real time, const Real dt,
                        const AthenaArray<Real> &prim, const AthenaArray<Real> &prim_scalar,
                        const AthenaArray<Real> &bcc, AthenaArray<Real> &cons,
                        AthenaArray<Real> &cons_scalar)
{
  // Ray-trace from top (outer X1) downwards at fixed R_disk.
  NRRadiation *prad = pmb->pnrrad;
  Coordinates *pco = pmb->pcoord;
  Real crat = prad? prad->crat : 1.0;
  Real prat = prad? prad->prat : 1.0;
  Real sigma_b = 0.25*crat*prat; // radiation constant in code units

  int is = pmb->is, ie = pmb->ie, js = pmb->js, je = pmb->je, ks = pmb->ks, ke = pmb->ke;
  // for each column (fixed j) and each k layer, integrate tau from top (i=ie) downwards
  int ncell = ie - is + 1;
  AthenaArray<Real> tau; //, fface, x1area, vol;
  tau.NewAthenaArray(ncell+2);
  //fface.NewAthenaArray(ncell+1);
  //x1area.NewAthenaArray(pmb->block_size.nx1+1);
  //vol.NewAthenaArray(pmb->ncells1);


  for (int k=ks; k<=ke; ++k) {
    for (int j=js; j<=je; ++j) {
      Real y = pco->x2v(j);
      // compute cumulative tau at faces
      for (int idx=0; idx<=ncell+1; ++idx) tau(idx)=0.0;
      for (int i=ie; i>=is; --i) {
        Real rho = prim(IDN,k,j,i);
        Real dz = pco->dx1f(i);
        tau(i) = tau(i+1) + kappa_star * rho * dz * rhounit * lunit;
      }

      // compute flux at faces
      /*for (int iface = 0; iface<=ncell; ++iface) {
        Real tau_here = tau(iface);
        Real F = (r_star / R_disk) * (r_star / R_disk) * sigma_b * t_star * t_star * t_star * t_star;
        if (grazing_mu > 0.0) F *= std::exp(-tau_here / grazing_mu);
        fface(iface) = F;
      }*/

      // compute face areas and cell volumes for this column
      //pco->Face1Area(k, j, is, ie+1, x1area);
      //pco->CellVolume(k, j, is, ie, vol);

      for (int i=is; i<=ie; ++i) {
        //int ic = i-is;
        //Real flux_in = fface(ic);
        //Real flux_out = fface(ic+1);
        //Real area_in = x1area(ic);
        //Real area_out = x1area(ic+1);
        Real F = r_star * r_star * sigma_b * t_star * t_star * t_star * t_star * 0.5 + 0.5*std::sin(y);
        if (grazing_mu > 0.0) F *= std::exp(-tau(i) / grazing_mu);

        Real dE = + dt * prim(IDN,k,j,i) * kappa_star * rhounit * lunit * F;
        
        // update conserved energy (IEN) while preserving kinetic and mag energy
        Real ekin = 0.5*(SQR(cons(IM1,k,j,i)) + SQR(cons(IM2,k,j,i)) + SQR(cons(IM3,k,j,i)))/cons(IDN,k,j,i);
        
        Real eint = cons(IEN,k,j,i) - ekin;
        eint += dE;

        //if (k==ks && j==js){
        //  std::cout << tau(i) << std::endl;
        //  std::cout << dE << std::endl;
        //  std::cout << eint << std::endl;
        //}

        // clamp to floors
        //Real eint_floor = 1.0/(pmb->peos->GetGamma()-1.0) * cons(IDN,k,j,i) * tfloor;
        //Real eint_ceil  = 1.0/(pmb->peos->GetGamma()-1.0) * cons(IDN,k,j,i) * tceiling;
        //eint = std::min(std::max(eint, eint_floor), eint_ceil);
        cons(IEN,k,j,i) = eint + ekin;
      }
    }
  }
}

//========================================================================================
