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
#ifdef MPI_PARALLEL
#include <mpi.h>
#endif

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
Real rho0, r_star, t_star, kappa_star, kappa_a;
Real rhounit, lunit, tunit, tfloor, tceiling;
Real grazing_mu; // grazing angle (cosine)
Real p0_over_r0, dfloor, pfloor, v_phi, amp;
int m_mode;
}


void Inner_rad_X1(MeshBlock *pmb, Coordinates *pco, NRRadiation *prad,
     const AthenaArray<Real> &w, FaceField &b, AthenaArray<Real> &ir,
      Real time, Real dt, int is, int ie, int js, int je, int ks, int ke, int ngh);
void Outer_rad_X1(MeshBlock *pmb, Coordinates *pco, NRRadiation *prad,
     const AthenaArray<Real> &w, FaceField &b, AthenaArray<Real> &ir,
      Real time, Real dt, int is, int ie, int js, int je, int ks, int ke, int ngh);

// User-defined boundary conditions for disk simulations
void DiskInnerX1(MeshBlock *pmb, Coordinates *pco, AthenaArray<Real> &prim,FaceField &b,
                 Real time, Real dt,
                 int il, int iu, int jl, int ju, int kl, int ku, int ngh);
void DiskOuterX1(MeshBlock *pmb, Coordinates *pco, AthenaArray<Real> &prim,FaceField &b,
                 Real time, Real dt,
                 int il, int iu, int jl, int ju, int kl, int ku, int ngh);

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

  grazing_mu = pin->GetOrAddReal("problem","grazing_mu", 0.1);
  m_mode     = pin->GetOrAddInteger("problem","m_mode", 0);
  amp        = pin->GetOrAddReal("problem","amp", 0.0);

  // gravitational parameter and pressure-over-rho normalization
  p0_over_r0 = pin->GetOrAddReal("problem","p0_over_r0", 0.0025);

  v_phi = pin->GetOrAddReal("problem","v_phi", 1.0);

  dfloor = pin->GetOrAddReal("hydro","dfloor", 1e-12);
  pfloor = pin->GetOrAddReal("hydro","pfloor", 1e-20);

  if (mesh_bcs[BoundaryFace::outer_x1] == GetBoundaryFlag("user")) {
    EnrollUserBoundaryFunction(BoundaryFace::outer_x1, DiskOuterX1);
    EnrollUserBoundaryFunction(BoundaryFace::inner_x1, DiskInnerX1);
  }
  if(NR_RADIATION_ENABLED || IM_RADIATION_ENABLED){
    if (mesh_bcs[BoundaryFace::outer_x1] == GetBoundaryFlag("user")) {
      EnrollUserRadBoundaryFunction(BoundaryFace::outer_x1, Outer_rad_X1);
      EnrollUserRadBoundaryFunction(BoundaryFace::inner_x1, Inner_rad_X1);
    }
  }
  // register explicit source (stellar heating) if radiation is enabled
  if (NR_RADIATION_ENABLED || IM_RADIATION_ENABLED) {
    EnrollUserExplicitSourceFunction(StellarHeatingRphi);
    
    // allocate global mesh-level tau storage for tracking optical depth from both directions
    int nfreq = pin->GetOrAddInteger("radiation","n_frequency",1);
    AllocateRealUserMeshDataField(2);
    // ruser_mesh_data[0]: tau_left (cumulative from left/top to each meshblock boundary)
    ruser_mesh_data[0].NewAthenaArray(mesh_size.nx3,
                                      mesh_size.nx2,
                                      nrbx1 + 1,
                                      nfreq);
    // ruser_mesh_data[1]: tau_right (cumulative from right/bottom to each meshblock boundary)
    ruser_mesh_data[1].NewAthenaArray(mesh_size.nx3,
                                      mesh_size.nx2,
                                      nrbx1 + 1,
                                      nfreq);
    for (int k=0; k<mesh_size.nx3; k++){
      for (int j=0; j<mesh_size.nx2; j++){
        for (int ib=0; ib<=nrbx1; ib++){
          for (int ifr=0; ifr<nfreq; ifr++){
            ruser_mesh_data[0](k,j,ib,ifr)=0.0;
            ruser_mesh_data[1](k,j,ib,ifr)=0.0;
          }
        }
      }
    }
    
    // flag to initialize/reset mesh-level tau storage each cycle
    AllocateIntUserMeshDataField(1);
    iuser_mesh_data[0].NewAthenaArray(1);
    iuser_mesh_data[0](0) = 0;
  }
}

//----------------------------------------------------------------------------------------
void MeshBlock::InitUserMeshBlockData(ParameterInput *pin) {
  if (NR_RADIATION_ENABLED || IM_RADIATION_ENABLED) {
    // enroll opacity function with radiation module
    pnrrad->EnrollOpacityFunction(DiskRphiOpacity);
    
    // allocate local meshblock-level tau storage for both directions
    AllocateRealUserMeshBlockDataField(2);
    // ruser_meshblock_data[0]: tau_left (integrated left to right)
    ruser_meshblock_data[0].NewAthenaArray(block_size.nx3,
                                           block_size.nx2,
                                           block_size.nx1,
                                           pnrrad->nfreq);
    // ruser_meshblock_data[1]: tau_right (integrated right to left)
    ruser_meshblock_data[1].NewAthenaArray(block_size.nx3,
                                           block_size.nx2,
                                           block_size.nx1,
                                           pnrrad->nfreq);
    for (int k=0; k<block_size.nx3; k++){
      for (int j=0; j<block_size.nx2; j++){
        for (int i=0; i<block_size.nx1; i++){
          for (int ifr=0; ifr<pnrrad->nfreq; ifr++){
            ruser_meshblock_data[0](k,j,i,ifr) = 0.0;
            ruser_meshblock_data[1](k,j,i,ifr) = 0.0;
          }
        }
      }
    }
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
        phydro->u(IM2,k,j,i) = dens * v_phi;
        phydro->u(IM3,k,j,i) = 0.0;
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
void MeshBlock::UserWorkInLoop(void){
  if(NR_RADIATION_ENABLED || IM_RADIATION_ENABLED){
    int lx3 = static_cast<int>(loc.lx3);
    int lx2 = static_cast<int>(loc.lx2);
    int lx1 = static_cast<int>(loc.lx1);
    int nrbx1 = pmy_mesh->mesh_size.nx1/block_size.nx1;
    
    // step 1. initialize on first call
    if (pmy_mesh->iuser_mesh_data[0](0) == 0){
      for (int k=0; k<pmy_mesh->mesh_size.nx3; k++){
        for (int j=0; j<pmy_mesh->mesh_size.nx2; j++){
          for (int ib=0; ib<=nrbx1; ib++){
            for (int ifr=0; ifr<pnrrad->nfreq; ifr++){
              pmy_mesh->ruser_mesh_data[0](k,j,ib,ifr)=0.0;
              pmy_mesh->ruser_mesh_data[1](k,j,ib,ifr)=0.0;
            }
          }
        }
      }
      pmy_mesh->iuser_mesh_data[0](0) = 1;
    }
    
    // step 2. compute local tau from inner->outer and outer->inner inside this block
    for (int k = ks; k <= ke; ++k) {
      for (int j = js; j <= je; ++j) {
        for (int ifr=0; ifr<pnrrad->nfreq; ++ifr){
          // local cumulative tau from inner side (is) toward outer side (ie)
          ruser_meshblock_data[0](k-ks,j-js,0,ifr) =
            kappa_star*phydro->u(IDN,k,j,is)*pcoord->dx1f(is)*rhounit*lunit;
          for (int i = is+1; i <= ie; ++i) {
            Real dtau = kappa_star
                         *phydro->u(IDN,k,j,i)*pcoord->dx1f(i)*rhounit*lunit;
            ruser_meshblock_data[0](k-ks,j-js,i-is,ifr) =
              ruser_meshblock_data[0](k-ks,j-js,i-1-is,ifr) + dtau;
          }

          // local cumulative tau from outer side (ie) toward inner side (is)
          ruser_meshblock_data[1](k-ks,j-js,ie-is,ifr) =
            kappa_star*phydro->u(IDN,k,j,ie)*pcoord->dx1f(ie)*rhounit*lunit;
          for (int i = ie-1; i >= is; --i) {
            Real dtau = kappa_star
                         *phydro->u(IDN,k,j,i)*pcoord->dx1f(i)*rhounit*lunit;
            ruser_meshblock_data[1](k-ks,j-js,i-is,ifr) =
              ruser_meshblock_data[1](k-ks,j-js,i+1-is,ifr) + dtau;
          }
          int tj  = lx2*block_size.nx2+(j-js);
          int tk  = lx3*block_size.nx3+(k-ks);

          // Store per-block total optical depth into boundary slots.
          Real block_tau = ruser_meshblock_data[0](k-ks,j-js,ie-is,ifr);
          pmy_mesh->ruser_mesh_data[0](tk,tj,lx1+1,ifr) = block_tau; // tau_top increments
          pmy_mesh->ruser_mesh_data[1](tk,tj,lx1,ifr)   = block_tau; // tau_bottom increments

          if (lx1 == 0) {
            pmy_mesh->ruser_mesh_data[0](tk,tj,0,ifr) = 0.0;
          }
          if (lx1 == nrbx1-1) {
            pmy_mesh->ruser_mesh_data[1](tk,tj,nrbx1,ifr) = 0.0;
          }
        }
      }
    }
  }
  return;
}

void Mesh::UserWorkInLoop(void){
  // Assemble global tau at meshblock surfaces for both directions.
  if(NR_RADIATION_ENABLED || IM_RADIATION_ENABLED){
    MeshBlock *pmb = my_blocks(0);
    if (pmb == nullptr || pmb->pnrrad == nullptr) return;
    int nfreq = pmb->pnrrad->nfreq;

#ifdef MPI_PARALLEL
    int ntot = mesh_size.nx3*mesh_size.nx2*(nrbx1+1)*nfreq;
    MPI_Allreduce(MPI_IN_PLACE,
                  ruser_mesh_data[0].data(),
                  ntot, MPI_ATHENA_REAL, MPI_SUM, MPI_COMM_WORLD);
    MPI_Allreduce(MPI_IN_PLACE,
                  ruser_mesh_data[1].data(),
                  ntot, MPI_ATHENA_REAL, MPI_SUM, MPI_COMM_WORLD);
#endif

    // Prefix sum for tau from inner->outer side.
    for (int tk=0; tk<mesh_size.nx3; ++tk) {
      for (int tj=0; tj<mesh_size.nx2; ++tj) {
        for (int ib=1; ib<=nrbx1; ++ib) {
          for (int ifr=0; ifr<nfreq; ++ifr) {
            ruser_mesh_data[0](tk,tj,ib,ifr) += ruser_mesh_data[0](tk,tj,ib-1,ifr);
          }
        }
      }
    }

    // Suffix sum for tau from outer->inner side.
    for (int tk=0; tk<mesh_size.nx3; ++tk) {
      for (int tj=0; tj<mesh_size.nx2; ++tj) {
        for (int ib=nrbx1-1; ib>=0; --ib) {
          for (int ifr=0; ifr<nfreq; ++ifr) {
            ruser_mesh_data[1](tk,tj,ib,ifr) += ruser_mesh_data[1](tk,tj,ib+1,ifr);
          }
        }
      }
    }

    // Reset init flag for next cycle.
    iuser_mesh_data[0](0) = 0;
  }
  return;
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
  // Ray-trace with two-directional optical depth tracking
  // Upper rays: attenuated by tau from top down to current position (tau_left)
  // Lower rays: attenuated by tau from bottom up to current position (tau_right)
  NRRadiation *prad = pmb->pnrrad;
  Coordinates *pco = pmb->pcoord;
  Real crat = prad? prad->crat : 1.0;
  Real prat = prad? prad->prat : 1.0;
  Real sigma_b = 0.25*crat*prat; // radiation constant in code units

  int is = pmb->is, ie = pmb->ie, js = pmb->js, je = pmb->je, ks = pmb->ks, ke = pmb->ke;
  int lx3 = static_cast<int>(pmb->loc.lx3);
  int lx2 = static_cast<int>(pmb->loc.lx2);
  int lx1 = static_cast<int>(pmb->loc.lx1);
  for (int k=ks; k<=ke; ++k) {
    for (int j=js; j<=je; ++j) {
      Real y = pco->x2v(j);
      int tj  = lx2*pmb->block_size.nx2+(j-js);
      int tk  = lx3*pmb->block_size.nx3+(k-ks);
      
      for (int i=is; i<=ie; ++i) {
        Real F_upper = r_star * r_star * sigma_b * t_star * t_star * t_star * t_star * (1.+ amp*0.5*(1.+std::sin(m_mode*y)));
        Real F_lower = r_star * r_star * sigma_b * t_star * t_star * t_star * t_star * (1.+ amp*0.5*(1.+std::sin(m_mode*y-PI)));
        F_upper = std::max(F_upper, 0.0);
        F_lower = std::max(F_lower, 0.0);

        if (grazing_mu > 0.0) {
          // tau_left: cumulative from left (top) boundary to current position
          Real tau_left_bound = pmb->pmy_mesh->ruser_mesh_data[0](tk,tj,lx1,0);
          Real tau_left_local = pmb->ruser_meshblock_data[0](k-ks,j-js,i-is,0);
          Real tau_here_upper = tau_left_bound + tau_left_local;
          
          // tau_right: cumulative from right (bottom) boundary to current position
          Real tau_right_bound = pmb->pmy_mesh->ruser_mesh_data[1](tk,tj,lx1+1,0);
          Real tau_right_local = pmb->ruser_meshblock_data[1](k-ks,j-js,i-is,0);
          Real tau_here_lower = tau_right_bound + tau_right_local;
          
          F_upper = F_upper * std::exp(-tau_here_upper / grazing_mu);
          F_lower = F_lower * std::exp(-tau_here_lower / grazing_mu);
        }
        Real dEdt = prim(IDN,k,j,i) * kappa_star * rhounit * lunit * (F_upper + F_lower);

        Real dE = dt * dEdt;
        
        // update conserved energy (IEN) while preserving kinetic and mag energy
        Real ekin = 0.5*(SQR(cons(IM1,k,j,i)) + SQR(cons(IM2,k,j,i)) + SQR(cons(IM3,k,j,i)))/cons(IDN,k,j,i);
        
        Real eint = cons(IEN,k,j,i) - ekin;
        eint += dE;
        cons(IEN,k,j,i) = eint + ekin;
      }
    }
  }
}

void DiskOuterX1(MeshBlock *pmb,Coordinates *pco, AthenaArray<Real> &prim, FaceField &b,
                 Real time, Real dt,
                 int il, int iu, int jl, int ju, int kl, int ku, int ngh) {
  
    for (int k=kl; k<=ku; ++k) {
      for (int j=jl; j<=ju; ++j) {
        for (int i=1; i<=ngh; ++i) {
          prim(IM1,k,j,iu+i) = prim(IM1,k,j,iu);
          prim(IM2,k,j,iu+i) = prim(IM2,k,j,iu);
          prim(IM3,k,j,iu+i) = prim(IM3,k,j,iu);
          if (NON_BAROTROPIC_EOS)
            prim(IEN,k,j,iu+i) = prim(IEN,k,j,iu);
        }
      }
    }
}

void DiskInnerX1(MeshBlock *pmb,Coordinates *pco, AthenaArray<Real> &prim, FaceField &b,
                 Real time, Real dt,
                 int il, int iu, int jl, int ju, int kl, int ku, int ngh) {
  
    for (int k=kl; k<=ku; ++k) {
      for (int j=jl; j<=ju; ++j) {
        for (int i=1; i<=ngh; ++i) {
          prim(IM1,k,j,il-i) = prim(IM1,k,j,il);
          prim(IM2,k,j,il-i) = prim(IM2,k,j,il);
          prim(IM3,k,j,il-i) = prim(IM3,k,j,il);
          if (NON_BAROTROPIC_EOS)
            prim(IEN,k,j,il-i) = prim(IEN,k,j,il);
        }
      }
    }
}

void Outer_rad_X1(MeshBlock *pmb, Coordinates *pco, NRRadiation *prad,
     const AthenaArray<Real> &w, FaceField &b, AthenaArray<Real> &ir,
      Real time, Real dt, int is, int ie, int js, int je, int ks, int ke, int ngh)
{
  for (int k=ks; k<=ke; ++k) {
    for (int j=js; j<=je; ++j) {
      for (int i=1; i<=ngh; ++i) {  
        for(int ifr=0; ifr<prad->nfreq; ++ifr){
          for(int n=0; n<prad->nang; ++n){
            Real& miux=prad->mu(0,k,j,is,n);
            if(miux > 0.0){
              ir(k,j,ie+i,ifr*prad->nang+n)
                = ir(k,j,ie,ifr*prad->nang+n);
            }else{
              ir(k,j,ie+i,ifr*prad->nang+n) = 0.;
            }
          }
        }
      }//i
    }//j
  }//k
  return;
}

void Inner_rad_X1(MeshBlock *pmb, Coordinates *pco, NRRadiation *prad,
     const AthenaArray<Real> &w, FaceField &b, AthenaArray<Real> &ir,
      Real time, Real dt, int is, int ie, int js, int je, int ks, int ke, int ngh)
{
  for (int k=ks; k<=ke; ++k) {
    for (int j=js; j<=je; ++j) {
      for (int i=1; i<=ngh; ++i) {  
        for(int ifr=0; ifr<prad->nfreq; ++ifr){
          for(int n=0; n<prad->nang; ++n){
            Real& miux=prad->mu(0,k,j,is,n);
            if(miux < 0.0){
              ir(k,j,is-i,ifr*prad->nang+n)
                = ir(k,j,is,ifr*prad->nang+n);
            }else{
              ir(k,j,is-i,ifr*prad->nang+n) = 0.;
            }
          }
        }
      }//i
    }//j
  }//k
  return;
}

//========================================================================================
