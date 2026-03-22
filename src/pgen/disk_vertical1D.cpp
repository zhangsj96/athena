//========================================================================================
// Athena++ problem generator: 1D vertical slab with stellar irradiation from the top
// X1 = z, with user-controlled grazing-angle beam injection at outer_x1.
// Incoming stellar flux is split between the two nearest incoming ordinates so that the
// discrete boundary condition matches the target grazing_mu as closely as possible while
// conserving the injected normal flux.
//========================================================================================

// C++ headers
#include <algorithm>
#include <cmath>
#include <limits>

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
Real rho0;
Real r_star, t_star, kappa_star, kappa_a;
Real rhounit, lunit, tfloor, tceiling;
Real grazing_mu;
Real p0_over_r0, dfloor, pfloor;

struct BeamStencil {
  int lower = -1;
  int upper = -1;
  Real lower_flux_fraction = 0.0;
  Real upper_flux_fraction = 0.0;
  bool valid = false;
};

Real StellarFlux(const NRRadiation *prad) {
  Real crat = prad != nullptr ? prad->crat : 1.0;
  Real prat = prad != nullptr ? prad->prat : 1.0;
  Real sigma_b = 0.25 * crat * prat;
  Real flux = r_star * r_star * sigma_b * t_star * t_star * t_star * t_star;
  return std::max(flux, static_cast<Real>(0.0));
}

BeamStencil BuildBeamStencil(NRRadiation *prad, int k, int j, int i) {
  BeamStencil stencil;
  Real target_mu = std::max(static_cast<Real>(0.0),
                            std::min(grazing_mu, static_cast<Real>(1.0)));
  Real lower_mu = -1.0;
  Real upper_mu = 2.0;
  Real nearest_delta = std::numeric_limits<Real>::max();
  int nearest = -1;

  for (int n = 0; n < prad->nang; ++n) {
    Real mux = prad->mu(0, k, j, i, n);
    if (mux >= 0.0) {
      continue;
    }

    Real mu_abs = -mux;
    Real delta = std::abs(mu_abs - target_mu);
    if (delta < nearest_delta) {
      nearest_delta = delta;
      nearest = n;
    }
    if (mu_abs <= target_mu && mu_abs > lower_mu) {
      lower_mu = mu_abs;
      stencil.lower = n;
    }
    if (mu_abs >= target_mu && mu_abs < upper_mu) {
      upper_mu = mu_abs;
      stencil.upper = n;
    }
  }

  if (nearest < 0) {
    return stencil;
  }

  if (stencil.lower < 0 || stencil.upper < 0 || stencil.lower == stencil.upper ||
      std::abs(upper_mu - lower_mu) <= static_cast<Real>(1.0e-12)) {
    stencil.lower = nearest;
    stencil.upper = -1;
    stencil.lower_flux_fraction = 1.0;
    stencil.upper_flux_fraction = 0.0;
    stencil.valid = true;
    return stencil;
  }

  stencil.upper_flux_fraction = (target_mu - lower_mu) / (upper_mu - lower_mu);
  stencil.lower_flux_fraction = 1.0 - stencil.upper_flux_fraction;
  stencil.valid = true;
  return stencil;
}

void InjectBeamBin(NRRadiation *prad, AthenaArray<Real> &ir, int k, int j, int i, int ifr,
                   int n, Real flux_fraction, Real total_flux) {
  if (n < 0 || flux_fraction <= 0.0 || total_flux <= 0.0) {
    return;
  }

  Real mux = prad->mu(0, k, j, i, n);
  Real mu_abs = -mux;
  Real weight = prad->wmu(n);
  if (mu_abs <= 0.0 || weight <= 0.0) {
    return;
  }

  ir(k, j, i, ifr * prad->nang + n) = flux_fraction * total_flux / (weight * mu_abs);
}
}  // namespace

void OuterHydroX1(MeshBlock *pmb, Coordinates *pco, AthenaArray<Real> &prim, FaceField &b,
                  Real time, Real dt, int il, int iu, int jl, int ju, int kl, int ku,
                  int ngh);
void OuterRadX1(MeshBlock *pmb, Coordinates *pco, NRRadiation *prad,
                const AthenaArray<Real> &w, FaceField &b, AthenaArray<Real> &ir,
                Real time, Real dt, int is, int ie, int js, int je, int ks, int ke,
                int ngh);
void VerticalOpacity(MeshBlock *pmb, AthenaArray<Real> &prim);

//----------------------------------------------------------------------------------------
void Mesh::InitUserMeshData(ParameterInput *pin) {
  rho0 = pin->GetOrAddReal("problem", "rho0", 1.0);
  p0_over_r0 = pin->GetOrAddReal("problem", "p0_over_r0", 0.0025);
  grazing_mu = pin->GetOrAddReal("problem", "grazing_mu", 0.1);

  r_star = pin->GetOrAddReal("radiation", "r_star", 1.0e-4);
  t_star = pin->GetOrAddReal("radiation", "t_star", 1.0);
  kappa_star = pin->GetOrAddReal("radiation", "kappa_star", 1.0);
  kappa_a = pin->GetOrAddReal("radiation", "kappa_a", 0.0);
  rhounit = pin->GetOrAddReal("radiation", "density_unit", 1.0);
  lunit = pin->GetOrAddReal("radiation", "length_unit", 1.0);
  tfloor = pin->GetOrAddReal("radiation", "tfloor", 1.0e-12);
  tceiling = pin->GetOrAddReal("radiation", "tceiling", 1.0e6);

  dfloor = pin->GetOrAddReal("hydro", "dfloor", 1.0e-12);
  pfloor = pin->GetOrAddReal("hydro", "pfloor", 1.0e-20);

  if (mesh_bcs[BoundaryFace::outer_x1] == GetBoundaryFlag("user")) {
    EnrollUserBoundaryFunction(BoundaryFace::outer_x1, OuterHydroX1);
  }
  if ((NR_RADIATION_ENABLED || IM_RADIATION_ENABLED) &&
      mesh_bcs[BoundaryFace::outer_x1] == GetBoundaryFlag("user")) {
    EnrollUserRadBoundaryFunction(BoundaryFace::outer_x1, OuterRadX1);
  }
}

//----------------------------------------------------------------------------------------
void MeshBlock::InitUserMeshBlockData(ParameterInput *pin) {
  if (NR_RADIATION_ENABLED || IM_RADIATION_ENABLED) {
    pnrrad->EnrollOpacityFunction(VerticalOpacity);
  }
}

//----------------------------------------------------------------------------------------
void MeshBlock::ProblemGenerator(ParameterInput *pin) {
  for (int k = ks; k <= ke; ++k) {
    for (int j = js; j <= je; ++j) {
      for (int i = is; i <= ie; ++i) {
        Real z = pcoord->x1v(i);
        Real dens = rho0 * std::exp(-0.5 * SQR(z) / p0_over_r0);
        dens = std::max(dens, dfloor);
        phydro->u(IDN, k, j, i) = dens;
        phydro->u(IM1, k, j, i) = 0.0;
        phydro->u(IM2, k, j, i) = 0.0;
        phydro->u(IM3, k, j, i) = 0.0;

        Real press = std::max(p0_over_r0 * dens, pfloor);
        Real eint = press / (peos->GetGamma() - 1.0);
        phydro->u(IEN, k, j, i) = eint;

        if (NR_RADIATION_ENABLED || IM_RADIATION_ENABLED) {
          AthenaArray<Real> ir_cm;
          ir_cm.NewAthenaArray(pnrrad->n_fre_ang);
          Real gast = std::max(p0_over_r0, tfloor);
          gast = std::min(gast, tceiling);
          for (int n = 0; n < pnrrad->n_fre_ang; ++n) {
            ir_cm(n) = gast * gast * gast * gast;
          }

          Real *mux = &(pnrrad->mu(0, k, j, i, 0));
          Real *muy = &(pnrrad->mu(1, k, j, i, 0));
          Real *muz = &(pnrrad->mu(2, k, j, i, 0));
          Real *ir_lab = &(pnrrad->ir(k, j, i, 0));
          pnrrad->pradintegrator->ComToLab(0, 0, 0, mux, muy, muz, ir_cm, ir_lab);
        }
      }
    }
  }
}

//----------------------------------------------------------------------------------------
void VerticalOpacity(MeshBlock *pmb, AthenaArray<Real> &prim) {
  NRRadiation *prad = pmb->pnrrad;
  Real absorption_kappa = (kappa_star > 0.0 ? kappa_star : kappa_a);
  int il = pmb->is - NGHOST;
  int iu = pmb->ie + NGHOST;
  int jl = pmb->js;
  int ju = pmb->je;
  int kl = pmb->ks;
  int ku = pmb->ke;
  if (ju > jl) {
    jl -= NGHOST;
    ju += NGHOST;
  }
  if (ku > kl) {
    kl -= NGHOST;
    ku += NGHOST;
  }

  for (int k = kl; k <= ku; ++k) {
    for (int j = jl; j <= ju; ++j) {
      for (int i = il; i <= iu; ++i) {
        Real rho = prim(IDN, k, j, i);
        for (int ifr = 0; ifr < prad->nfreq; ++ifr) {
          prad->sigma_s(k, j, i, ifr) = 0.0;
          prad->sigma_a(k, j, i, ifr) = absorption_kappa * rho * rhounit * lunit;
          prad->sigma_p(k, j, i, ifr) = absorption_kappa * rho * rhounit * lunit;
          prad->sigma_pe(k, j, i, ifr) = absorption_kappa * rho * rhounit * lunit;
        }
      }
    }
  }
}

//----------------------------------------------------------------------------------------
void OuterHydroX1(MeshBlock *pmb, Coordinates *pco, AthenaArray<Real> &prim, FaceField &b,
                  Real time, Real dt, int il, int iu, int jl, int ju, int kl, int ku,
                  int ngh) {
  for (int k = kl; k <= ku; ++k) {
    for (int j = jl; j <= ju; ++j) {
      for (int i = 1; i <= ngh; ++i) {
        prim(IDN, k, j, iu + i) = prim(IDN, k, j, iu);
        prim(IM1, k, j, iu + i) = prim(IM1, k, j, iu);
        prim(IM2, k, j, iu + i) = prim(IM2, k, j, iu);
        prim(IM3, k, j, iu + i) = prim(IM3, k, j, iu);
        if (NON_BAROTROPIC_EOS) {
          prim(IEN, k, j, iu + i) = prim(IEN, k, j, iu);
        }
      }
    }
  }
}

//----------------------------------------------------------------------------------------
void OuterRadX1(MeshBlock *pmb, Coordinates *pco, NRRadiation *prad,
                const AthenaArray<Real> &w, FaceField &b, AthenaArray<Real> &ir,
                Real time, Real dt, int is, int ie, int js, int je, int ks, int ke,
                int ngh) {
  Real total_flux = StellarFlux(prad);

  for (int k = ks; k <= ke; ++k) {
    for (int j = js; j <= je; ++j) {
      BeamStencil stencil = BuildBeamStencil(prad, k, j, ie);
      for (int i = 1; i <= ngh; ++i) {
        int ig = ie + i;
        for (int ifr = 0; ifr < prad->nfreq; ++ifr) {
          for (int n = 0; n < prad->nang; ++n) {
            Real mux = prad->mu(0, k, j, ie, n);
            if (mux > 0.0) {
              ir(k, j, ig, ifr * prad->nang + n) = ir(k, j, ie, ifr * prad->nang + n);
            } else {
              ir(k, j, ig, ifr * prad->nang + n) = 0.0;
            }
          }

          if (stencil.valid) {
            InjectBeamBin(prad, ir, k, j, ig, ifr, stencil.lower,
                          stencil.lower_flux_fraction, total_flux);
            InjectBeamBin(prad, ir, k, j, ig, ifr, stencil.upper,
                          stencil.upper_flux_fraction, total_flux);
          }
        }
      }
    }
  }
}

//========================================================================================
