#pragma once
#include "spade.h"
#include <iostream>
#include <stdio.h>

#define DEBUG 0
namespace local
{
  template <
    typename ghost_t,
    typename sample_t,
    typename xs_t,
    typename array_t,
    typename gas_t,
    typename vlaw_t,
    typename rhs_t,
    typename vector_t,
    typename vscheme_t>
  inline void rhs_irreg_visc(
                             const array_t& prim,
                             rhs_t& rhs,
                             const gas_t& gas,
                             const vlaw_t& vlaw,
                             const vscheme_t& vscheme,
                             const ghost_t& ghost,
                             const sample_t& sample,
                             const xs_t& xs,
                             const vector_t& tau_w,
                             const vector_t& q_w)
  {
    using float_t       = vlaw_t::value_type;
    using pnt_t         = xs_t::value_type;
    using real_t        = pnt_t::value_type;
    using dble_t        = typename array_t::fundamental_type;
    using index_t       = spade::grid::cell_idx_t;
    using grad_t        = spade::ctrs::array<typename array_t::alias_type, 3>;
    using v3_t          = spade::ctrs::array<real_t, 3>;
    using flux_t        = rhs_t::alias_type;
    using smpl_t        = typename array_t::alias_type;
    using vdata_t       = spade::omni::stencil_data_t<typename vscheme_t::omni_type,array_t>;
    //
    const auto ghst_img = ghost.image(prim.device());
    const auto smpl_img = spade::utils::make_vec_image(sample);
    const auto xs_img   = spade::utils::make_vec_image(xs.data(prim.device()));
    const auto prim_img = prim.image();
    auto rhs_img        = rhs.image();
    const auto tau_img  = spade::utils::make_vec_image(tau_w);
    const auto q_img    = spade::utils::make_vec_image(q_w);

    //
    const auto& grid    = prim.get_grid();
    const int rank      = grid.group().rank();
    const auto grid_img = grid.image(prim.device());

    int nx = grid.get_num_cells(0);
    int ny = grid.get_num_cells(1);
    int nz = grid.get_num_cells(2);
#if (DEBUG==1)
    print("ghost.size() = ",ghst_img.indices.size());
#endif
    std::size_t wm_offset = 0;
    for (int dir = 0; dir < ghst_img.aligned.size(); ++dir)
    {
        const auto& list = ghst_img.aligned[dir];
        std::size_t size_here = list.indices.size();
        const auto range    = spade::dispatch::ranges::from_array(list.indices, prim.device());
        auto irreg_rhs = [=] _sp_hybrid (const std::size_t& idx) mutable
        {
            std::size_t wm_idx = idx + wm_offset;
            //
            //      +  +  +
            //
            //      +  +  +
            //        --- (i,j+1/2)
            //      +  o  +
            //  ----------------- boundary & o = "irreg. pt."
            //
#if (DEBUG==1)
            print("here =======================");
#endif
        //
            int sign = list.signs[idx];     //ghst_img.signs<0 irregular point is above
            //ghost cell index
            auto icellg = list.indices[idx][0];
            auto icell = icellg;
            //get irregular point index
            icell.i(dir) += sign;
            //        auto icell_p = icell;
            //        icell_p.i(dir) += sign;
            //Compute distance to the wall
            const pnt_t& bndy_x = list.boundary_points[idx];
            const pnt_t& cell_x = grid_img.get_coords(icell);
            v3_t invdx;
            for (int d=0;d<invdx.size();++d) invdx[d] = 1./grid_img.get_dx(d,icell.lb());
            const real_t& dx    = grid_img.get_dx(dir,icell.lb());
            //==========================================================================================
            // face index away from the immersed boundary
            //==========================================================================================
            //
            // compute viscous flux at (i,j+1/2)
            //
            flux_t flux_p,flux_m;
#if (DEBUG==1)
            print("flux_p");
#endif
            spade::grid::face_idx_t iface_p = spade::grid::cell_to_face(icell, dir, 1);//spade::utils::max(sign,0));
#if (DEBUG==1)
            print("iface_p = ",iface_p);
#endif
            //double check that vflux and flux_p are identical
            vdata_t input_data;
            spade::omni::retrieve(grid_img,prim_img,iface_p,input_data);
            flux_p = vscheme(input_data);
            //visc_flux(flux_p,prim_img,gas,vlaw,iface_p,icell,invdx,dir, 1);
#if (DEBUG==1)
            print("flux_p",flux_p);
#endif
            spade::grid::face_idx_t iface_m = spade::grid::cell_to_face(icell, dir, 0);//spade::utils::max(sign,0));
#if (DEBUG==1)
            print("iface_m = ",iface_m);
#endif
            spade::omni::retrieve(grid_img,prim_img,iface_m,input_data);
            flux_m = vscheme(input_data);
            //visc_flux(flux_m,prim_img,gas,vlaw,iface_m,icell,invdx,dir,-1);
            //
            //Assemble flux on the boundary
            //
            spade::linear_algebra::dense_mat<float_t, 3> tau2(0.);
            tau2(1,0) = tau_img[wm_idx];
            tau2(0,1) = tau2(1,0);
            //compute matrix transform for viscous flux
            v3_t tvec,tvec2;
            const smpl_t& smpl_val = smpl_img[wm_idx];
            tvec[0]=smpl_val.u();
            tvec[1]=smpl_val.v();
            tvec[2]=smpl_val.w();
            const auto& nvec = list.boundary_normals[idx];
            dble_t vn = spade::ctrs::dot_prod(tvec,nvec);
            tvec -= vn*nvec;
            tvec /= spade::ctrs::array_norm(tvec);
            tvec2 = spade::ctrs::cross_prod(nvec,tvec);
            spade::linear_algebra::dense_mat<float_t, 3> mat(0.);
            mat(0,0)=nvec [0];mat(0,1)=nvec [1];mat(0,2)=nvec [2];
            mat(1,0)=tvec [0];mat(1,1)=tvec [1];mat(1,2)=tvec [2];
            mat(2,0)=tvec2[0];mat(2,1)=tvec2[1];mat(2,2)=tvec2[2];
            //
#if (DEBUG==1)
            print("bndy_x = ",bndy_x);
            print("cell_x = ",cell_x);
            print("==== irreg ====");
            print("icell  = ",icell.i(0));
            print("jcell  = ",icell.i(1));
            print("kcell  = ",icell.i(2));
            print("lbcell = ",icell.i(3));
            print("==== ghost ====");
            print("icell  = ",icellg.i(0));
            print("jcell  = ",icellg.i(1));
            print("kcell  = ",icellg.i(2));
            print("lbcell = ",icellg.i(3));
            print("dir    = ",dir);
            print("sign   = ",sign);
            print("nvec   = ",nvec);
            print("tvec   = ",tvec);
            print("tvec2  = ",tvec2);
#endif
            //
            spade::linear_algebra::dense_mat<float_t, 3> matt(0.);
            matt=spade::linear_algebra::transpose(mat);
#if (DEBUG==1)
            print("tau  = ",tau2);
#endif
            tau2 = matt*tau2;
            tau2 = tau2*mat;

#if (DEBUG==1)
            //
            print("mat  = ",mat);
            print("matt = ",matt);
            print("tau2 = ",tau2);
#endif
            //
            //compute stress tensor at the wall
            //
            flux_t flux_bndy;
            flux_bndy.continuity() = 0.0;
            flux_bndy.energy()     = 0.0;
            flux_bndy.x_momentum() =-tau2(0,dir);
            flux_bndy.y_momentum() =-tau2(1,dir);
            flux_bndy.z_momentum() =-tau2(2,dir);
            //

if ((tau2(0,dir) != tau2(0,dir))||(tau2(1,dir) != tau2(1,dir))||(tau2(2,dir) != tau2(2,dir)))
  {
    print("NaN irreg_rhs \n");
  }


            //apply divergence (start with assuming regular stencil)
            bool can_set_rhs = true;
            can_set_rhs = can_set_rhs && icell.i() >= 0;
            can_set_rhs = can_set_rhs && icell.j() >= 0;
            can_set_rhs = can_set_rhs && icell.k() >= 0;
            can_set_rhs = can_set_rhs && icell.i() < nx;
            can_set_rhs = can_set_rhs && icell.j() < ny;
            can_set_rhs = can_set_rhs && icell.k() < nz;
            if (can_set_rhs)
            {
                flux_t temp = rhs_img.get_elem(icell);
                //top: sign=-1 and bottom: sign=1
                if (sign > 0)
                {
                    temp -= ( flux_m - flux_bndy)/dx;
                }
                else
                {
                    temp -= ( flux_bndy-flux_p)/dx;
                }
#if (DEBUG==1)
                flux_t checkFlux;
                checkFlux = ( flux_m-flux_p)/dx;
                print("temp    = ",temp);
                print("check   = ",checkFlux);
                print("flux_m = ",flux_m);
                print("flux_p = ",flux_p);
                print("flux_b = ",flux_bndy);
                print("dx     = ",dx);
                if (nvec[1]<0)
                {
                    char c;
                    std::cin >> c;
                }
#endif

                rhs_img.set_elem(icell, temp);
            }
        //

      };
      spade::dispatch::execute(range, irreg_rhs); 
      wm_offset += size_here;
    }
#if (DEBUG==1)
    print("Irregular right-hand-side");
#endif
  }
}
