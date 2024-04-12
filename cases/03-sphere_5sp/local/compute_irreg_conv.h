#pragma once
#include "spade.h"
#include <iostream>
#include <stdio.h>

namespace detail{
	template <typename cent_t, typename diss_t, typename ducr_t, typename real_t, typename flx_t>
	_sp_hybrid inline auto swapdiss(const spade::convective::hybrid_scheme_t<cent_t, diss_t, ducr_t, flx_t>& base, const real_t& new_diss_val)
	{
		spade::state_sensor::const_sensor_t new_sensor(new_diss_val);
		return spade::convective::hybrid_scheme_t(base.scheme0, base.scheme1, new_sensor, base.flux_designator());
	}
}

namespace local
{
	template <typename ghost_t, typename array_t, typename gas_t, typename rhs_t, typename tscheme_t, typename xs_t>
    inline void rhs_irreg_conv(const array_t& prim, rhs_t& rhs, const gas_t& gas, const tscheme_t& tscheme, const ghost_t& ghost, const xs_t& xs)
    {
		using pnt_t         = xs_t::value_type;
		using real_t        = pnt_t::value_type;
		using dble_t        = typename array_t::fundamental_type;
		using index_t       = spade::grid::cell_idx_t;
		using grad_t        = spade::ctrs::array<typename array_t::alias_type, 3>;
		using v3_t          = spade::ctrs::array<real_t, 3>;
		using flux_t        = rhs_t::alias_type;
		//
		const auto ghst_img = ghost.image(prim.device());
		const auto prim_img = prim.image();
		auto rhs_img        = rhs.image();
		//
		const auto& grid    = prim.get_grid();
		const int rank      = grid.group().rank();
		const auto grid_img = grid.image(prim.device());
		//
		int nx = grid.get_num_cells(0);
		int ny = grid.get_num_cells(1);
		int nz = grid.get_num_cells(2);

		//
		for (int dir = 0; dir < ghst_img.aligned.size(); ++dir)
        {
			const auto& list = ghst_img.aligned[dir];
			std::size_t size_here = list.indices.size();
			const auto range    = spade::dispatch::ranges::from_array(list.indices, prim.device());
			auto irreg_rhs = [=] _sp_hybrid (const std::size_t& idx) mutable
            {
				//
				//      +  +  +
				//
				//      +  +  +
				//        --- (i,j+1/2)
				//      +  o  +
				//  ----------------- boundary & o = "irreg. pt."
				//
				//
				int sign = list.signs[idx];     //ghst_img.signs<0 irregular point is above
				//ghost cell index
				auto icellg = list.indices[idx][0];
				auto icell = icellg;
				//get irregular point index
				icell.i(dir) += sign;
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
				spade::grid::face_idx_t iface_p = spade::grid::cell_to_face(icell, dir, 1);//spade::utils::max(sign,0));
				//
				const real_t local_diss = 8e-2;
				auto new_tscheme = detail::swapdiss(tscheme,local_diss);
				using new_scheme_t = typename spade::utils::remove_all<decltype(new_tscheme)>::type;
				using stencil_type = spade::omni::stencil_union<typename tscheme_t::omni_type, typename new_scheme_t::omni_type>;
				using vdata_t = spade::omni::stencil_data_t<stencil_type, array_t>;
				//
				vdata_t input_data;
				spade::omni::retrieve(grid_img,prim_img,iface_p,input_data);
				//
				flux_t flux_p_reg = tscheme(input_data);
				flux_p            = new_tscheme(input_data);
				//
				
				spade::grid::face_idx_t iface_m = spade::grid::cell_to_face(icell, dir, 0);
				spade::omni::retrieve(grid_img,prim_img,iface_m,input_data);
				//
				flux_t flux_m_reg = tscheme(input_data);
				flux_m            = new_tscheme(input_data);
				//
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
						temp -= ( flux_m_reg - flux_m)/dx;
					}
					else
					{
						temp -= ( flux_p-flux_p_reg)/dx;
					}
					rhs_img.set_elem(icell, temp);
				}

			};
			spade::dispatch::execute(range, irreg_rhs);
		}
	}
}
