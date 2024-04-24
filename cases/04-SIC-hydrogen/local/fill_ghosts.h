#pragma once

#include "spade.h"

namespace local
{
  template <typename arr_t, typename ghost_t>
  static void zero_ghost_rhs(arr_t& array, const ghost_t& ghost)
    {
      auto arr_img  = array.image();
      const auto ghst_img = ghost.image(array.device());
      const auto& grid = array.get_grid();
      int nx = grid.get_num_cells(0);
      int ny = grid.get_num_cells(1);
      int nz = grid.get_num_cells(2);

        for (int dir = 0; dir < ghst_img.aligned.size(); ++dir)
        {
            const auto list = ghst_img.aligned[dir];
            const auto range = spade::dispatch::ranges::from_array(list.indices, array.device());
            int nlayer = ghost.num_layers();
            auto fill_ghst = [=] _sp_hybrid (const std::size_t& idx) mutable
            {
                for (int ii = 0; ii < nlayer; ++ii)
                {
                  auto icell = list.indices[idx][ii];
                  bool can_set_rhs = true;
                  can_set_rhs = can_set_rhs && (icell.i() >= 0);
                  can_set_rhs = can_set_rhs && (icell.j() >= 0);
                  can_set_rhs = can_set_rhs && (icell.k() >= 0);
                  can_set_rhs = can_set_rhs && (icell.i() < nx);
                  can_set_rhs = can_set_rhs && (icell.j() < ny);
                  can_set_rhs = can_set_rhs && (icell.k() < nz);
                  typename arr_t::alias_type zer = 0.0;
                  if (can_set_rhs) arr_img.set_elem(icell, zer);
                }
            };
            spade::dispatch::execute(range, fill_ghst);
        }
    }
  template <typename prim_t, typename rhs_t, typename grid_t, typename arr_t>
  inline void zero_rhs_inside(const prim_t& prim, const grid_t& grid, rhs_t& rhs, const arr_t& inout)
  {
    //using prim_t = typename decltype(prim)::alias_type;
    using coor_t = typename grid_t::coord_type;
    using point_t = spade::coords::point_t<coor_t>;

    auto grid_img = grid.image(prim.device());
    auto rhs_img  = rhs.image();
    auto inout_img= inout.image();

    auto var_range = spade::dispatch::support_of(prim,spade::grid::exclude_exchanges);

    auto loop = [=] _sp_hybrid (const spade::grid::cell_idx_t& i_cell) mutable
      {
        const point_t x = grid_img.get_coords(i_cell);
        const int inside= inout_img.get_elem(i_cell);
        if (inside==1)
          {
			  for (int v=0;v<18;v++)
              {
                rhs_img(v,i_cell) = 0.;
              }
          }
      };
    spade::dispatch::execute(var_range, loop);
  }

	template<typename image_type, typename rtype, typename norm_type>
	_sp_hybrid static void ghost_slipwall(const image_type& image, const rtype& distWallToImage, const rtype& distGhostToWall, const norm_type& nvec, image_type& ghost_bc)
	{

		// Copy image state over to enforce adiabatic and zero pressure gradient
		ghost_bc = image;
		
		// Get normal and tangential velocity at image point
		using vec_t = spade::ctrs::array<rtype, 3>;
		vec_t u = {image.u(), image.v(), image.w()};
		vec_t unorm = spade::ctrs::array_val_cast<rtype>(spade::ctrs::dot_prod(u, nvec)*nvec);
		vec_t utang = u - unorm;
					
		// Copy tangential velocity
		for (int i = 0; i < 3; ++i)
		{
			ghost_bc.u(i) = utang[i];
		}

		// Linear ramp on normal velocity
		for (int i = 0; i < 3; ++i)
		{
			ghost_bc.u(i) += -(distGhostToWall / distWallToImage)*u[i];
		}
		
		return;
	}
    
  template <typename arr_t, typename ghost_t, typename xs_t, typename datas_t>
  static void fill_ghost_vals(arr_t& array, const ghost_t& ghosts, const xs_t& xs, const datas_t& datas)
    {
        auto geom_img  = array.get_grid().image(spade::partition::local, array.device());
        auto arr_img   = array.image();
        auto ghst_img  = ghosts.image(array.device());
        auto xs_img    = spade::utils::make_vec_image(xs.data(array.device()));
        auto data_img  = spade::utils::make_vec_image(datas);
        
        using alias_type = typename arr_t::alias_type;
        using real_t     = typename arr_t::fundamental_type;
        
        const auto range = spade::dispatch::ranges::from_array(datas, array.device());

        std::size_t offset = 0;
        for (int dir = 0; dir < ghosts.aligned.size(); ++dir)
        {
            const auto& list = ghst_img.aligned[dir];
            
            std::size_t list_size = list.indices.size();
            std::size_t nlayers   = list.num_layer();

            auto o_range = spade::dispatch::ranges::make_range(0UL, list_size);
            auto i_range = spade::dispatch::ranges::make_range(0UL, nlayers);
            
            spade::dispatch::kernel_threads_t kpool(i_range, array.device());
            
            using threads_type = decltype(kpool);
            
            auto loop = [=] _sp_hybrid (const std::size_t& idx, const threads_type& threads) mutable
            {
                threads.exec([&](const std::size_t& ilayer)
                {
                    std::size_t idx_1d = offset+idx*nlayers+ilayer;
                    const alias_type& sampl_value  = data_img[idx_1d];
                    const auto& nvec    = list.closest_normals[idx][ilayer];
                    const auto icell    = list.indices[idx][ilayer];
					const auto& sampl_x = xs_img[idx_1d];
                    const auto& bndy_x  = list.closest_points[idx][ilayer];
                    const auto& ghst_x  = geom_img.get_coords(list.indices[idx][ilayer]);
                    const bool  do_fill = list.can_fill[idx][ilayer];
					
					//        +
					//        |      |
					//        |      d1
					//  ______|____ _|_
					//        +      d0

					// Set distances
					const real_t d1 = spade::ctrs::array_norm(sampl_x - bndy_x);
					const real_t d0 = spade::ctrs::array_norm(bndy_x  - ghst_x);
					
					// Set ghost bc
					alias_type ghost_value;
					ghost_slipwall(sampl_value, d1, d0, nvec, ghost_value);
					
					//
                    if (do_fill) arr_img.set_elem(icell, ghost_value);
                });
            };
            
            spade::dispatch::execute(o_range, loop, kpool);
            
            offset    += list_size*nlayers;
        }
        
        //Now do the diagonals
        const auto& list = ghst_img.diags;
            
        std::size_t diag_size = list.indices.size();
        
        auto diag_range = spade::dispatch::ranges::make_range(0UL, diag_size);
        auto diag_loop = [=] _sp_hybrid (const std::size_t& idx) mutable
        {
            std::size_t idx_1d = offset+idx;
            const alias_type& sampl_value = data_img[idx_1d];
            const auto& nvec    = list.closest_normals[idx];
            const auto icell    = list.indices[idx];
            const auto& bndy_x  = list.closest_points[idx];
            const auto& ghst_x  = geom_img.get_coords(list.indices[idx]);
			const auto& sampl_x = xs_img[idx_1d];
            const bool  do_fill = list.can_fill[idx];
            
            //        +
            //        |      |
            //        |      d1
            //  ______|____ _|_
            //        +      d0
    
            const real_t d1 = spade::ctrs::array_norm(sampl_x - bndy_x);
            const real_t d0 = spade::ctrs::array_norm(bndy_x  - ghst_x);

			// Set ghost bc
			alias_type ghost_value;
			ghost_slipwall(sampl_value, d1, d0, nvec, ghost_value);
			
			if (do_fill) arr_img.set_elem(icell, ghost_value);
        };
        spade::dispatch::execute(diag_range, diag_loop, array.device());        
    }
}
