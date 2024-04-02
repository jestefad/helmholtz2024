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
  template <typename prim_t, typename rhs_t, typename grid_t, typename gas_t, typename arr_t>
  inline void zero_rhs_inside(const prim_t& prim,const grid_t& grid, rhs_t& rhs,const gas_t& gas,const arr_t& inout)
  {
    //using prim_t = typename decltype(prim)::alias_type;
    using real_t = typename prim_t::value_type;
    using coor_t = typename grid_t::coord_type;
    using index_t = spade::grid::cell_idx_t;
    using point_t = spade::coords::point_t<coor_t>;

    //spade::ctrs::array<real_t, 3> dx = prim.get_grid().get_dx();
    //auto grid = prim.get_grid();
    auto grid_img = grid.image(prim.device());
    auto prim_img = prim.image();
    auto rhs_img  = rhs.image();
    auto inout_img= inout.image();

    auto var_range = spade::dispatch::support_of(prim,spade::grid::exclude_exchanges);

    auto loop = [=] _sp_hybrid (const spade::grid::cell_idx_t& i_cell) mutable
      {
        const point_t x = grid_img.get_coords(i_cell);
        const int inside= inout_img.get_elem(i_cell);
        if (inside==1)
          {
            for (int v=0;v<5;v++)
              {
                rhs_img(v,i_cell) = 0.;
              }
          }
        //for (int dir=0;dir<dim;++dir)
        //  {
        //    //
        //    index_t i_cell_l = i_cell;
        //    i_cell_l.i(dir)--;
        //    index_t i_cell_r = i_cell;
        //    i_cell_r.i(dir)++;
        //
        //    prim_t q_l = prim_img.get_elem(i_cell_l);
        //    prim_t q_c = prim_img.get_elem(i_cell);
        //    prim_t q_r = prim_img.get_elem(i_cell_r);
        //
        //    // left flux:
        //    real_t u_n = 0.5*(q_l.u(dir) + q_c.u(dir));
        //    //conti
        //    real_t mass_flux_l = 0.0; // to-do
        //    real_t mass_flux_r = 0.0; // to-do
        //    rhs_image(0,i_cell) -= ((mass_flux_r - mass_flux_l))/dx[dir];
        //  }
      };
    // spade::dispatch::execute(var_range, loop, prim.device());
    spade::dispatch::execute(var_range, loop);
  }


    
  template <typename arr_t, typename ghost_t, typename xs_t, typename datas_t, typename vector_t>
  static void fill_ghost_vals(arr_t& array, const ghost_t& ghosts, const xs_t& xs, const datas_t& datas, const datas_t& datas2, const vector_t& grad_wm)
    {
        auto geom_img  = array.get_grid().image(spade::partition::local, array.device());
        auto arr_img   = array.image();
        auto ghst_img  = ghosts.image(array.device());
        auto xs_img    = spade::utils::make_vec_image(xs.data(array.device()));
        auto data_img  = spade::utils::make_vec_image(datas);
        auto data2_img = spade::utils::make_vec_image(datas2);
        auto grad_img  = spade::utils::make_vec_image(grad_wm);
        
        using alias_type = typename arr_t::alias_type;
        using real_t     = typename arr_t::fundamental_type;
        
        const auto range = spade::dispatch::ranges::from_array(datas, array.device());

        std::size_t offset = 0;
        std::size_t wm_offset = 0;
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
                    std::size_t idx_wm = wm_offset + idx;
                    const alias_type& sampl_value  = data_img[idx_1d];
                    const alias_type& sampl_value2 = data2_img[idx_1d];
                    const auto& nvec    = list.closest_normals[idx][ilayer];
                    const auto icell    = list.indices[idx][ilayer];
                    const auto& bndy_x  = list.closest_points[idx][ilayer];
                    const auto& ghst_x  = geom_img.get_coords(list.indices[idx][ilayer]);
                    const bool  do_fill = list.can_fill[idx][ilayer];
                    
                    alias_type ghost_value = real_t(0.0);

                    //take pressure and temperature from far point
                    ghost_value.p()   = sampl_value2.p();
                    ghost_value.T()   = sampl_value2.T();

                    bool slipwall = true;
//                    if (ghst_x[0]<0.1&&ghst_x[1]>-0.04)
//                      {
//                        ghost_value.p()   = sampl_value.p();
//                        ghost_value.T()   = sampl_value.T();
//                        slipwall = true;
//                      }

                    using vec_t = spade::ctrs::array<real_t, 3>;
                    vec_t u = {sampl_value2.u(), sampl_value2.v(), sampl_value2.w()};
                    vec_t unorm = spade::ctrs::array_val_cast<real_t>(spade::ctrs::dot_prod(u, nvec)*nvec);
                    vec_t utang = u - unorm;
                    real_t utang_magn = spade::ctrs::array_norm(utang);
                    //        +
                    //        |      |
                    //        |      d1
                    //  ______|____ _|_
                    //        +      d0
                    const auto& sampl_x = xs_img[idx_1d];
                    const real_t d1 = spade::ctrs::array_norm(sampl_x - bndy_x);
                    const real_t d0 = spade::ctrs::array_norm(bndy_x  - ghst_x);

                    // correct tangential velocity to with gradient at image point two
                    real_t samp_grad   = grad_img[idx_wm];
                    real_t utang_magn2 = utang_magn-samp_grad*(d0+d1);
                    //
                    if (slipwall)
                      {
                        for (int i = 0; i < 3; ++i)
                          {
                            ghost_value.u(i) = utang[i];
                          }
                      }
                    else
                      {
                        for (int i = 0; i < 3; ++i)
                          {
                            ghost_value.u(i) += utang[i]*utang_magn2/(utang_magn+1e-6);
                          }
                        for (int i = 0; i < 3; ++i)
                          {
                            ghost_value.u(i) += -(d0/d1)*unorm[i];
                          }
                      }
					//
					if (ilayer==1)
					{
						auto idom = icell;
						int dijk[3]={};
						dijk[dir]=(nvec[dir]>0)? 1 : -1;
						idom.i() = idom.i() + dijk[0];
						idom.j() = idom.j() + dijk[1];
						idom.k() = idom.k() + dijk[2];
                        auto indomain = arr_img.get_elem(idom);
                        ghost_value.u(0) = indomain.u();
                        ghost_value.u(1) = indomain.v();
                        ghost_value.u(2) = indomain.w();
                        ghost_value.p()  = indomain.p();
                        ghost_value.T()  = indomain.T();
					}
					//
                    if (do_fill) arr_img.set_elem(icell, ghost_value);
                });
            };
            
            spade::dispatch::execute(o_range, loop, kpool);
            
            offset    += list_size*nlayers;
            wm_offset += list_size;
        }
        
        //Now do the diagonals
        const auto& list = ghst_img.diags;
            
        std::size_t diag_size = list.indices.size();
        
        auto diag_range = spade::dispatch::ranges::make_range(0UL, diag_size);
        auto diag_loop = [=] _sp_hybrid (const std::size_t& idx) mutable
        {
            std::size_t idx_1d = offset+idx;
            const alias_type& sampl_value2 = data2_img[idx_1d];
            const auto& nvec    = list.closest_normals[idx];
            const auto icell    = list.indices[idx];
            const auto& bndy_x  = list.closest_points[idx];
            const auto& ghst_x  = geom_img.get_coords(list.indices[idx]);
            const bool  do_fill = list.can_fill[idx];
            
            alias_type ghost_value = real_t(0.0);
            ghost_value.p()   = sampl_value2.p();
            ghost_value.T()   = sampl_value2.T();
            
            using vec_t = spade::ctrs::array<real_t, 3>;
            vec_t u = {sampl_value2.u(), sampl_value2.v(), sampl_value2.w()};
            vec_t unorm = spade::ctrs::array_val_cast<real_t>(spade::ctrs::dot_prod(u, nvec)*nvec);
            vec_t utang = u - unorm;
            const auto& sampl_x = xs_img[idx_1d];
            
            for (int i = 0; i < 3; ++i)
            {
                ghost_value.u(i) += utang[i];
            }
            
            //        +
            //        |      |
            //        |      d1
            //  ______|____ _|_
            //        +      d0
    
            const real_t d1 = spade::ctrs::array_norm(sampl_x - bndy_x);
            const real_t d0 = spade::ctrs::array_norm(bndy_x  - ghst_x);
            
            for (int i = 0; i < 3; ++i)
            {
                ghost_value.u(i) += -(d0/d1)*unorm[i];
            }

            // if (do_fill) arr_img.set_elem(icell, ghost_value);
        };
        spade::dispatch::execute(diag_range, diag_loop, array.device());        
    }
}
