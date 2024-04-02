#pragma once

#include "spade.h"

namespace local
{
  template<typename grid_t, typename data_t>
    bool ghost_inside(const grid_t& grid,const data_t& io_data, const int lb)
  {
    for (int k=0;k<grid.get_num_cells(2);++k)
      {
        for (int j=0;j<grid.get_num_cells(1);++j)
          {
            for (int i=0;i<grid.get_num_cells(0);++i)
              {
                spade::grid::cell_idx_t i_cell2(i,j,k,lb);
                int inout = io_data.get_elem(i_cell2);
                if (inout == 1) return true;
              }
          }
      }
    return false;
  }

  template <typename arr_t, typename ghost_t, typename geom_t>
    inline void generate_inout_map(arr_t& inout, const ghost_t& ghost, const geom_t& geom)
    {
      // 1st loop over the ghosts and marm them as inside
      auto inout_img  = inout.image();
      const auto ghst_img = ghost.image(inout.device());
      const auto& grid = inout.get_grid();
      int nx = grid.get_num_cells(0);
      int ny = grid.get_num_cells(1);
      int nz = grid.get_num_cells(2);
      //
      for (int dir = 0; dir < ghst_img.aligned.size(); ++dir)
        {
          const auto list = ghst_img.aligned[dir];
          const auto range = spade::dispatch::ranges::from_array(list.indices, inout.device());
          int nlayer = ghost.num_layers();
          auto inout_ghst = [=] _sp_hybrid (const std::size_t& idx) mutable
            {
              for (int ii = 0; ii < nlayer; ++ii)
                {
                  if (list.can_fill[idx][ii])
                    {
                      auto icell = list.indices[idx][ii];
                      typename arr_t::alias_type inout = 1;
                      inout_img.set_elem(icell, inout);
                    }
                }
            };
          spade::dispatch::execute(range, inout_ghst);
        }
      //
      //now loop over the blocks and mark inside and out side blocks
      //
      auto grid_img = grid.image(inout.device());
      for (int lb=0;lb<grid.get_num_local_blocks();++lb)
        {
          const int nxb = (grid.get_num_cells(0)+1)/2;
          const int nyb = (grid.get_num_cells(1)+1)/2;
          const int nzb = (grid.get_num_cells(2)+1)/2;
          spade::grid::cell_idx_t i_cell(nxb,nyb,nzb,lb);
          auto x = grid_img.get_coords(i_cell);
          //check if there is a ghost point inside
          typename arr_t::alias_type inside = 1;
          if (ghost_inside(grid,inout_img,lb))
            {
              /*
                instead of using inout testing (via ray tracing)
                we could stencil walking but didn't seem to be too
                expensive
              */
              for (int k=0;k<grid.get_num_cells(2);++k)
                {
                  for (int j=0;j<grid.get_num_cells(1);++j)
                    {
                      for (int i=0;i<grid.get_num_cells(0);++i)
                        {
                          spade::grid::cell_idx_t i_cell2(i,j,k,lb);
                          x = grid_img.get_coords(i_cell2);
                          if (geom.is_interior(x)) inout_img.set_elem(i_cell2, inside);
                        }
                    }
                }
            }
          else
            {
              if (geom.is_interior(x))
                {
                  for (int k=0;k<grid.get_num_cells(2);++k)
                    {
                      for (int j=0;j<grid.get_num_cells(1);++j)
                        {
                          for (int i=0;i<grid.get_num_cells(0);++i)
                            {
                              spade::grid::cell_idx_t i_cell2(i,j,k,lb);
                              inout_img.set_elem(i_cell2, inside);
                            }
                        }
                    }
                }
            }
        }
    }
}
