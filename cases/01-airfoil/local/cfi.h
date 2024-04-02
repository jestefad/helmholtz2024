#pragma once

namespace local
{
    template <typename val_t> using triple_array = spade::ctrs::array<spade::ctrs::array<spade::ctrs::array<val_t, 3>, 3>, 3>;
    template <typename float_t> struct cfi_diss_t
    : public spade::state_sensor::sensor_interface_t<cfi_diss_t<float_t>, spade::omni::info_list_t<spade::omni::info::index>>
    {
        using base_t = spade::state_sensor::sensor_interface_t<cfi_diss_t<float_t>, spade::omni::info_list_t<spade::omni::info::index>>;
        using base_t::get_sensor;
        using base_t::info_type;

        float_t maxval, minval, radius;
        spade::ctrs::array<int, 3> nx;

        spade::utils::const_vec_image_t<triple_array<bool>> bbxs;

    cfi_diss_t(const auto& ivec, float_t minval_in, float_t maxval_in, float_t radius_in, spade::ctrs::array<int, 3> nx_in)
      : bbxs{spade::utils::make_vec_image(ivec)}, minval{minval_in}, maxval{maxval_in}, radius{radius_in}, nx{nx_in} {}

        _sp_hybrid float_t get_sensor(const auto& info) const
        {
            const auto& idx    = spade::omni::access<spade::omni::info::index>(info);
            const auto& bbx    = bbxs[idx.lb()];
            const auto  icoord = spade::grid::get_index_coord<float_t>(idx);
            float_t output = float_t(0.0);

            spade::bound_box_t<float_t, 3> diss_bound;

            const auto get_min = [&](const int idir, const int didx)
            {
                spade::ctrs::array<float_t, 3> low{float_t(-100.0), float_t(-100.0), float_t(nx[idir])-radius};
                return low[1+didx];
            };

            const auto get_max = [&](const int idir, const int didx)
            {
                spade::ctrs::array<float_t, 3> hi{radius, float_t(100.0), float_t(100.0)};
                return hi[1+didx];
            };


            for (int dk = -1; dk <= 1; ++dk)
            {
                diss_bound.min(2) = get_min(2, dk);
                diss_bound.max(2) = get_max(2, dk);
                for (int dj = -1; dj <= 1; ++dj)
                {
                    diss_bound.min(1) = get_min(1, dj);
                    diss_bound.max(1) = get_max(1, dj);
                    for (int di = -1; di <= 1; ++di)
                    {
                        diss_bound.min(0) = get_min(0, di);
                        diss_bound.max(0) = get_max(0, di);
                        bool diss_here  = bbx[1+di][1+dj][1+dk];
                        float_t val_loc = float_t(int(diss_here && diss_bound.contains(icoord)));
                        if ((di != 0) || (dj != 0) || (dk != 0)) output += val_loc;
                    }
                }
            }

            output = spade::utils::min(output, float_t(1.0));
            return (float_t(1.0)-output)*minval+output*maxval;
        }
    };

    template <typename grid_t>
    requires(grid_t::dim() == 3)
    inline spade::device::shared_vector<triple_array<bool>>
      compute_cfi_vals(const grid_t& grid, int shielded_num=1)
    {
        using output_t = spade::device::shared_vector<triple_array<bool>>;
        output_t output;
        spade::ctrs::array<int, 3> maxlevel{0, 0, 0};

        for (int lb = 0; lb < grid.get_num_local_blocks(); ++lb)
          {
            const auto lb_tg   = spade::utils::tag[spade::partition::local](lb);
            const auto lb_glob = grid.get_partition().to_global(lb_tg);

            const auto& my_node = grid.get_blocks().get_amr_node(lb_glob.value);

            for (int dir = 0; dir < 3; ++dir)
              {
                maxlevel[dir] = spade::utils::max(maxlevel[dir], my_node.level[dir]);
              }
        }

        for (int lb = 0; lb < grid.get_num_local_blocks(); ++lb)
          {
            const auto lb_tg   = spade::utils::tag[spade::partition::local](lb);
            const auto lb_glob = grid.get_partition().to_global(lb_tg);
            triple_array<bool> val;
            val[1][1][1] = false;
            const auto& my_node = grid.get_blocks().get_amr_node(lb_glob.value);
            const auto& neighs  = grid.get_blocks().get_neighs(lb_glob.value);
            for (const auto& e:neighs)
            {
                const auto& neigh_node = *(e.endpoint);
                int dx = 1+e.edge[0];
                int dy = 1+e.edge[1];
                int dz = 1+e.edge[2];

                const auto my_level    = my_node.level;
                const auto neigh_level = neigh_node.level;

                //IMPORTANT PART FOR IDENTIFYING CFI
                val[dx][dy][dz] = false;

                for (int i = 0; i < 3; ++i)
                {
                    bool add_diss_here = my_level[i] != neigh_level[i];
                    val[dx][dy][dz] = val[dx][dy][dz] || (add_diss_here);
                }
            }
            bool full_diss_here = false;
            for (int dir = 0; dir < 3; ++dir)
              {
                if (maxlevel[dir] - my_node.level[dir] > shielded_num) full_diss_here = true;
              }
            if (full_diss_here) val = true;
            output.push_back(val);
        }

        output.transfer();
        return output;
    }
}
