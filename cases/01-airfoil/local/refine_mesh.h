#pragma once

namespace local
{
    template <typename grid_t, typename geom_t, typename coor_t>
    inline void refine_mesh(
        grid_t& grid,
        const geom_t& geom,
        const coor_t& dx_bndy,
        int dir,
        const spade::ctrs::array<coor_t, 3>& aspect,
        const spade::ctrs::array<bool,   3>& periodic)
    {
        const auto dxs = grid.get_dx(0UL);
        const auto get_num_levels      = [](const auto& dx, const auto& dx_t) {return ceil (log(dx/dx_t)/log(2.0));};
        const auto get_num_levels_near = [](const auto& dx, const auto& dx_t) {return round(log(dx/dx_t)/log(2.0));};
        int num_level = get_num_levels(dxs[dir], dx_bndy);
        int dir0 = (dir + 1) % grid.dim();
        int dir1 = (dir + 2) % grid.dim();
        
        coor_t dx0 = dx_bndy*aspect[dir0]/aspect[dir];
        coor_t dx1 = dx_bndy*aspect[dir1]/aspect[dir];
        
        int n0 = get_num_levels_near(dxs[dir0], dx0);
        int n1 = get_num_levels_near(dxs[dir1], dx1);
        
        using refine_t = spade::ctrs::array<bool, 3>;
        for (int l = 0; l < num_level; ++l)
        {
            refine_t refine_type = true;
            refine_type[dir]  = true;
            refine_type[dir0] = (l < n0);
            refine_type[dir1] = (l < n1);
            const auto bndy_intersect = [&](const auto& lb)
            {
                auto bnd = grid.get_bounding_box(lb);

				//cbrehm1-s
				if (l<=num_level-3)
				{
					const auto xc = spade::ctrs::array_cast<spade::coords::point_t<coor_t>>(bnd.center());
					bool interior = geom.is_interior(xc);

					if (!interior&&xc[1]>0&&xc[0]>0.)
					{
						for (int d=0; d<3; ++d)
						{
							bnd.min(d)-=0.25;
							bnd.max(d)+=0.25;
						}
					}
				}
				else if (l==num_level-2)
				{
					const auto xc = spade::ctrs::array_cast<spade::coords::point_t<coor_t>>(bnd.center());
					bool interior = geom.is_interior(xc);

					if (!interior&&xc[1]>0&&xc[0]>0.0)
					{
						for (int d=0; d<3; ++d)
						{
							bnd.min(d)-=0.14;
							bnd.max(d)+=0.14;
						}
					}
				}
				else if (l==num_level-1)
				{
					const auto xc = spade::ctrs::array_cast<spade::coords::point_t<coor_t>>(bnd.center());
					bool interior = geom.is_interior(xc);

					if (!interior&&xc[1]>0&&xc[0]>0.1)
					{
						for (int d=0; d<3; ++d)
						{
							bnd.min(d)-=0.023;
							bnd.max(d)+=0.023;
						}
					}
				}
				//cbrehm1-e
                return geom.box_contains_boundary(bnd);
            };
            auto rblks = grid.select_blocks(bndy_intersect, spade::partition::global);
            grid.refine_blocks(rblks, periodic, refine_type, spade::amr::constraints::factor2);
        }
        
        const auto& blocks = grid.get_blocks();
        const auto neigh_select = [&](const auto& lb)
        {
            const auto& neighs = blocks.get_neighs(lb.value);
            bool required = false;
            refine_t refine_type = false;
            for (const auto& e: neighs)
            {
                int neigh_level = e.endpoint->level[dir];
                int my_level    = grid.get_blocks().get_amr_node(lb.value).level[dir];
                required = required || ((neigh_level == num_level) && (my_level < num_level));
                for (int d = 0; d < grid.dim(); ++d)
                {
                    refine_type[d] = refine_type[d] || (e.endpoint->level[d] > grid.get_blocks().get_amr_node(lb.value).level[d]);
                }
            }
            const auto bnd = grid.get_bounding_box(lb);
            const auto xc = spade::ctrs::array_cast<spade::coords::point_t<coor_t>>(bnd.center());
            bool interior = geom.is_interior(xc);
            return std::make_tuple(required && !interior, refine_type);
        };
        
        using glob_t = decltype(spade::utils::tag[spade::partition::global](0UL));
        std::vector<glob_t> globs;
        std::vector<refine_t> refs;
        for (std::size_t lb = 0; lb < grid.get_num_global_blocks(); ++lb)
        {
            auto lb_glob = spade::utils::tag[spade::partition::global](lb);
            const auto [reqd, ref] = neigh_select(lb_glob);
            if (reqd)
            {
                globs.push_back(lb_glob);
                refs.push_back(ref);
            }
        }
        grid.refine_blocks(globs, periodic, refs, spade::amr::constraints::factor2);
    }
}
