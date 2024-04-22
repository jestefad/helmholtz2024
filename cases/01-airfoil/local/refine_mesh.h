#pragma once

#include <set>

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
    
    template <typename grid_t, typename geom_t, typename coor_t>
    static void refine_by_components(grid_t& grid,
        const geom_t& geom,
        std::vector<int> components,
        std::vector<comp_refine_info_t<coor_t>> refine_infos,
        const spade::ctrs::array<bool, 3>& periodic)
    {
    	std::vector<std::vector<std::size_t>> corresp_faces;
    	corresp_faces.resize(refine_infos.size());
    	for (std::size_t j = 0; j < refine_infos.size(); ++j)
    	{
    		int cid = refine_infos[j].comp_id;
    		auto& vec = corresp_faces[j];
    		for (std::size_t idx = 0; idx < components.size(); ++idx)
    		{
    			if ((cid == components[idx]) || refine_infos[j].any)
    			{
    				vec.push_back(idx);
    			}
    		}
    	}
    	
    	// We will use a BVH to check which block each point is in.
        // Might want to consider pre-computing this somewhere
        constexpr int dim = 3;
        spade::geom::bvh_t<dim, coor_t> block_bvh;
        spade::geom::bvh_params_t bvhparams{3, 1000};
        
        // Perform in computational coordinates, note that we strictly
        // need a 2D BVH if the grid is 2D
        
        const auto dx_max = grid.compute_dx_max();
        const auto dx_min = grid.compute_dx_min();
        
        const auto bbx = grid.get_bounds();
        using bnd_t = spade::bound_box_t<coor_t, dim>;
        bnd_t bnd;
        for (int d = 0; d < dim; ++d)
        {
            bnd.max(d) = bbx.max(d) + 10*dx_max[d];
            bnd.min(d) = bbx.min(d) - 10*dx_max[d];
        }
        
        const auto el_check = [&](const std::size_t& lb_glob, const auto& bnd_in)
        {
            const auto lb = spade::utils::tag[spade::partition::global](lb_glob);
            auto block_box = grid.get_bounding_box(lb);
            const auto dx = grid.get_dx(lb);
            return block_box.intersects(bnd_in);
        };
        
        block_bvh.calculate(bnd, grid.get_num_global_blocks(), el_check, bvhparams);
    	
    	bool finished_refinement = false;
    	
    	static_assert(dim==3, "refine in 3d only");
    	
    	
    	auto lb_dummy = spade::utils::tag[spade::partition::global](0UL);
    	using blk_t = decltype(lb_dummy);
    	using ref_t = spade::ctrs::array<bool, dim>;
    	
    	std::vector<std::set<std::size_t>> finest_levels;
    	finest_levels.resize(refine_infos.size());
    	
    	while (!finished_refinement)
    	{
    		finished_refinement = true;
    		for (int i = 0; i < refine_infos.size(); ++i)
    		{
    			std::map<std::size_t, ref_t> dict;
    			int checked_count = 0;
    			for (int j = 0; j < corresp_faces[i].size(); ++j)
    			{
    				std::size_t f = corresp_faces[i][j];
    				spade::ctrs::array<coor_t, 3> req_spacing;
    				int d = refine_infos[i].axis;
    				int d0 = d  + 1;
    				int d1 = d0 + 1;
    				d0 %= dim;
    				d1 %= dim;
    				
    				req_spacing[d]  = refine_infos[i].dx_target;
    				req_spacing[d0] = req_spacing[d]*refine_infos[i].aspect[d0]/refine_infos[i].aspect[d];
    				req_spacing[d1] = req_spacing[d]*refine_infos[i].aspect[d1]/refine_infos[i].aspect[d];
    				
    				const auto xc = geom.centroid(f);
    				const auto eval = [&](const std::size_t& lb_cand)
            		{
                		auto lb_tagged = spade::utils::tag[spade::partition::global](lb_cand);
                		const auto block_box = grid.get_bounding_box(lb_tagged);
                		if (block_box.contains(xc))
                		{
                			++checked_count;
                			const auto dx = grid.get_dx(lb_tagged);
                			ref_t ref_dir = false;
                			for (int d = 0; d < 3; ++d)
                			{
                				ref_dir[d] = dx[d] > req_spacing[d];
                			}
                			if (ref_dir[0] || ref_dir[1] || ref_dir[2])
                			{
                				if (dict.find(lb_tagged.value) == dict.end())
                				{
                					dict[lb_tagged.value] = ref_dir;
                				}
                				else
                				{
                					ref_t last = dict[lb_tagged.value];
                					for (int d = 0; d < 3; ++d)
                					{
                						last[d] = last[d] || ref_dir[d];
                					}
                					dict[lb_tagged.value] = last;
                				}
                			}
                			else
                			{
                				finest_levels[i].insert(lb_tagged.value);
                			}
                		}
            		};
            		
            		spade::bound_box_t<coor_t, 3> tri_bnd;
            		auto x0 = geom.points[geom.faces[f][0]];
            		auto x1 = geom.points[geom.faces[f][1]];
            		auto x2 = geom.points[geom.faces[f][2]];
            		
            		tri_bnd.min(0) = spade::utils::min(x0[0], x1[0], x2[0]);
            		tri_bnd.min(1) = spade::utils::min(x0[1], x1[1], x2[1]);
            		tri_bnd.min(2) = spade::utils::min(x0[2], x1[2], x2[2]);
            		
            		tri_bnd.max(0) = spade::utils::max(x0[0], x1[0], x2[0]);
            		tri_bnd.max(1) = spade::utils::max(x0[1], x1[1], x2[1]);
            		tri_bnd.max(2) = spade::utils::max(x0[2], x1[2], x2[2]);
            		
    				block_bvh.check_elements(eval, tri_bnd);
    			}
    			
    			std::vector<blk_t> to_refine;
    			std::vector<ref_t> refine_dirs;
    			
    			for (const auto& p: dict)
    			{
    				to_refine.push_back(spade::utils::tag[spade::partition::global](p.first));
    				refine_dirs.push_back(p.second);
    			}
    			
				if (to_refine.size() > 0)
    			{
    				for (auto& v: finest_levels) v.clear();
    				finished_refinement = false;
    				grid.refine_blocks(to_refine, periodic, refine_dirs, spade::amr::constraints::factor2);
    				block_bvh.calculate(bnd, grid.get_num_global_blocks(), el_check, bvhparams);
    			}
    		}
    	}
    	
    	std::map<std::size_t, ref_t> neigh_refs_map;
    	
    	for (int i = 0; i < refine_infos.size(); ++i)
    	{
    		const auto& infos = refine_infos[i];
    		const auto& fblks = finest_levels[i];
    		
    		for (const std::size_t lb_val: fblks)
    		{
    			const auto& self = grid.get_blocks().get_amr_node(lb_val);
    			const auto& neighs = grid.get_blocks().get_neighs(lb_val);
            	bool required = false;
            	ref_t refine_type = false;
            	for (const auto& e: neighs)
            	{
            		const auto& neigh = *e.endpoint;
            		const std::size_t neigh_lb = neigh.tag;
            		const auto& my_level = self.level;
            		const auto& ng_level = neigh.level;
                	for (int d = 0; d < grid.dim(); ++d)
                	{
                    	refine_type[d] = ng_level[d] < my_level[d];
                	}
                	
                	const auto bnd = grid.get_bounding_box(spade::utils::tag[spade::partition::global](neigh_lb));
            		const auto xc = spade::ctrs::array_cast<spade::coords::point_t<coor_t>>(bnd.center());
            		bool interior = geom.is_interior(xc);
            		if (!interior && (refine_type[0] || refine_type[1] || refine_type[2]))
            		{
            			if (neigh_refs_map.find(neigh_lb) == neigh_refs_map.end())
            			{
            				neigh_refs_map[neigh_lb] = refine_type;
            			}
            			else
            			{
            				ref_t old = neigh_refs_map[neigh_lb];
            				for (int d = 0; d < 3; ++d)
            				{
            					old[d] = old[d] || refine_type[d];
            				}
            				neigh_refs_map[neigh_lb] = old;
            			}
            		}
            	}
    		}
    	}
    	
    	//Refine the neighbors of the finest-level blocks
    	std::vector<blk_t> n_refines;
    	std::vector<ref_t> n_direcs;
    	
    	for (const auto& p: neigh_refs_map)
    	{
    		n_refines.push_back(spade::utils::tag[spade::partition::global](p.first));
    		n_direcs.push_back(p.second);
    	}
    	
    	grid.refine_blocks(n_refines, periodic, n_direcs, spade::amr::constraints::factor2);
    }
}
