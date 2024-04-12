#pragma once

namespace local
{
    template <typename arr_t, typename geom_t>
    inline void vtkout(const std::string& fname, const arr_t& arr, const geom_t& geom)
    {
        print("Output vtk...");
        using real_t = typename arr_t::value_type;
        using coor_t = typename geom_t::value_type;
        using pnt_t = spade::coords::point_t<coor_t>;
        
        const auto dxminv = arr.get_grid().compute_dx_min();
        const auto dxmin  = spade::utils::min(dxminv[0], dxminv[1], dxminv[2]);
        const auto bnd    = arr.get_grid().get_blocks().get_bounds();
        
        std::vector<pnt_t> sampl_points;
        const coor_t sampl_dist = 1.5;
        std::vector<std::size_t> ood_mask;
        
        for (std::size_t i = 0; i < geom.faces.size(); ++i)
        {
            const pnt_t ctr = geom.centroid(i);
            const auto nv   = geom.normals[i];
            auto x = ctr;
            x += sampl_dist*dxmin*nv;
            if (!bnd.contains(x))
            {
                const auto xx = bnd.center();
                x[0] = xx[0];
                x[1] = xx[1];
                x[2] = xx[2];
                ood_mask.push_back(i);
            }
            sampl_points.push_back(x);
        }
        
        auto sampl_oper = spade::sampling::create_interpolation(arr, sampl_points, spade::sampling::multilinear);
        auto valsg       = spade::sampling::sample_array(arr, sampl_oper);
        std::vector<typename arr_t::alias_type> vals = valsg;
        for (const auto ii: ood_mask)
        {
            vals[ii] = 0.0;
        }
        
        std::ofstream mf(fname);
        mf << "# vtk DataFile Version 3.0\nvtk output\nASCII\nDATASET POLYDATA\nPOINTS " << geom.points.size() << " double\n";
        for (std::size_t i = 0; i < geom.points.size(); ++i)
        {
            mf << geom.points[i][0] << " ";
            mf << geom.points[i][1] << " ";
            mf << geom.points[i][2] << "\n";
        }
        mf << "POLYGONS " << geom.faces.size() << " " << 4*geom.faces.size() << "\n";
        for (std::size_t i = 0; i < geom.faces.size(); ++i)
        {
            mf << "3 " << geom.faces[i][0] << " " << geom.faces[i][1] << " " << geom.faces[i][2] << "\n";
        }
        mf << "CELL_DATA " << geom.faces.size() << "\n";
        for (int v = 0; v < arr_t::alias_type::size(); ++v)
        {
            const std::string scname = arr_t::alias_type::name(v);
            mf << "SCALARS " << scname << " double\nLOOKUP_TABLE default\n";
            for (std::size_t i = 0; i < geom.faces.size(); ++i)
            {
                const auto& q = vals[i];
                mf << q[v] << "\n";
            }
        }
    }
}