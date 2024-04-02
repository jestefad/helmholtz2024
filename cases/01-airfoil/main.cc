#include <stdio.h>
#include <vector>
#include "print_type.h"

#include "scidf.h"
#include "spade.h"
#include <time.h>
#include "sample_cloud.h"
#include "fill_ghosts.h"
#include "surf_vtk.h"
#include "vtkout.h"
#include "compute_wm.h"
#include "compute_irreg_visc.h"
#include "compute_irreg_conv.h"
#include "profile_reader.h"
#include "refine_mesh.h"
#include "inout.h"
#include "restart.h"
#include "cfi.h"

using real_t = float;
using coor_t = double;
using flux_t = spade::fluid_state::flux_t<real_t>;
using prim_t = spade::fluid_state::prim_t<real_t>;
using cons_t = spade::fluid_state::cons_t<real_t>;

int main(int argc, char** argv)
{
    std::vector<std::string> args;
    for (auto i: range(0, argc)) args.push_back(std::string(argv[i]));
    std::string input_filename = "input.sdf";
    if (args.size() > 1)
    {
        input_filename = args[1];
    }
    
    //input reader ---> let's call champs_init
    scidf::node_t input;
    scidf::clargs_t clargs(argc, argv);
    scidf::read(input_filename, input, clargs);

    //select fluid model
    spade::fluid_state::ideal_gas_t<real_t> air;
    air.gamma = input["Fluid"]["gamma"];
    air.R     = input["Fluid"]["Rgas"];

    //set initial conditions
    const real_t mach = input["Fluid"]["mach"];
    const real_t aoa  = input["Fluid"]["aoa"];

    //time-integration interval, inital condition, CFL, ...
    const std::string init_file    = input["Config"]["init_file"];
    const real_t      targ_cfl     = input["Config"]["cfl"];
    const int         nt_max       = input["Config"]["nt_max"];
    const int         nt_skip      = input["Config"]["nt_skip"];
    const int         nt_surf      = input["Config"]["nt_surf"];
    const bool       do_restart    = input["Config"]["do_restart"];
	const int        nt_restart    = (do_restart) ? input["Config"]["nt_restart"] : 0;
    const int        restart_skip  = input["Config"]["restart_skip"];
    const bool       do_output     = input["Config"]["do_output"];
    const std::vector<int> devices = input["Config"]["devices"];
    
    const bool        output_rhs      = input["Debug"]["output_rhs"];
    const std::string output_rhs_name = input["Debug"]["output_rhs_name"];
    const bool        print_perf      = input["Debug"]["print_perf"];

    //domain setup
    spade::ctrs::array<int,    3> num_blocks     = input["Grid"]["nblck"];
    spade::ctrs::array<int,    3> num_cells      = input["Grid"]["ncell"];
    spade::ctrs::array<int,    3> exchange_cells = input["Grid"]["nexch"];
    spade::ctrs::array<bool,   3> periodic       = input["Grid"]["periodic"];
    
    spade::ctrs::array<coor_t, 6> bnd            = input["Grid"]["bounds"];
    spade::bound_box_t<coor_t, 3> bounds;
    bounds.bnds = bnd;

    //geometry file name
    const std::string geom_fname  = input["Grid"]["geom"];
    const int         maxlevel    = input["Grid"]["maxlevel"];
    const coor_t      sampl_dist  = input["Grid"]["sampl_dist"];
    const coor_t      sampl_dist2 = input["Grid"]["sampl_dist2"];
    const coor_t  out_sampl_dist  = input["Grid"]["out_sampl_dist"];
    const bool       is_external  = input["Grid"]["is_external"];

    //refinement
    const coor_t                   dx_boundary = input["Refine"]["dx_boundary"];
    const int                     dx_direction = input["Refine"]["dx_direction"];
    const spade::ctrs::array<coor_t, 3> aspect = input["Refine"]["aspect"];

    //Creates an identity matrix
    spade::coords::identity<coor_t> coords;

    //output path
    std::filesystem::path out_path("surface");
    if (!std::filesystem::is_directory(out_path)) std::filesystem::create_directory(out_path);
    
    spade::parallel::compute_env_t env(&argc, &argv, devices);
    env.exec([&](spade::parallel::pool_t& group)
    {
        //represents the arrangement of blocks in 3-D space
        spade::amr::amr_blocks_t blocks(num_blocks, bounds);
    
        //create the Cartesian grid with specified blocks arrangement and coordinate system, MPI group)
        if (group.isroot()) print("Create grid");
        spade::grid::cartesian_grid_t grid(num_cells, blocks, coords, group);
        if (group.isroot()) print("Done");
    
        
        // read the geometry file
        if (group.isroot()) print("Read", geom_fname);
        spade::geom::vtk_geom_t<3, 3, coor_t> geom;
        spade::geom::read_vtk_geom(geom_fname, geom);
        if (group.isroot()) print("Done");
    
        //array of boolean
        using refine_t = typename decltype(blocks)::refine_type;
        //controlling refinement in (x,y,z)-directions
        refine_t ref0  = {true, true, true};
    
        if (group.isroot()) print("Begin refine");

        const bool boundary_refinement = true;
        if (boundary_refinement) local::refine_mesh(grid,geom,dx_boundary,dx_direction,aspect,periodic);

		if (group.isroot()) spade::io::output_vtk("debug/grid.vtk", grid);

//        //iteratively refine the mesh
//        int iter = 0;
//        for (int l=0;l<maxlevel;++l)
//          {
//            const auto select_func = [&](const auto& lb)
//            {
//              const auto bnd = grid.get_bounding_box(lb);
//              const real_t xmin = bnd.min(0);
//              const real_t xmax = bnd.max(0);
//              const real_t ymin = bnd.min(1);
//              const real_t ymax = bnd.max(1);
//              return (ymin<0.01);
//            };
//            auto rblks = grid.select_blocks(select_func, spade::partition::global);
//            grid.refine_blocks(rblks, periodic);
//          }
        
        if (group.isroot()) print("Done");
        if (group.isroot()) print("Num. points:", grid.get_grid_size());
        
        //seems to be figuring out the grid spacing to used at the wall
        const auto dxs  = grid.compute_dx_min();
        const real_t dx = spade::utils::min(dxs[0], dxs[1], dxs[2]);
    
        //initialize primitive variable and fluxes
        prim_t fill1 = 0.0;
        flux_t fill2 = 0.0;
        spade::grid::grid_array prim (grid, fill1, exchange_cells, spade::device::best, spade::mem_map::tiled);
        
        //Note that we don't use exchange cells for the residual array
        spade::grid::grid_array rhs  (grid, fill2, {0, 0, 0}, spade::device::best, spade::mem_map::tiled);

        //
        const real_t Tinf  = 300.0;
        const real_t Pinf  = 101327.;
        const real_t rhoInf= Pinf/(Tinf*air.R);
        //const real_t mu = 6.532698816e-4;
        const real_t mu = 6.532698816e-5;
        //const real_t mu = 6.094754e-6;
        const real_t pr = 0.71;
        const real_t Uinf  = mach*sqrt(air.gamma*air.R*Tinf);
        const real_t theta = spade::consts::pi*aoa/180.0;
        constexpr real_t pi = spade::consts::pi;
        const real_t uu = Uinf*cos(theta);
        const real_t vv = Uinf*sin(theta);
        const real_t ww = 0.0;
        
        //set convective schemes
    
        const auto s0 = spade::convective::cent_keep<4>(air);
        //const auto tscheme = spade::convective::cent_keep<2>(air);
        //spade::convective::rusanov_t flx(air);
        spade::convective::fweno_t s1(air);
        //spade::convective::weno_t<decltype(flx), spade::convective::disable_smooth> tscheme(flx);
        // spade::state_sensor::ducros_t ducr(1.0);
        spade::state_sensor::const_sensor_t const_sens(real_t(8e-2));
        //spade::convective::hybrid_scheme_t tscheme(s0, s1, ducr);

        auto cfi_vals = local::compute_cfi_vals(grid);
        //const real_t alphamin = 5e-4;
		const real_t alphamin = 5e-1;
        const real_t alphamax = 5e-1;
        local::cfi_diss_t<real_t> cfi_diss(cfi_vals.data(prim.device()), alphamin, alphamax, real_t(3.0), num_cells);
        //spade::convective::hybrid_scheme_t tscheme(s0, s1, cfi_diss);
		// spade::convective::hybrid_scheme_t tscheme(s0, s1, cfi_diss);
		spade::convective::hybrid_scheme_t tscheme(s0, s1, const_sens);

        //const auto tscheme = spade::convective::cent_keep<2>(air);
        //spade::convective::first_order_t tscheme(air);
        //ini Wale model
        const real_t cw    = 0.55;
        const real_t delta = 2e-4;
        const real_t pr_t  = 0.9;
        spade::subgrid_scale::wale_t eddy_visc(air, cw, delta, pr_t);
    
        //set fluid properties
        spade::viscous_laws::constant_viscosity_t<real_t> visc(mu, pr);
        
        spade::viscous_laws::sgs_visc_t sgs_visc(visc, eddy_visc);
        //spade::viscous::visc_lr vscheme(visc, air);
        spade::viscous::visc_lr vscheme(sgs_visc, air);

        //computing the ghost points
        if (group.isroot()) print("Compute ghosts");
        // Old version
        // const auto ghosts = spade::ibm::compute_ghosts(prim, grid, geom);
        // New version
        const auto ghosts = spade::ibm::compute_boundary_info(prim, geom, spade::omni::compose(tscheme, vscheme), is_external);
        if (group.isroot()) print("Done");
        
        if (group.isroot()) spade::io::mkdir("debug");
        group.sync();
        std::string file_title = "boundary.pid." + std::to_string(group.rank());
        if (group.isroot()) print("Output ghosts");
        spade::io::output_vtk("debug", file_title, ghosts, grid);
        if (group.isroot()) print("Done");
        
        if (group.isroot()) spade::io::output_vtk("debug/grid.vtk", grid);
        
        //computing the image points for the ghosts --> this is happening on the CPU?
        if (group.isroot()) print("Compute ips");
        auto ips = spade::ibm::compute_ghost_sample_points(ghosts, grid, sampl_dist*dx);
        //compute a second image point
        auto ips2 = spade::ibm::compute_ghost_sample_points(ghosts, grid, sampl_dist2*dx);
        spade::io::output_vtk("debug/ips.vtk" , ips);
        spade::io::output_vtk("debug/ips2.vtk", ips2);
        if (group.isroot()) print("Done");
        
        //establish exchange routines
        if (group.isroot()) print("Compute exchg");
        auto handle = spade::grid::make_exchange(prim, periodic);
        if (group.isroot()) print("Done");
        //
        int fill3=0;
        spade::grid::grid_array inout_cpu(grid, fill3, {2, 2, 2}, spade::device::cpu);
        spade::grid::grid_array inout_gpu(grid, fill3, {2, 2, 2}, spade::device::best);
        // compute inout map
        local::generate_inout_map(inout_cpu,ghosts,geom);
        //
        inout_gpu.data = inout_cpu.data;
        //
        spade::io::output_vtk("debug","inout",inout_gpu);
        //
        //group.pause();
        //
        //create interpolation operators
        const auto exclude = [&](const spade::grid::cell_idx_t& ii) { return geom.is_interior(grid.get_coords(ii)); };
        
        //const auto strategy = spade::sampling::prioritize(spade::sampling::multilinear, spade::sampling::nearest);
        const auto strategy = spade::sampling::prioritize(spade::sampling::multilinear, spade::sampling::nearest, spade::sampling::force_nearest);
        const auto strategy2= spade::sampling::prioritize(spade::sampling::wlsqr,spade::sampling::multilinear, spade::sampling::nearest, spade::sampling::force_nearest);
        
        if (group.isroot()) print("Compute interp 1");
        auto interp  = [&](){
            try
            {
                auto tmp = spade::sampling::create_interpolation(prim, ips,  strategy,  exclude);
                return tmp;
            }
            catch (const spade::except::points_exception<coor_t>& e)
            {
                spade::io::output_vtk("debug/failed1." + std::to_string(group.rank()) + ".vtk", e.data);
                throw e;
            }
        }();
        
        if (group.isroot()) print("Compute interp 2");
        auto interp2  = [&](){
            try
            {
                auto tmp = spade::sampling::create_interpolation(prim, ips2,  strategy2,  exclude);
                return tmp;
            }
            catch (const spade::except::points_exception<coor_t>& e)
            {
                spade::io::output_vtk("debug/failed2." + std::to_string(group.rank()) + ".vtk", e.data);
                throw e;
            }
        }();
        if (group.isroot()) print("Done");
        //std::cin.get();
        // auto surf_cloud = local::get_surf_sampl(geom, out_sampl_dist*dx);
        // auto surf_sampl = spade::grid::create_interpolation(prim, surf_cloud);
    
        //initial condition should be moved to the ini routine
        if (group.isroot())
          {
            print("Uinf  =", Uinf);
            print("Pinf  =", Pinf);
            print("Tinf  =", Tinf);
            print("Rinf  =", rhoInf);
            print("Re_inf=",Uinf*rhoInf/mu);
          }
        //
        //
        auto ini = [=] _sp_hybrid (const spade::coords::point_t<coor_t>& x, const spade::grid::cell_idx_t& ii)
        {
            prim_t output;
            output.p() = Pinf;
            output.T() = Tinf;
            output.u() = uu;
            output.v() = vv;
            output.w() = 0.0;
            
            // output.p() = Pinf + 10.0*(sin(x[0])+cos(x[1])*cos(x[2]));
            // output.T() = ii.lb();
            // output.u() = ii.i();
            // output.v() = ii.j();
            // output.w() = ii.k();
            return output;
        };

        spade::algs::fill_array(prim, ini);

        if (do_restart)
          {
            prim_t fill_tmp=0;
            std::string nstr = spade::utils::zfill(nt_restart, 8);
            std::string filename = "restart/restart_"+nstr + ".sol";
            if (group.isroot()) print("read restart file...",filename);
            spade::grid::grid_array restart_data(grid, fill_tmp, {2, 2, 2}, spade::device::cpu);
            //restart_data.data = time_int.solution().data;
            local::read_restart(filename,restart_data);
            prim.data = restart_data.data;
          }


        //interpolate into the image points
        if (group.isroot()) print("Sample data");
        auto sampldata  = spade::sampling::sample_array(prim, interp);
        auto sampldata2 = spade::sampling::sample_array(prim, interp2);
        if (group.isroot()) print("Done");
    
        //read ini condition from a file
        if (init_file != "none")
        {
            if (group.isroot()) print("reading...");
            // spade::io::binary_read(init_file, prim);
            if (group.isroot()) print("Init done.");
            handle.exchange(prim, group);
        }
        
        real_t time0 = 0.0;
        
        //create a reduce operator (to compute global CFL)
        //creating a lambda to compute the |u|+c (currently not used)
        const auto   sigma_func = [=] _sp_hybrid (const prim_t& q) { return sqrt(q.u()*q.u() + q.v()*q.v() + q.w()*q.w()) + sqrt(air.gamma*air.R*q.T()); };
        const auto   get_sigma  = spade::algs::make_reduction(prim, sigma_func, spade::algs::max);
        const real_t sigma_ini  = spade::algs::transform_reduce(prim, get_sigma);
        const real_t dt         = targ_cfl*dx/sigma_ini;
        //
        print("[I] D-CFL = ",dt/(dx*dx)*mu);
        print("[I] dx    = ",dx);
        //
        //defines the prims2cons/cons2prim for the time-integrator, passed into integrator_t
        cons_t transform_state;
        spade::fluid_state::state_transform_t trans(transform_state, air, spade::grid::exclude_exchanges);
        //
        using vec_t = spade::device::auto_vector<real_t, decltype(prim.device())>;
        vec_t tau_w, q_w, grad_w;
        std::size_t num_tau_w = 0;
        num_tau_w += ghosts.aligned[0].indices.size();
        num_tau_w += ghosts.aligned[1].indices.size();
        num_tau_w += ghosts.aligned[2].indices.size();
        //
        tau_w.resize(num_tau_w);
        q_w.resize(num_tau_w);
        grad_w.resize(num_tau_w);
        //
        //create a lambda for rhs evaluation
        const auto srcc = [=] _sp_hybrid (const prim_t& q) { return flux_t{0.0, 0.0, 9.86, 0.0, 0.0}; };
        auto calc_rhs = [&](auto& resid, const auto& sol, const auto& t)
        {
            resid = 0.0;
            spade::timing::tmr_t t0;
            t0.start();
            spade::pde_algs::flux_div(sol, resid, spade::omni::compose(tscheme, vscheme), spade::pde_algs::ldbalnp);
            t0.stop();
            //spade::pde_algs::source_term(sol, resid, srcc);
            //local::rhs_keep(sol,resid,air);
            spade::timing::tmr_t t1;
            t1.start();
            local::rhs_irreg_visc(sol,resid,air,visc,vscheme,ghosts,sampldata2,ips,tau_w,q_w);
            t1.stop();
            
            spade::timing::tmr_t t2;
            t2.start();
            local::rhs_irreg_conv(sol,resid,air,tscheme,ghosts,sampldata2,ips,tau_w,q_w);
            t2.stop();
            //local::rhs_keep_irreg();
            spade::timing::tmr_t t3;
            t3.start();
            local::zero_ghost_rhs(resid, ghosts);            
            t3.stop();
            
            spade::timing::tmr_t t4;
            t4.start();
            local::zero_rhs_inside(sol,grid,resid,air,inout_gpu);
            t4.stop();
            
            if (group.isroot() && print_perf)
            {
                print("============================================== RHS ==============================================");
                std::string fmt0;
                int nn = 20;
                fmt0 += spade::utils::pad_str("flux_div", nn);
                fmt0 += spade::utils::pad_str("irr_visc", nn);
                fmt0 += spade::utils::pad_str("irr_conv", nn);
                fmt0 += spade::utils::pad_str("zero_ghs", nn);
                fmt0 += spade::utils::pad_str("zero_rhs", nn);
                
                std::string fmt1;
                fmt1 += spade::utils::pad_str(t0.duration(), nn);
                fmt1 += spade::utils::pad_str(t1.duration(), nn);
                fmt1 += spade::utils::pad_str(t2.duration(), nn);
                fmt1 += spade::utils::pad_str(t3.duration(), nn);
                fmt1 += spade::utils::pad_str(t4.duration(), nn);
                
                print(fmt0);
                print(fmt1);
                print();
            }
            
            if (output_rhs)
            {
                if (group.isroot()) print("Output RHS");
                spade::io::output_vtk("output", output_rhs_name, resid);
                if (group.isroot()) print("Finished");
                group.sync();
                abort();
            }
        };
        
        using parray_t = decltype(prim);
    
        //create domain boundary conditions: 1) symmetry, freestream,
        const auto symmetry = [=] _sp_hybrid (const prim_t& q, const int dir)
        {
          prim_t q2 = q;
          q2.u(dir) = -q2.u(dir);
          return q2;
        };

        //create domain boundary conditions: 1) symmetry, freestream,
        const auto noslip = [=] _sp_hybrid (const prim_t& q, const int dir)
        {
          prim_t q2 = q;
          q2.u() = -q2.u();
          q2.v() = -q2.v();
          q2.w() = -q2.w();
          return q2;
        };
        //

        unsigned int m_w = 150;
        unsigned int m_z = 40;
        spade::device::shared_vector<unsigned int> m_wz;
        m_wz.push_back(150);
        m_wz.push_back(40);
        m_wz.transfer();
        auto m_wz_img = spade::utils::make_vec_image(m_wz.data(prim.device()));
        /* initialize random seed: */
        //srandom (time(NULL));
        //
        auto inflow = [=] _sp_hybrid (const prim_t& q_domain, const prim_t& q_ghost,const spade::coords::point_t<coor_t>& x_g,const int dir) mutable
        {
          prim_t q2 = q_ghost;
          return q2;
        };

        //create domain boundary conditions: 1) symmetry, freestream,
        const auto outflow = [=] _sp_hybrid (const prim_t& q_dom,const prim_t& q_ghost,const spade::coords::point_t<coor_t>& x_g, const int dir)
        {
          prim_t q2 = q_ghost;
          q2.p() = 101325.;
          return q2;
        };

        const auto top = [=] _sp_hybrid (const prim_t& q_dom,const prim_t& q_ghost,const spade::coords::point_t<coor_t>& x_g, const int&)
        {
          return prim_t{q_dom.p(), Tinf, Uinf, q_dom.v(), 0.};
        };

        const auto freestream = [=] _sp_hybrid (const prim_t& q, const int&)
        {
          return prim_t{Pinf, Tinf, uu, vv, ww};
        };

        //what is the type of the boundary condition???? >>> condition on coordinate???
    //    auto sym_bdy    = spade::boundary::zmin;
    //    auto extrap_bdy = spade::boundary::zmax || spade::boundary::ymin || spade::boundary::ymax || spade::boundary::xmax;
    //    auto const_bdy  = spade::boundary::xmin;
        //
        auto y_top    = spade::boundary::ymax;
        auto y_bot    = spade::boundary::ymin;
        auto x_out    = spade::boundary::xmax;
        auto x_in     = spade::boundary::xmin;

          //auto periodic_bdy  = spade::boundary::xmin || spade::boundary::xmax || spade::boundary::zmin || spade::boundary::zmax;
        //
        //lambda to apply boundary conditions, fill ghosts and then exchange the solution
        auto boundary_cond = [&](parray_t& sol, const real_t& t)
        {
            //call wall model here and then pass wall model data into ghost filling routine
            //another imagepoint interpolation call
            // local::fill_ghost_vals(sol, ghosts, ips,  sampldata);
            spade::timing::tmr_t t0;
            t0.start();
            handle.exchange(sol, group);
            t0.stop();

            //spade::algs::boundary_fill(sol, y_boundaries, freestream);
            
            spade::timing::tmr_t t1;
            t1.start();
            spade::algs::boundary_fill(sol, x_in, inflow);
            spade::algs::boundary_fill(sol, x_out, spade::boundary::extrapolate<2>);
            spade::algs::boundary_fill(sol, x_out, outflow);
            t1.stop();
            
            spade::timing::tmr_t t2;
            t2.start();
			spade::algs::boundary_fill(sol, y_bot, top);
            spade::algs::boundary_fill(sol, y_top, top);
            t2.stop();
            
            spade::timing::tmr_t t3;
            t3.start();
            spade::sampling::sample_array(sampldata,  sol, interp);
            spade::sampling::sample_array(sampldata2, sol, interp2);
            t3.stop();
            
            spade::timing::tmr_t t4;
            t4.start();
            local::compute_wm(tau_w, grad_w, q_w,sampldata, ghosts, ips, ips2, visc, air, prim.device());
            t4.stop();
            
            spade::timing::tmr_t t5;
            t5.start();
            local::fill_ghost_vals(sol, ghosts, ips2, sampldata, sampldata2, grad_w);
            t5.stop();
            
            
            if (group.isroot() && print_perf)
            {
                print("============================================== BDY ==============================================");
                std::string fmt0;
                int nn = 13;
                fmt0 += spade::utils::pad_str("exchg",   nn);
                fmt0 += spade::utils::pad_str("bfill-x", nn);
                fmt0 += spade::utils::pad_str("bfill-y", nn);
                fmt0 += spade::utils::pad_str("sampl",   nn);
                fmt0 += spade::utils::pad_str("wallm",   nn);
                fmt0 += spade::utils::pad_str("fillg",   nn);
                
                std::string fmt1;
                fmt1 += spade::utils::pad_str(t0.duration(), nn);
                fmt1 += spade::utils::pad_str(t1.duration(), nn);
                fmt1 += spade::utils::pad_str(t2.duration(), nn);
                fmt1 += spade::utils::pad_str(t3.duration(), nn);
                fmt1 += spade::utils::pad_str(t4.duration(), nn);
                fmt1 += spade::utils::pad_str(t5.duration(), nn);
                
                print(fmt0);
                print(fmt1);
                print();
            }
            
            //group.pause();
            //spade::algs::boundary_fill(sol, y_boundaries,   spade::boundary::extrapolate<1>);
            //spade::algs::boundary_fill(sol, extrap_bdy,   spade::boundary::extrapolate<1>);
            //spade::algs::boundary_fill(sol, periodic_bdy,    periodic);
            //spade::algs::boundary_fill(sol, const_bdy,  freestream);
        };
        //There are dispatch calls inside each of the routines call through boundary_cond
        boundary_cond(prim, time0);
        
        
        // spade::pde_algs::flux_div(prim, rhs, spade::omni::compose(tscheme, vscheme), spade::pde_algs::ldbalnp);

        
        /*
        auto array_img = prim.image();
        auto grid_geom = grid.image(prim.device());
        using sgs_t  = decltype(eddy_visc);
        using sten_t = spade::omni::prefab::cell_mono_t<sgs_t::info_type>;
        using data_t = spade::omni::stencil_data_t<sten_t, decltype(prim)>;
        data_t data;
        spade::grid::cell_idx_t icell = {0,2,0,24};
        spade::omni::retrieve(grid_geom,array_img,icell,data);
        print("At cell: ", icell);
        print("At coord:", grid_geom.get_coords(icell));
        auto mut = eddy_visc.get_mu_t(data.cell(0_c));
        */
        
        const real_t engscale = 1.0/rhoInf*Uinf*Uinf;
        const real_t masscale = 1.0/rhoInf;
        const real_t momscale = 1.0/rhoInf*Uinf;
        
        const auto resid_func = [=] _sp_hybrid (const flux_t& ff)
        {
            return
                spade::utils::abs(masscale*ff.continuity()) +
                spade::utils::abs(engscale*ff.energy())     +
                spade::utils::abs(momscale*ff.x_momentum()) +
                spade::utils::abs(momscale*ff.y_momentum()) +
                spade::utils::abs(momscale*ff.z_momentum());
        };
        const auto get_resid  = spade::algs::make_reduction(rhs, resid_func, spade::algs::max);
        const auto rcalc      = [&](const auto& rhs_in) { return spade::algs::transform_reduce(rhs_in, get_resid); };
        
        
        //define the time-integration routine
        spade::time_integration::time_axis_t axis(time0, dt);
        spade::time_integration::ssprk3hs_t alg;
        // spade::time_integration::rk2_t alg;
        // spade::time_integration::crank_nicholson_t alg(rcalc, spade::utils::converge_crit_t{1e-5, 65}, 0.65);
    
        // define q as the solution variable
        spade::time_integration::integrator_data_t q(std::move(prim), std::move(rhs), alg);
        
        const auto arsize = [&](const auto& v)
        {
            using v_t = typename spade::utils::remove_all<decltype(v)>::type::value_type;
            return v.size()*sizeof(v_t);
        };
        
        //finally defining the time integrator, it seems q.solution_data is the solution array and q.residual_data is the rhs
        spade::time_integration::integrator_t time_int(axis, alg, q, calc_rhs, boundary_cond, trans);
        if (group.isroot()) print("Start time integration...");
        // start time-integration loop
        real_t cur_cfl = targ_cfl;
        int cfl_skip = 1;
		int nt_min   = (do_restart)? nt_restart:0;
        for (auto nt: range(nt_min, nt_max+nt_restart+1))
        {
            if (nt % cfl_skip == 0)
            {
                const real_t sig = spade::algs::transform_reduce(time_int.solution(), get_sigma);
                cur_cfl = sig*dt/dx;
            }
            
            //seems that CFL is not recomputed
            if (group.isroot())
            {
                const int pn = 10;
                print(
                    "nt:  ",  spade::utils::pad_str(nt, pn),
                    "CFL: ",  spade::utils::pad_str(cur_cfl, pn)
                    );
            }
    
            // solution and surface output
            if (nt%nt_skip == 0)
            {
                if (group.isroot()) print("Output solution...");
                std::string nstr = spade::utils::zfill(nt, 8);
                std::string filename = "prims"+nstr;
                std::string fname_surf = "output/surf" + nstr + ".vtk";
                if (do_output) spade::io::output_vtk("output", filename, time_int.solution());
                if (do_output) local::vtkout(fname_surf, time_int.solution(), geom);
                if (group.isroot()) print("Done.");
            }

            if (nt%restart_skip==-1)
              {
                std::string nstr = spade::utils::zfill(nt, 8);
                std::string filename = "restart/restart_"+nstr + ".sol";
                if (group.isroot()) print("Writing restart file...",filename);
                //
                prim_t fill_tmp=0;
                spade::grid::grid_array restart_data(grid, fill_tmp, {2, 2, 2}, spade::device::cpu);
                restart_data.data = time_int.solution().data;
                //
                local::write_restart(filename,restart_data);
                // spade::grid::sample_array(surfdata, time_int.solution(), surf_sampl);
                // local::output_surf_vtk(filename, geom, surfdata.data(spade::device::cpu));
                if (group.isroot()) print("Done.");
              }
    
            //calling the time integrator
            {
                spade::timing::scoped_tmr_t tmr("adv");
                time_int.advance();
            }
        }
    });
    return 0;
}
