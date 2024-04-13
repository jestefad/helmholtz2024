#include <stdio.h>
#include <vector>

#include <iostream>
#include <string>

namespace debug
{
    template <typename thing_t> void print_type(const thing_t&)
    {
        //g++ only
        std::string pf(__PRETTY_FUNCTION__);
        std::string srch = "thing_t = ";
        std::size_t start = pf.find(srch) + srch.length();
        std::size_t end = pf.length()-1;
        std::cout << pf.substr(start, end-start) << std::endl;
    }

	template<typename ptype>
	void print_state(const ptype& q)
	{
		std::cout<<"DEBUG OUTPUT STATE"<<std::endl;
		for (int i=0; i<q.size(); ++i)
		{
			std::cout<<q.name(i)<<" = "<<q[i]<<std::endl;
		}
		return;
	}
}

#include "scidf.h"
#include "spade.h"
#include <time.h>
#include "sample_cloud.h"
#include "fill_ghosts.h"
#include "surf_vtk.h"
#include "vtkout.h"
//#include "compute_wm.h"
//#include "compute_irreg_visc.h"
#include "compute_irreg_conv.h"
#include "profile_reader.h"
#include "refine_mesh.h"
#include "inout.h"
#include "restart.h"
#include "cfi.h"

constexpr static std::size_t nspecies=5;
constexpr static std::size_t nreactions=5;

using real_t  = float;
using coor_t    = double;
using prim_t  = spade::fluid_state::prim_chem_t<real_t, nspecies>;
using cons_t  = spade::fluid_state::cons_chem_t<real_t, nspecies>;
using flux_t  = spade::fluid_state::flux_chem_t<real_t, nspecies>;
using gas_t   = spade::fluid_state::multicomponent_gas_t<real_t, nspecies>;
using react_t = spade::fluid_state::reactionMechanism_t<real_t, nspecies, nreactions>; // Number of reactions

int main(int argc, char** argv)
{
    std::string ifile = "input.sdf";
    if (argc > 1) ifile = std::string(argv[1]);
    
    scidf::node_t input;
    scidf::read(ifile, input);

	// Grid flags
	spade::ctrs::array<int, 3> nxyz            = input["Grid"]["nxyz"];
	spade::ctrs::array<int, 3> nblks           = input["Grid"]["nblks"];
	spade::ctrs::array<int, 3> nexch           = input["Grid"]["nexch"];
	const spade::ctrs::array<coor_t, 6> bnd    = input["Grid"]["bounds"];
	const std::string geom_fname               = input["Grid"]["geom"];
	const coor_t sampl_dist                    = input["Grid"]["sampl_dist"];
	const bool is_external                     = input["Grid"]["is_external"];
	spade::ctrs::array<bool, 3> periodic       = input["Grid"]["periodic"];
	const int maxLevel                         = input["Grid"]["maxlevel"];
	spade::bound_box_t<coor_t, 3> bounds;
	bounds.bnds = bnd;
	
	// Refinement flags
	const coor_t dx_boundary                = input["Refine"]["dx_boundary"];
	const int dx_direction                  = input["Refine"]["dx_direction"];
	const spade::ctrs::array<coor_t, 3> AR  = input["Refine"]["aspect"];

	// Debugging flags
	const bool   output_rhs          = input["Debug"]["output_rhs"];
	const bool   print_perf          = input["Debug"]["print_perf"];
	
	// Configuration flags
    const real_t targ_cfl            = input["Config"]["cfl"];
	const int    interval            = input["Config"]["interval"];
    const int    nt_max              = input["Config"]["nt_max"];
	const bool   do_output           = input["Config"]["do_output"];
    const std::vector<int> devices   = input["Config"]["devices"];
	const bool       do_restart      = input["Config"]["do_restart"];
	const int        nt_restart      = (do_restart) ? input["Config"]["nt_restart"] : 0;
    const int        restart_skip    = input["Config"]["restart_skip"];
	
	// Multispecies flags
    const std::string species_fname             = input["Multispecies"]["speciesFile"];
	const std::string gibbs_fname               = input["Multispecies"]["gibbsFile"];
	const std::string reaction_fname            = input["Multispecies"]["reactionFile"];
	const std::vector<std::string> speciesNames = input["Multispecies"]["speciesList"];

	// Initial conditions
	const real_t rhoinf                         = input["Fluid"]["rhoinf"];
	const spade::ctrs::array<real_t, 5> Yinf    = input["Fluid"]["Yinf"];
	const real_t Uinf                           = input["Fluid"]["Uinf"];
	const real_t aoa                            = input["Fluid"]["aoa"];
	const real_t Tinf                           = input["Fluid"]["Tinf"];
	const real_t Tvinf                          = input["Fluid"]["Tvinf"];
    
    spade::parallel::compute_env_t env(&argc, &argv, devices);
    
    env.exec([&](spade::parallel::pool_t& pool)
    {
        if (pool.isroot() && !pool.p2p_enabled())
        {
            print("Warning: P2P support not active. This may affect performance.");
        }
        if (pool.isroot())
        {
            print("Num. threads:", devices.size());
        }
        
        // Define the gas model
        spade::fluid_state::multicomponent_gas_t<real_t, nspecies> air5;

		// Import species data
		spade::fluid_state::import_species_data(species_fname, speciesNames, air5);
		
		// Initialize reaction mechanism
		spade::fluid_state::reactionMechanism_t<real_t, nspecies, nreactions> react5;

		// Import gibbs energy data
		spade::fluid_state::import_gibbsEnergy_data(gibbs_fname, speciesNames, react5);
		
		// Import reaction mechanism
		spade::fluid_state::import_reaction_data(reaction_fname, speciesNames, air5, react5);
		
		// initialize block structure
		spade::amr::amr_blocks_t blocks(nblks, bounds);
        
        // Cartesian coordinate system --> creates identity matrix
        spade::coords::identity<coor_t> coords;

		// Create the Cartesian grid with specified blocks arrangement and coordinate system, MPI group)
		if (pool.isroot()) print("Create grid");
        spade::grid::cartesian_grid_t grid(nxyz, blocks, coords, pool);
        if (pool.isroot()) print("Done");

		// Read the geometry file
        if (pool.isroot()) print("Read", geom_fname);
        spade::geom::vtk_geom_t<3, 3, coor_t> geom;
        spade::geom::read_vtk_geom(geom_fname, geom, is_external);
        if (pool.isroot()) print("Done");

        // Controlling refinement in (x,y,z)-directions
        using refine_t = typename decltype(blocks)::refine_type;
        refine_t ref0  = {true, true, true};
        if (pool.isroot()) print("Begin refine");
        const bool boundary_refinement = true;
        if (boundary_refinement) local::refine_mesh(grid,geom,dx_boundary,dx_direction,AR,periodic);
        pool.sync();

		// Get minimum grid spacing
		const auto dxs  = grid.compute_dx_min();
		const real_t dx = spade::utils::min(dxs[0], dxs[1], dxs[2]);
		if (pool.isroot()) print("Grid spacing (min) = ",dxs);
        const std::size_t num_points = grid.get_grid_size();
        if (pool.isroot()) print("points:", num_points);
		
        // Create arrays residing on the grid
        prim_t fill1 = 0.0;
        spade::grid::grid_array prim(grid, fill1, nexch, spade::device::best, spade::mem_map::tiled);
        
        flux_t fill2 = 0.0;
        spade::grid::grid_array rhs (grid, fill2, {0, 0, 0},      spade::device::best, spade::mem_map::tiled);

		// Initialize viscous laws
		// nothing here yet
		
		// Set convective scheme
		const auto flux_func = spade::convective::rusanov_chem_t<real_t, nspecies>(air5);
		spade::convective::charweno_t inviscidScheme(flux_func, air5);
		
		// Set viscous scheme
		//spade::viscous::visc_lr viscousScheme();

		// Set source term
		spade::fluid_state::chem_source_t<real_t, nspecies, nreactions> chem_source(air5, react5);
		
        // Computing the ghost points
        if (pool.isroot()) print("Compute ghosts");
		const auto ghosts = spade::ibm::compute_boundary_info(prim, geom, inviscidScheme, is_external);
        if (pool.isroot()) print("Done");

		// Dump ghosts and grid structure to file for checking
		if (pool.isroot()) spade::io::mkdir("debug");
        pool.sync();
        std::string file_title = "boundary.pid." + std::to_string(pool.rank());
        if (pool.isroot()) print("Output ghosts");
        spade::io::output_vtk("debug", file_title, ghosts, grid);
        if (pool.isroot()) print("Done");
        if (pool.isroot()) spade::io::output_vtk("debug/grid.vtk", grid);

		// Set first image point
        if (pool.isroot()) print("Compute ips");
        auto ips = spade::ibm::compute_ghost_sample_points(ghosts, grid, sampl_dist*dx);
        spade::io::output_vtk("debug/ips.vtk" , ips);
        if (pool.isroot()) print("Done");

		// Generate inout mapping
		int fill3=0;
        spade::grid::grid_array inout_cpu(grid, fill3, {2, 2, 2}, spade::device::cpu);
        spade::grid::grid_array inout_gpu(grid, fill3, {2, 2, 2}, spade::device::best);
		
        // Compute inout map
        local::generate_inout_map(inout_cpu,ghosts,geom);
        inout_gpu.data = inout_cpu.data;
        spade::io::output_vtk("debug","inout",inout_gpu);

		// Create interpolation operators
        const auto exclude = [&](const spade::grid::cell_idx_t& ii) { return geom.is_interior(grid.get_coords(ii)); };
        const auto strategy = spade::sampling::prioritize(spade::sampling::multilinear, spade::sampling::nearest, spade::sampling::force_nearest);
		
        if (pool.isroot()) print("Compute interp 1");
        auto interp  = [&](){
            try
            {
                auto tmp = spade::sampling::create_interpolation(prim, ips,  strategy,  exclude);
                return tmp;
            }
            catch (const spade::except::points_exception<coor_t>& e)
            {
                spade::io::output_vtk("debug/failed1." + std::to_string(pool.rank()) + ".vtk", e.data);
                throw e;
            }
        }();
        if (pool.isroot()) print("Done");
		
		// Initialize exchange
        if (pool.isroot()) print("init exchg.");
        auto handle = spade::grid::make_exchange(prim, periodic);
        if (pool.isroot()) print("done.");		
		
		// Lambda for initial condition
        auto pimg = prim.image();
        using point_type = decltype(grid)::coord_point_type;
        const auto ini = [=] _sp_hybrid (const point_type& x, const spade::grid::cell_idx_t& ii)
        {
            prim_t output;

			coor_t Rsurf = 0.045;
			coor_t Rloc  = sqrt(x[0]*x[0] + x[1]*x[1] + x[2]*x[2]);
			coor_t dist  = Rloc - Rsurf;
			coor_t rampMax = 0.1;
			
			// Initial condition
			real_t theta = spade::consts::pi*aoa/180.0;
			if (Rloc >= Rsurf)
			{
				// Velocity ramp
				output.u()   = Uinf * cos(theta) * spade::utils::min(1, dist / rampMax);
				output.v()   = Uinf * sin(theta) * spade::utils::min(1, dist / rampMax);
				output.w()   = 0.0;
			}
			else
			{
				// Velocity ramp
				output.u()   = Uinf * cos(theta) * spade::utils::min(1, dist / rampMax);
				output.v()   = Uinf * sin(theta) * spade::utils::min(1, dist / rampMax);
				output.w()   = 0.0;
			}
			for (int s = 0; s<output.nspecies(); ++s) output.Ys(s) = Yinf[s];
			output.T()   = Tinf;
			output.Tv()  = Tvinf;
			output.p()   = spade::fluid_state::get_pressure(rhoinf, Yinf, Tinf, Tvinf, air5);
            
            return output;
        };
        
        // Fill the initial condition
        spade::algs::fill_array(prim, ini);
		
		// Read initial condition from restart file if needed
        if (do_restart)
          {
            prim_t fill_tmp=0;
            std::string nstr = spade::utils::zfill(nt_restart, 8);
            std::string filename = "restart/restart_"+nstr + ".sol";
            if (pool.isroot()) print("read restart file...",filename);
            spade::grid::grid_array restart_data(grid, fill_tmp, {2, 2, 2}, spade::device::cpu);
            local::read_restart(filename,restart_data);
            prim.data = restart_data.data;
          }

		// Run exchange
        if (pool.isroot()) print("exec exchg.");
        handle.exchange(prim, pool);
        if (pool.isroot()) print("done.");

		// Interpolate into the image points
        if (pool.isroot()) print("Sample data");
        auto sampldata  = spade::sampling::sample_array(prim, interp);
        if (pool.isroot()) print("Done");

		// Compute max wavespeed
		const auto sigma_func = [=] _sp_hybrid (const prim_t& q) {return sqrt(q.u()*q.u() + q.v()*q.v() + q.w()*q.w()) + spade::fluid_state::get_sos(q, air5);};
		const auto get_sigma  = spade::algs::make_reduction(prim, sigma_func, spade::algs::max);
		const auto sigma_ini  = spade::algs::transform_reduce(prim, get_sigma);

		// Calculate timestep
		real_t time0    = float_t(0.0);
		const real_t dt = targ_cfl * dx / sigma_ini;
		print("umax+sos = ",sigma_ini);
		print("dt       = ",dt);

		// Create state transformation function
		cons_t transform_state;
		spade::fluid_state::state_transform_t trans(transform_state, air5);

		// Set RHS lambda
		auto calc_rhs = [&](auto& rhs_in, const auto& prim_in, const auto& t)
     	{
			// Initialize residual
			rhs_in = real_t(0.0);
			
			// Compute flux divergence
			spade::timing::tmr_t t0;
			t0.start();
			//spade::pde_algs::flux_div(prim, rhs, spade::omni::compose(inviscidScheme, viscousScheme), spade::pde_algs::ldbalnp);
			spade::pde_algs::flux_div(prim_in, rhs_in, inviscidScheme);
			t0.stop();

			// Irregular convective scheme
			spade::timing::tmr_t t1;
			t1.start();
			local::rhs_irreg_conv(prim_in, rhs_in, air5, inviscidScheme, ghosts, ips); // CHECK THIS !!!!
			t1.stop();
			
			// Add chemical source term
			spade::timing::tmr_t t2;
			t2.start();
			//spade::pde_algs::source_term(prim_in, rhs_in, chem_source);
			t2.stop();

			spade::timing::tmr_t t3;
            t3.start();
            local::zero_ghost_rhs(rhs_in, ghosts); // Zero RHS in ghost cells
            t3.stop();
            
            spade::timing::tmr_t t4;
            t4.start();
			local::zero_rhs_inside(prim_in, grid, rhs_in, inout_gpu); // Zero RHS inside solid domain
            t4.stop();
			
			if (pool.isroot() && print_perf)
            {
                print("============================================== RHS ==============================================");
                std::string fmt0;
                int nn = 20;
                fmt0 += spade::utils::pad_str("flux_div", nn);
				fmt0 += spade::utils::pad_str("irr_conv", nn);
				fmt0 += spade::utils::pad_str("source  ", nn);
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
		};
		
		// Lambda for supersonic_inflow
		const auto supersonic_inflow = [=] _sp_hybrid (const prim_t& q_domain, const prim_t& q_ghost, const point_type& x_g, const int dir)
        {
			prim_t qguard;
			for (int s = 0; s<qguard.nspecies(); ++s) qguard.Ys(s) = Yinf[s];
			real_t theta = spade::consts::pi*aoa/180.0;
			qguard.u()   = Uinf * cos(theta);
			qguard.v()   = Uinf * sin(theta);
			qguard.w()   = 0.0;
			qguard.T()   = Tinf;
			qguard.Tv()  = Tvinf;
			qguard.p()   = spade::fluid_state::get_pressure(rhoinf, Yinf, Tinf, Tvinf, air5);
			return qguard;
		};

		// Lambda for supersonic outflow
		const auto supersonic_outflow = [=] _sp_hybrid (const prim_t& q_domain, const prim_t& q_ghost, const point_type& x_g, const int dir)
        {
			prim_t qguard = q_domain;
			return qguard;
		};

		// Lambda for slipwall/symmetry
		const auto slipwall = [=] _sp_hybrid (const prim_t& q_domain, const prim_t& q_ghost, const point_type& x_g, const int dir)
        {
			prim_t qguard = q_domain;
			qguard.u(dir) = - q_domain.u(dir);
			return qguard;
		};
		
		// Link to domain bounds
        auto x_out    = spade::boundary::xmax;
        auto x_in     = spade::boundary::xmin;
		auto y_out    = spade::boundary::ymax;
        auto y_in     = spade::boundary::ymin;
		auto z_out    = spade::boundary::zmax;
        auto z_in     = spade::boundary::zmin;
		
		// Create lambda to apply BCs and run exchange
		using parray_t = decltype(prim);
		auto boundary_cond = [&](parray_t& q, const real_t& t)
     	{
			
			// Run exchange
			spade::timing::tmr_t t0;
			t0.start();
			handle.exchange(q, pool);
			t0.stop();

			spade::timing::tmr_t t1;
			t1.start();
			// X-min BC
			spade::algs::boundary_fill(q, x_in, supersonic_inflow);
			// X-max BC
			spade::algs::boundary_fill(q, x_out, supersonic_outflow);
			t1.stop();

			spade::timing::tmr_t t2;
			t2.start();
			// Y-min BC
			spade::algs::boundary_fill(q, y_in, slipwall);
			// Y-max BC
			spade::algs::boundary_fill(q, y_out, supersonic_outflow);
			t2.stop();

			spade::timing::tmr_t t3;
			t3.start();
			// Z-min BC
			spade::algs::boundary_fill(q, z_in, slipwall);
			// Z-max BC
			spade::algs::boundary_fill(q, z_out, supersonic_outflow);
			t3.stop();

			// Image interpolation
			spade::timing::tmr_t t4;
            t4.start();
            spade::sampling::sample_array(sampldata,  q, interp);
            t4.stop();

			// Ghost filling
			spade::timing::tmr_t t5;
            t5.start();
            local::fill_ghost_vals(q, ghosts, ips, sampldata); // CHECK THIS !!!!
            t5.stop();

			if (pool.isroot() && print_perf)
            {
                print("============================================== BDY ==============================================");
                std::string fmt0;
                int nn = 13;
                fmt0 += spade::utils::pad_str("exchg  ",   nn);
                fmt0 += spade::utils::pad_str("bfill-x", nn);
				fmt0 += spade::utils::pad_str("bfill-y", nn);
				fmt0 += spade::utils::pad_str("bfill-z", nn);
				fmt0 += spade::utils::pad_str("sample ", nn);
				fmt0 += spade::utils::pad_str("fillg  ", nn);
                
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
			
		};
		
		// Now we apply the BC onto primitive array and run exchange
		boundary_cond(prim, time0);

		// Setup time integration
		spade::time_integration::time_axis_t axis(time0, dt); // Pass time integrator time + dt to track computational time?
		spade::time_integration::ssprk3_t alg; // Initialize integrator
		spade::time_integration::integrator_data_t qdata(std::move(prim), std::move(rhs), alg);
		spade::time_integration::integrator_t time_int(axis, alg, qdata, calc_rhs, boundary_cond, trans);

		// Frequency to recompute CFL
		int cfl_sample_interval = 1;
		// Current CFL
		real_t cur_cfl = targ_cfl;
		// Number of solution advances
		int adv_count = 0;
		
		// Time integration loop
		int nt_min   = (do_restart)? nt_restart:0;
        for (auto nt: range(nt_min, nt_max+nt_restart+1))
		{
			// Get solution and physical time
			const auto& sol = time_int.solution();
			const auto time_loc = time_int.time();
			
			// Recompute CFL
			if (nt % cfl_sample_interval == 0)
            {
				// Compute max wave speed
				const auto sig_max = spade::algs::transform_reduce(sol, get_sigma);
				// Compute CFL
				cur_cfl = sig_max * dt / dx;
			}

			//print some nice things to the screen
            if (pool.isroot())
            {
                const int pn = 16;
                print(
                    "nt:   ",  spade::utils::pad_str(nt, pn),
                    "CFL:  ",  spade::utils::pad_str(cur_cfl, pn),
					"Time: ",  spade::utils::pad_str(time_loc, pn)
                    );
            }

			
			// Solution output frequency
            if ((nt%interval == 0) && do_output || nt==nt_max)
            {
                if (pool.isroot()) print("Output solution...");
                std::string nstr = spade::utils::zfill(nt, 8);
                std::string filename = "prims"+nstr;
                spade::io::output_vtk("output", filename, sol);
            }

			// Restart output frequency
			if (nt%restart_skip==-1)
              {
                std::string nstr = spade::utils::zfill(nt, 8);
                std::string filename = "restart/restart_"+nstr + ".sol";
                if (pool.isroot()) print("Writing restart file...",filename);
                //
                prim_t fill_tmp=0;
                spade::grid::grid_array restart_data(grid, fill_tmp, {2, 2, 2}, spade::device::cpu);
                restart_data.data = time_int.solution().data;
                //
                local::write_restart(filename,restart_data);
                if (pool.isroot()) print("Done.");
              }

			// Advance solution
			{
				spade::timing::scoped_tmr_t tmr("adv");
				time_int.advance();
			}

			// Count the timesteps
			adv_count ++;
		}
		
    });
	
    return 0;
}
