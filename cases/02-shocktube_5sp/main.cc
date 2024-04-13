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

constexpr static std::size_t nspecies=5;
constexpr static std::size_t nreactions=5;

using real_t  = double;
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
    const int    nx                  = input["nx"];
    const int    ny                  = input["ny"];
    const int    nz                  = input["nz"];
    const int    nxb                 = input["nxb"];
    const int    nyb                 = input["nyb"];
    const int    nzb                 = input["nzb"];
    const int    interval            = input["interval"];
    const int    nt_max              = input["nt_max"];
    const bool   do_output           = input["do_output"];
    const bool   output_rhs          = input["output_rhs"];
    const real_t targ_cfl            = input["cfl"];
    const std::vector<int> devices   = input["devices"];
	const bool        print_perf     = input["print_perf"];
    const std::string species_fname  = input["speciesFile"];
	const std::string gibbs_fname    = input["gibbsFile"];
	const std::string reaction_fname = input["reactionFile"];
	const std::vector<std::string> speciesNames = input["speciesList"];
    
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
        
        const real_t time_max            = 1.0;
        const int    nt_skip             = 250;
        const int    nguard              = 2;
        const real_t xmin                = 0.0;
        const real_t xmax                = 1.0;
        const real_t ymin                = 0.0;
        const real_t ymax                = 0.25;
        const real_t zmin                = 0.0;
        const real_t zmax                = 0.25;
        
        // Define the gas model
		gas_t air;

		// Import species data
		spade::fluid_state::import_species_data(species_fname, speciesNames, air);

		// Initialize reaction mechanism
		react_t react;

		// Import gibbs energy data
		spade::fluid_state::import_gibbsEnergy_data(gibbs_fname, speciesNames, react);
		
		// Import reaction mechanism
		spade::fluid_state::import_reaction_data(reaction_fname, speciesNames, air, react);		
		
		// initialize block structure
        spade::ctrs::array<int, 3> num_blocks     = {nxb, nyb, nzb};
        spade::ctrs::array<int, 3> cells_in_block = {nx,  ny,  nz };
        spade::ctrs::array<int, 3> exchange_cells = nguard;

		// Domain bounds
        spade::bound_box_t<real_t, 3> bounds;
        bounds.min(0) =  xmin;
        bounds.max(0) =  xmax;
        bounds.min(1) =  ymin;
        bounds.max(1) =  ymax;
        bounds.min(2) =  zmin;
        bounds.max(2) =  zmax;
        
        //cartesian coordinate system
        spade::coords::identity<real_t> coords;
        
        //create grid
        spade::amr::amr_blocks_t      blocks(num_blocks, bounds);
        spade::grid::cartesian_grid_t grid(cells_in_block, blocks, coords, pool);
        spade::ctrs::array<bool, 3>   periodic = {false,true,true};
        
        spade::io::mkdir("debug");
        spade::io::output_vtk("debug/grid.vtk", grid);
        const std::size_t num_points = grid.get_grid_size();
        if (pool.isroot()) print("points:", num_points);
        pool.sync();
        
        //create arrays residing on the grid
        prim_t fill1 = 0.0;
        spade::grid::grid_array prim(grid, fill1, exchange_cells, spade::device::best, spade::mem_map::tiled);
        
        flux_t fill2 = 0.0;
        spade::grid::grid_array rhs (grid, fill2, {0, 0, 0},      spade::device::best, spade::mem_map::tiled);
		
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
			
			if (x[0]<0.5)
			{
				// Left state
				output.Ys(0) = (4.45191405E-2);
				output.Ys(1) = (1.56276835E-5);
				output.Ys(2) = (1.03442127E-3);
				output.Ys(3) = (7.21998130E-1);
				output.p()   = 195000.0;
				output.T()   = 9000.0;
				output.Tv()  = 9000.0;
				output.u()   = 0.0;
				output.v()   = 0.0;
				output.w()   = 0.0;
			}
			else
			{
				// Right state
				output.Ys(0) = 0.767;
				output.Ys(1) = 0.233;
				output.Ys(2) = 1E-10;
				output.Ys(3) = 1E-10;
				output.p()   = 10000.0;
				output.T()   = 300.0;
				output.Tv()  = 300.0;
				output.u()   = 0.0;
				output.v()   = 0.0;
				output.w()   = 0.0;
			}
            
            return output;
        };
        
        // Fill the initial condition
        spade::algs::fill_array(prim, ini);
        if (pool.isroot()) print("exec exchg.");
        handle.exchange(prim, pool);
        if (pool.isroot()) print("done.");

		// Compute max wavespeed
		const auto sigma_func = [=] _sp_hybrid (const prim_t& q) {return sqrt(q.u()*q.u() + q.v()*q.v() + q.w()*q.w()) + spade::fluid_state::get_sos(q, air);};
		const auto get_sigma  = spade::algs::make_reduction(prim, sigma_func, spade::algs::max);
		const auto sigma_ini  = spade::algs::transform_reduce(prim, get_sigma);

		// Calculate timestep
		real_t time0    = float_t(0.0);
		const auto dxs  = grid.compute_dx_min();
		const real_t dx = spade::utils::min(dxs[0], dxs[1], dxs[2]);
		const real_t dt = targ_cfl * dx / sigma_ini;

		// Create state transformation function
		cons_t transform_state;
		spade::fluid_state::state_transform_t trans(transform_state, air);		
		
		// Lambda for left state BC
		const auto leftState = [=] _sp_hybrid (const prim_t& q_domain, const prim_t& q_ghost, const point_type& x_g, const int dir)
        {
			prim_t qguard;
			// Left state
			qguard.Ys(0) = (4.45191405E-2);
			qguard.Ys(1) = (1.56276835E-5);
			qguard.Ys(2) = (1.03442127E-3);
			qguard.Ys(3) = (7.21998130E-1);
			qguard.p()   = 195000.0;
			qguard.T()   = 9000.0;
			qguard.Tv()  = 9000.0;
			qguard.u()   = 0.0;
			qguard.v()   = 0.0;
			qguard.w()   = 0.0;
			return qguard;
		};
		
		// Lambda for right state BC
		const auto rightState = [=] _sp_hybrid (const prim_t& q_domain, const prim_t& q_ghost, const point_type& x_g, const int dir)
        {
			prim_t qguard;
			// Right state
			qguard.Ys(0) = 0.767;
			qguard.Ys(1) = 0.233;
			qguard.Ys(2) = 1E-10;
			qguard.Ys(3) = 1E-10;
			qguard.p()   = 10000.0;
			qguard.T()   = 300.0;
			qguard.Tv()  = 300.0;
			qguard.u()   = 0.0;
			qguard.v()   = 0.0;
			qguard.w()   = 0.0;
			return qguard;
		};
		
		// Link to domain bounds
        auto x_out    = spade::boundary::xmax;
        auto x_in     = spade::boundary::xmin;

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
			spade::algs::boundary_fill(q, x_in, leftState);
			
			// X-max BC
			spade::algs::boundary_fill(q, x_out, rightState);
			t1.stop();

			if (pool.isroot() && print_perf)
            {
                print("============================================== BDY ==============================================");
                std::string fmt0;
                int nn = 13;
                fmt0 += spade::utils::pad_str("exchg  ",   nn);
                fmt0 += spade::utils::pad_str("bfill-x", nn);
                
                std::string fmt1;
                fmt1 += spade::utils::pad_str(t0.duration(), nn);
                fmt1 += spade::utils::pad_str(t1.duration(), nn);
                
                print(fmt0);
                print(fmt1);
                print();
            }
			
		};
		
		// Now we apply the BC onto primitive array and run exchange
		boundary_cond(prim, time0);
		
		// Initialize viscous laws
		// nothing here yet
		
		// Set convective scheme
		const auto flux_func = spade::convective::rusanov_chem_t<real_t, air.nspecies()>(air);
		//spade::convective::weno_t inviscidScheme(flux_func);
		spade::convective::charweno_t inviscidScheme(flux_func, air);
		
		// Set viscous scheme
		//spade::viscous::visc_lr viscousScheme();

		// Set source term
		spade::fluid_state::chem_source_t<real_t, air.nspecies(), react.nreact()> chem_source(air, react);

		// Set RHS lambda
		int count = 0;
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
			
			// Add chemical source term
			spade::timing::tmr_t t1;
			t1.start();
			//spade::pde_algs::source_term(prim_in, rhs_in, chem_source);
			t1.stop();

			if (output_rhs)
			{
				if (pool.isroot()) print("Output rhs...");
				std::string nstr = spade::utils::zfill(count, 8);
				std::string filename = "rhs_in"+nstr;
				spade::io::output_vtk("output", filename, rhs_in);

				if (pool.isroot()) print("Output prim...");
				nstr = spade::utils::zfill(count, 8);
				filename = "prim_in"+nstr;
				spade::io::output_vtk("output", filename, prim_in);
			}
			
			if (pool.isroot() && print_perf)
            {
                print("============================================== RHS ==============================================");
                std::string fmt0;
                int nn = 20;
                fmt0 += spade::utils::pad_str("flux_div", nn);
				fmt0 += spade::utils::pad_str("source  ", nn);
                
                std::string fmt1;
                fmt1 += spade::utils::pad_str(t0.duration(), nn);
                fmt1 += spade::utils::pad_str(t1.duration(), nn);
                
                print(fmt0);
                print(fmt1);
                print();
            }
			++count;
		};

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
		for (auto nt: range(0,nt_max+1))
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
                const int pn = 10;
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

			// End point based on physical time
			if (time_loc > time_max)
            {
                if (pool.isroot()) print("Simulation time limit reached, exiting.");
                break;
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
