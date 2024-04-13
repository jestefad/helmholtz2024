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
}

#include "scidf.h"
#include "spade.h"

using real_t = float;
using flux_t = spade::fluid_state::flux_t<real_t>;
using prim_t = spade::fluid_state::prim_t<real_t>;
using cons_t = spade::fluid_state::cons_t<real_t>;

int main(int argc, char** argv)
{
    std::string ifile = "input.sdf";
    if (argc > 1) ifile = std::string(argv[1]);
    
    scidf::node_t input;
    scidf::read(ifile, input);
    const int    nx                = input["nx"];
    const int    ny                = input["ny"];
    const int    nz                = input["nz"];
    const int    nxb               = input["nxb"];
    const int    nyb               = input["nyb"];
    const int    nzb               = input["nzb"];
    const int    interval          = input["interval"];
    const int    nt_max            = input["nt_max"];
    const bool   do_output         = input["do_output"];
    const bool   compare_rhs       = input["compare_rhs"];
    const real_t targ_cfl          = input["cfl"];
    const std::vector<int> devices = input["devices"];
    
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
        
        const real_t time_max            = 100.0;
        const int    nt_skip             = 250;
        const int    nguard              = 2;
        const real_t xmin                = -spade::consts::pi;
        const real_t xmax                =  spade::consts::pi;
        const real_t ymin                = -spade::consts::pi;
        const real_t ymax                =  spade::consts::pi;
        const real_t zmin                = -spade::consts::pi;
        const real_t zmax                =  spade::consts::pi;
        
        const real_t mach                = 0.1;
        const real_t reynolds            = 1600.0;
        
        const real_t u0        = 1.0;
        const real_t T0        = 1.0;
        const real_t gamma     = 1.4;
        const real_t rho0      = 1.0;
        const real_t L         = 1.0;
        const real_t prandtl   = 0.71;
        
        const real_t sos       = u0/mach;
        const real_t rgas      = sos*sos/(gamma*T0);
        const real_t mu0       = rho0*u0*L/reynolds;
        const real_t p0        = rho0*u0*u0/(gamma*mach*mach);
        
        //define the gas model
        
        spade::fluid_state::ideal_gas_t<real_t> air(gamma, rgas);
        
        spade::ctrs::array<int, 3> num_blocks     = {nxb, nyb, nzb};
        spade::ctrs::array<int, 3> cells_in_block = {nx,  ny,  nz };
        spade::ctrs::array<int, 3> exchange_cells = nguard;
        
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
        spade::ctrs::array<bool, 3>   periodic = true;
        
        spade::io::mkdir("debug");
        spade::io::output_vtk("debug/grid.vtk", grid);
        const std::size_t num_points = grid.get_grid_size();
        if (pool.isroot()) print("points:", num_points);
        pool.sync();
        
        //create arrays residing on the grid
        prim_t fill1 = 0.0;
        spade::grid::grid_array prim(grid, fill1, exchange_cells, spade::device::best, spade::mem_map::tiled_small);
        
        flux_t fill2 = 0.0;
        spade::grid::grid_array rhs (grid, fill2, {0, 0, 0},      spade::device::best, spade::mem_map::tiled);
        
        if (pool.isroot()) print("exchg.");
        auto handle = spade::grid::make_exchange(prim, periodic);
        if (pool.isroot()) print("done.");

        auto pimg = prim.image();
        using point_type = decltype(grid)::coord_point_type;
        const auto ini = [=] _sp_hybrid (const point_type& x, const spade::grid::cell_idx_t& ii)
        {
            prim_t output;
            
            output.p() =  p0 + (1.0/16.0)*rho0*u0*u0*(cos(2*x[0]/L) + cos(2*x[1]/L))*(cos(2*x[2]/L) + 2.0);
            output.T() =  T0;
            output.u() =  u0*sin(x[0]/L)*cos(x[1]/L)*cos(x[2]/L);
            output.v() = -u0*cos(x[0]/L)*sin(x[1]/L)*cos(x[2]/L);
            output.w() =  0.0;
            
            return output;
        };
        
        //fill the initial condition
        spade::algs::fill_array(prim, ini);
        if (pool.isroot()) print("exec exchg.");
        handle.exchange(prim, pool);
        if (pool.isroot()) print("done.");
        
        const auto c2 = spade::convective::cent_keep<4>(air);
        const auto central = spade::convective::cent_keep<4>(air);
        spade::convective::rusanov_t flx(air);
        // spade::convective::fweno_t fweno(air);
        spade::state_sensor::ducros_t ducr(real_t(1.0e-3));
        
        spade::convective::rusanov_fds_t flx_fds(air);
        spade::convective::weno_fds_t<decltype(flx_fds), spade::convective::disable_smooth> fweno(air);
        
        spade::convective::hybrid_scheme_t hyb_scheme(central, fweno, ducr, spade::convective::diss_flux);
        auto conv_scheme = hyb_scheme;
        
        // auto conv_scheme = central;
        
        const spade::viscous_laws::constant_viscosity_t visc_law(real_t(1.8e-5), real_t(0.72));
        
        const real_t cw    = 0.55;
        const real_t delta = 1e-3;
        const real_t pr_t  = 0.9;
        spade::subgrid_scale::wale_t eddy_visc(air, cw, delta, pr_t);
        
        spade::viscous_laws::sgs_visc_t sgs_visc(visc_law, eddy_visc);
        
        const spade::viscous::visc_lr visc_scheme(sgs_visc, air);
        
        //compute max wavespeed
        const auto sigma_func = [=] _sp_hybrid (const prim_t& q) { return sqrt(q.u()*q.u() + q.v()*q.v() + q.w()*q.w()) + sqrt(gamma*rgas*q.T()); };
        const auto get_sigma  = spade::algs::make_reduction(prim, sigma_func, spade::algs::max);
        const auto sigma_ini  = spade::algs::transform_reduce(prim, get_sigma);
        
        //calculate timestep
        real_t time0        = 0.0;
        const auto   dxs    = grid.compute_dx_min();
        const real_t dx     = spade::utils::min(dxs[0], dxs[1], dxs[2]);
        const real_t dt     = targ_cfl*dx/sigma_ini;
        const real_t t_char = L/u0;
        
        
        //define the conservative variable transformation
        cons_t transform_state;
        spade::fluid_state::state_transform_t trans(transform_state, air);
    
        auto bc = [&](auto& q, const auto& t)
        {
            spade::timing::tmr_t tmr;
            tmr.start();
            handle.exchange(q, pool);
            tmr.stop();
            if (pool.isroot()) print("bndy:", tmr.duration(), "ms");
        };
        
        //define the residual calculation
        int rhs_count = 0;
        real_t total_ms = 0.0;        
        
        auto calc_rhs = [&](auto& rhs_in, const auto& q, const auto& t)
        {
            spade::timing::tmr_t tmr;
            tmr.start();
            const auto traits = spade::algs::make_traits(spade::pde_algs::fldbc, spade::pde_algs::overwrite);
            spade::pde_algs::flux_div(q, rhs_in, spade::omni::compose(visc_scheme, conv_scheme), traits);
            // spade::pde_algs::flux_div(q, rhs_in, central, traits);
            tmr.stop();
            if (pool.isroot()) print("rhs: ", tmr.duration(), "ms");
            ++rhs_count;
            total_ms += tmr.duration();
            
            if (compare_rhs)
            {
                rhs_in = 0.0;
                spade::pde_algs::flux_div(q, rhs_in, spade::omni::compose(visc_scheme, conv_scheme));
                spade::io::output_vtk("output", "rhs_0", rhs_in);
                rhs_in = 0.0;
                spade::pde_algs::flux_div(q, rhs_in, spade::omni::compose(visc_scheme, conv_scheme), traits);
                spade::io::output_vtk("output", "rhs_1", rhs_in);
                print(spade::utils::where());
                std::cin.get();
            }
        };
        
        
        spade::time_integration::time_axis_t axis(time0, dt);
        spade::time_integration::ssprk3_t alg;
        spade::time_integration::integrator_data_t qdata(std::move(prim), std::move(rhs), alg);
        spade::time_integration::integrator_t time_int(axis, alg, qdata, calc_rhs, bc, trans);
        
        
        int cfl_sampl_interval = 1;
        real_t cur_cfl = targ_cfl;
        
        int adv_count = 0;
        real_t adv_dur = 0.0;
        
        for (auto nt: range(0, nt_max+1))
        {
            const auto& sol = time_int.solution();
            const auto time_loc = time_int.time()/t_char;
            
            if (nt % cfl_sampl_interval == 0)
            {
                spade::timing::tmr_t tmr;
                tmr.start();
                const auto sig_max = spade::algs::transform_reduce(sol, get_sigma);
                tmr.stop();
                cur_cfl = sig_max*dt/dx;
                if (pool.isroot()) print("reduc took", tmr.duration(), "ms");
            }
            
            //print some nice things to the screen
            if (pool.isroot())
            {
                const int pn = 10;
                print(
                    "nt:  ",  spade::utils::pad_str(nt,      pn),
                    "CFL: ",  spade::utils::pad_str(cur_cfl, pn)
                    );
            }
            
            //output the solution
            if ((nt%interval == 0) && do_output)
            {
                if (pool.isroot()) print("Output solution...");
                std::string nstr = spade::utils::zfill(nt, 8);
                std::string filename = "prims"+nstr;
                spade::io::output_vtk("output", filename, sol);
            }

            if (time_loc > time_max)
            {
                if (pool.isroot()) print("Simulation time limit reached, exiting.");
                break;
            }
            
            //advance the solution
            spade::timing::tmr_t tmr;
            tmr.start();
            time_int.advance();
            tmr.stop();
            
            adv_count ++;
            adv_dur   += tmr.duration();
    
            if (pool.isroot()) print("advance:", tmr.duration(), "ms");
        }
        
        if (pool.isroot())
        {
            double rhs_avg = total_ms/rhs_count;
            double adv_avg = adv_dur/adv_count;
            int nstage = qdata.residual_data.size();
            
            double pct_rhs = nstage*rhs_avg/adv_avg;
            
            print("Average RHS eval:   ", rhs_avg);
            print("Average Adv:        ", adv_avg);
            print("RHS SOL MUPS:       ", num_points/(nstage*rhs_avg*1000.0));
            print("Achieved MUPS:      ", num_points/(adv_avg*1000.0));
            print("Proportion RHS (%): ", 100.0*pct_rhs);
        }
    });
    
    return 0;
}
