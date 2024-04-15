#pragma once
#define DEBUG 0
#include <concepts>

namespace local
{
    // std::vector<real_t>;
    // spade::device::device_vector<real_t>;
    
    template <typename vector_t, typename sample_data_t, typename ghosts_t, typename img_pts_t, typename visc_t, typename gas_t, typename device_t>
    inline void compute_wm(
        vector_t& tau_out,                // shear stress
        vector_t& grad_out,               // gradient at the location of image point 2
        vector_t& qw_out,                 // heat transfer
        const sample_data_t& sample_data, // data from the interpolation points
        const ghosts_t& ghosts,           // ghost data
        const img_pts_t& ips,             // location of image points (vector<point_t>) -> ray point to wall model
        const img_pts_t& ips2,            // second image point used to enforce boundary condition
        const visc_t& visc,               // viscous law
        const gas_t& gas,                 // air/gas model
        const device_t& dev               // device (cpu/gpu)
        )
    {
      auto tau_img               = spade::utils::make_vec_image(tau_out);
      auto grad_img              = spade::utils::make_vec_image(grad_out);
      const auto sample_data_img = spade::utils::make_vec_image(sample_data);
      const auto ips_img         = spade::utils::make_vec_image(ips.data(dev));
      const auto ips2_img        = spade::utils::make_vec_image(ips2.data(dev));
      const auto ghost_img       = ghosts.image(dev);
      auto qw_img                = spade::utils::make_vec_image(qw_out);
      //
      using pnt_t  = typename img_pts_t::value_type;
      using prim_t = typename sample_data_t::value_type;
      using real_t = typename prim_t::value_type;
      using coor_t = typename pnt_t::value_type;
      using vec_t  = spade::ctrs::array<real_t, 3>;
      //
      std::size_t wm_offset  = 0;
      std::size_t ips_offset = 0;
      for (int dir = 0; dir < ghosts.aligned.size(); ++dir)
      {
        const std::size_t nlayers   = ghosts.aligned[dir].num_layer();
        const std::size_t size_here = ghosts.aligned[dir].indices.size();
        const std::size_t size_here2= ghosts.aligned[dir].indices.size()*nlayers;
        //
        // printf("dir = %d, size_here = %ld, size_here2 = %ld \n",dir,size_here,size_here2);
        //
        auto range = spade::dispatch::ranges::make_range(0UL, size_here);
        const auto list = ghost_img.aligned[dir];
        auto loop = [=] _sp_hybrid (const std::size_t& i) mutable
        {
            const prim_t q_f = sample_data_img[i*nlayers + ips_offset];

            const auto& nvec = list.closest_normals[i][0];
            vec_t vel   = {q_f.u(), q_f.v(), q_f.w()};
            const vec_t unorm = spade::ctrs::array_val_cast<real_t>(spade::ctrs::dot_prod(vel, nvec)*nvec);
            vec_t u_tan_vec  = vel - unorm;
            const real_t u_tan = spade::ctrs::array_norm(u_tan_vec);
            const real_t rho_f = q_f.p()/(q_f.T()*gas.R); //use "gas" to compute this!
            //compute length of ray point
            const auto& bndy_x = list.closest_points[i][0]; //get closest bndy point t ghost
            const auto& samp_x = ips_img [i*nlayers + ips_offset ];
            const auto& samp2_x= ips2_img[i*nlayers + ips_offset ];
            const coor_t y_f   = spade::ctrs::array_norm(samp_x - bndy_x); //We need to compute the distance
            const coor_t y2_f  = spade::ctrs::array_norm(samp2_x- bndy_x); //We need to compute the distance
            const real_t mu    = visc.get_visc(q_f);
  //#if (DEBUG==1)
  //            print("nvec  = ",nvec);
  //            print("u_tan = ",u_tan);
  //            print("y_f   = ",y_f);
  //            print("rho_f = ",rho_f);
  //#endif
              const auto upls = [=](const coor_t& y_plus)
              {
                  constexpr coor_t c_b  = 5.0333908790505579;
                  constexpr coor_t c_a1 = 8.148221580024245;
                  constexpr coor_t c_a2 = -6.9287093849022945;
                  constexpr coor_t c_b1 = 7.4600876082527945;
                  constexpr coor_t c_b2 = 7.468145790401841;
                  constexpr coor_t c_c1 = 2.5496773539754747;
                  constexpr coor_t c_c2 = 1.3301651588535228;
                  constexpr coor_t c_c3 = 3.599459109332379;
                  constexpr coor_t c_c4 = 3.6397531868684494;
                  return c_b + c_c1*log((y_plus+c_a1)*(y_plus+c_a1)+c_b1*c_b1) - c_c2*log((y_plus+c_a2)*(y_plus+c_a2)+c_b2*c_b2)
                  -c_c3*atan2(c_b1, y_plus+c_a1)-c_c4*atan2(c_b2, y_plus+c_a2);
              };
              
              const auto func = [&](const coor_t& u_tau)
              {
                  const coor_t u_plus_end0 = upls(rho_f*u_tau*y_f/mu);
                  const coor_t u_plus_end1 = u_tan/u_tau;
                  return u_plus_end0 - u_plus_end1;
              };
  
              const coor_t u_tau_ini = u_tan*0.1;
              const auto result = spade::num_algs::newton(u_tau_ini, func, spade::num_algs::num_deriv(coor_t(1.0e-6)), 100, coor_t(1.0e-6));
              tau_img[wm_offset + i] = rho_f*result.x*result.x;
              //
              coor_t eps     = 0.01*y2_f;
              coor_t u_plus  = upls(rho_f*result.x*y2_f/mu)*result.x;
              coor_t u2_plus = upls(rho_f*result.x*(y2_f+eps)/mu)*result.x;
              grad_img[wm_offset + i] = (u2_plus-u_plus)/eps;
              //
              // utau   =sqrt(tau_w/rho_f) & u+=u/utau & y+=rho_f*utau/mu*y_f
              // du+/dy+=1/utau*mu/(utau*rho_f)*du/dy=mu/(utau^2*rho_f)*du/dy=mu/tau_w*du/dy
              //
//#warning "HARDCODED SHEAR STRESS"
//            tau_img[wm_offset + i] = 9.86;
//            tau_img[wm_offset + i] = 21.0;

//              if ((wm_offset+i)%1000==0) printf("wm: dir = %d, i = %ld, grad_w = %f, tau_w = %f, u2+ = %f, u+ = %f, mu = %f, utang = %f \n",
//                                                dir,  wm_offset+i,grad_img[wm_offset + i],tau_img[wm_offset + i],u2_plus,u_plus,mu,u_tan);


            //   if (tau_img[wm_offset + i]!=tau_img[wm_offset + i] || tau_img[wm_offset + i]>1000.)
            //   {
            //     tau_img[wm_offset + i] = 0;
            //     printf("NaN occurred in the wall model, i=%ld, utan=%f, tauw=%f \n",i,u_tan,tau_img[wm_offset + i]);
            //     printf("nvec = (%f,%f,%f), utan=%f, y_f=%f, rho_f=%f, bndy_x=(%f,%f,%f), samp_x=(%f,%f,%f) \n",nvec[0],nvec[1],nvec[2],u_tan,y_f,rho_f,bndy_x[0],bndy_x[1],bndy_x[2],samp_x[0],samp_x[1],samp_x[2]);
            //   };

            // if (i==0)
            //   {
            //     printf("i = %ld, tau    = %f, nvec = (%f,%f,%f), utan=%f, y_f=%f, rho_f=%f, bndy_x=(%f,%f,%f), samp_x=(%f,%f,%f) \n",wm_offset+i,tau_img[wm_offset + i],nvec[0],nvec[1],nvec[2],u_tan,y_f,rho_f,bndy_x[0],bndy_x[1],bndy_x[2],samp_x[0],samp_x[1],samp_x[2]);
            //     printf("i = %ld, grad_w = %f",wm_offset+i,grad_img[wm_offset + i]);
            //   }
            // qw_img[wm_offset + i]  = -visc.get_diffuse()*q_f.T()/y_f;


            // qw_img[wm_offset + i]  = 0.0;
#if (DEBUG==1)
            print("results = ",result.its);
            print("results = ",result.eps);
            print("q_f = ",q_f);
            print("tau = ",tau_img[wm_offset + i] );
            char c;
            std::cin >> c;
#endif
        };
        spade::dispatch::execute(range, loop, dev);
        wm_offset  += size_here;
        ips_offset += size_here2;
      }
    }
}
