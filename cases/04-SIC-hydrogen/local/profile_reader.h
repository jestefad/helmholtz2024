#pragma once

#include "spade.h"
#include <fstream>

namespace local
{
  //
  _sp_hybrid inline int randGen(unsigned int& m_w,unsigned int& m_z)
  {
    m_z = 36969 * (m_z & 65535) + (m_z >> 16);
    m_w = 18000 * (m_w & 65535) + (m_w >> 16);
    return  (m_z << 16) + m_w; /* 32-bit result */
  }
  //
  template <typename ys_t,typename profile_t, typename gas_t,typename q_t>
    inline void read_lam_boundary_layer(const std::string& filename,
                                        ys_t&        ys,
                                        profile_t&   profile,
                                        gas_t&       gas,
                                        q_t&         qref)
    {
      std::fstream file;
      file.open(filename);
      std::string line;
      using prim_t = profile_t::value_type;
      //
      while (std::getline(file, line))
        {
          std::istringstream iss(line);
          double val;
          iss >> val;
          ys.push_back(val);
          // read state vector
          // eta-coordinate, density, u-velocity, scaled v-velocity, temperature
          prim_t q;
          double rho;
          iss >> rho;
          iss >> q.u();
          iss >> q.v();
          q.w() = 0.;
          iss >> q.T();
          q.p()=rho*q.T();
          //normalize flow quantities
          q.u() *= qref.u();
          q.v() *= qref.u();
          q.w() *= qref.u();
          q.p() *= qref.p();
          q.T() *= qref.T();
          //
          profile.push_back(q);
          //
        }
      ys.transfer();
      profile.transfer();
    }

  template <typename realvec_t, typename profile_t, typename gas_t,typename q_t>
    inline void read_turb_boundary_layer(const std::string& filename,
                                        realvec_t&   ys,
                                        realvec_t&   R11,
                                        realvec_t&   R22,
                                        realvec_t&   R33,
                                        realvec_t&   R12,
                                        realvec_t&   R13,
                                        realvec_t&   R23,
                                        profile_t&   profile,
                                        gas_t&       gas,
                                        q_t&         qref)
    {
      std::fstream file;
      file.open(filename);
      std::string line;
      using prim_t = profile_t::value_type;
      //
      while (std::getline(file, line))
        {
          std::istringstream iss(line);
          //y/d99, y+, urms, vrms, wrms, uv, uw, vw, umed, vmed, wmed
          double val;
          //y/d99
          iss >> val;
          ys.push_back(val);
          //y+
          iss >> val;
          double yplus=val;
          // read state vector
          // urms
          iss >> val;
          R11.push_back(val*val);
          // vrms
          iss >> val;
          R22.push_back(val*val);
          // wrms
          iss >> val;
          R33.push_back(val*val);
          // uv
          iss >> val;
          R12.push_back(val);
          // uw
          iss >> val;
          R13.push_back(val);
          // vw
          iss >> val;
          R23.push_back(val);
          q_t q;
          //u-mean
          iss >> val;
          q.u() = val;
          //v-mean
          iss >> val;
          q.v() = val;
          //w-mean
          iss >> val;
          q.w() = val;
          // temperature
          q.T() = qref.T();
          // pressure
          q.p() = qref.p();
          //
          profile.push_back(q);
          //
        }
      ys.transfer();
      R11.transfer();
      R22.transfer();
      R33.transfer();
      R12.transfer();
      R13.transfer();
      R23.transfer();
      profile.transfer();
    }
}
