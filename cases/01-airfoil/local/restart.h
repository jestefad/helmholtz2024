#pragma once

#include <fstream>

namespace local
{

  template <typename array_t>
  inline void write_restart(const std::string& filename,
                            const array_t&     array)
  {
    std::vector<typename array_t::value_type> raw = array.data;
    std::fstream file;
    file.open(filename,std::ios::trunc|std::ios::binary|std::ios::out);
    file.write(reinterpret_cast<char* >(&raw[0]),sizeof(typename array_t::value_type)*raw.size());
  }

  template <typename array_t>
    inline void read_restart(const std::string& filename,
                             array_t&     array)
    {
      std::vector<typename array_t::value_type> raw;
      raw.resize(array.data.size());
      std::fstream file;
      file.open(filename,std::ios::binary|std::ios::in);
      if (file.fail())
        {
          throw spade::except::sp_exception("Attempted to open non-existent file: " + filename);
        }
      file.read(reinterpret_cast<char* >(&raw[0]),sizeof(typename array_t::value_type)*raw.size());
      array.data = raw;
    }
}
