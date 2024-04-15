#pragma once

#include <iostream>
#include <string>

namespace debug
{
    template <typename thing_t> void print_type(const thing_t& t)
    {
        //g++ only
        std::string pf(__PRETTY_FUNCTION__);
        std::string srch = "thing_t = ";
        std::size_t start = pf.find(srch) + srch.length();
        std::size_t end = pf.length()-1;
        std::cout << pf.substr(start, end-start) << std::endl;
    }
}