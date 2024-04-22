#pragma once

#include "spade.h"

namespace local
{
    
    template <typename coor_t>
    struct comp_refine_info_t
    {
        int comp_id;
        bool any;
        coor_t dx_target;
        int axis;
        spade::ctrs::array<coor_t, 3> aspect;
    };
    
    template <typename coor_t>
    const scidf::iconversion_t& operator >> (const scidf::iconversion_t& i, comp_refine_info_t<coor_t>& c)
    {
        
        std::vector<std::string> raws;
        i >> raws;
        if (raws.size() != 4) throw scidf::sdf_exception("Wrong number of fields in comp id info parse");
        const auto parse_member = [&](const std::string& name, const std::string& raw_in, auto& member)
        {
            std::vector<std::string> arr = scidf::str::split(raw_in, ":");
            if (arr.size() != 2) throw scidf::sdf_exception("Ill-formatted comp id info parse");
            scidf::iconversion_t i0(arr[0]);
            std::string name_found;
            i0 >> name_found;
            if (name_found != name) throw scidf::sdf_exception("Ill-formatted comp id info parse: found \"" + name_found + "\", but expecting \"" + name + "\"");
            scidf::iconversion_t i1(arr[1]);
            i1 >> member;
        };
        
        std::string thing;
        parse_member("comp",   raws[0], thing);
        parse_member("dx",     raws[1], c.dx_target);
        parse_member("axis",   raws[2], c.axis);
        parse_member("aspect", raws[3], c.aspect);
        
        if (thing == "any")
        {
            c.any = true;
            c.comp_id = -1;
        }
        else
        {
            scidf::iconversion_t i0(thing);
            i0 >> c.comp_id;
            c.any = false;
        }
        
        return i;
    }
}