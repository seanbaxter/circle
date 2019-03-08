// _Variant_storage is a union
// _M_first and _M_rest are variant members in the union

std::variant<int, double, vec3_t, std::vector<short> >

{
  <std::__detail::__variant::_Variant_base<int, double, vec3_t, std::vector<short, std::allocator<short> > >> = {
    <std::__detail::__variant::_Move_assign_base<false, int, double, vec3_t, std::vector<short, std::allocator<short> > >> = {
      <std::__detail::__variant::_Copy_assign_base<false, int, double, vec3_t, std::vector<short, std::allocator<short> > >> = {
        <std::__detail::__variant::_Move_ctor_base<false, int, double, vec3_t, std::vector<short, std::allocator<short> > >> = {
          <std::__detail::__variant::_Copy_ctor_base<false, int, double, vec3_t, std::vector<short, std::allocator<short> > >> = {
            <std::__detail::__variant::_Variant_storage<false, int, double, vec3_t, std::vector<short, std::allocator<short> > >> = {
              _M_u = {
                _M_first = {
                  _M_storage = 1374389535
                },
                _M_rest = {
                  _M_first = {
                    _M_storage = 3.1400000000000001
                  },
                  _M_rest = {
                    _M_first = {
                      _M_storage = {
                        x = 1.26443839e+11,
                        y = 2.14249992,
                        z = 5.88422041e-39
                      }
                    },
                    _M_rest = {
                      _M_first = {
                        _M_storage = {
                          _M_storage = "\037\205\353Q\270\036\t@\320\022@\000\000\000\000\000\020\006@\000\000\000\000"
                        }
                      },
                      _M_rest = {<No data fields>}
                    }
                  }
                }
              },
              _M_index = 1 '\001',
              static _S_vtable = <optimized out>
            }, <No data fields>}, <No data fields>}, <No data fields>}, <No data fields>}, <No data fields>},
  <std::_Enable_default_constructor<true, std::variant<int, double, vec3_t, std::vector<short, std::allocator<short> > > >> = {<No data fields>},
  <std::_Enable_copy_move<true, true, true, true, std::variant<int, double, vec3_t, std::vector<short, std::allocator<short> > > >> = {<No data fields>},
  members of std::variant<int, double, vec3_t, std::vector<short, std::allocator<short> > >:
  static __accepted_index = 1,
  static __exactly_once = true
}