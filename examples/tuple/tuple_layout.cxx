typedef std::tuple<int, double, vec3_t, std::vector<short> > my_tuple_t;

{
  <std::_Tuple_impl<0, int, double, vec3_t, std::vector<short, std::allocator<short> > >> = {
    <std::_Tuple_impl<1, double, vec3_t, std::vector<short, std::allocator<short> > >> = {
      <std::_Tuple_impl<2, vec3_t, std::vector<short, std::allocator<short> > >> = {
        <std::_Tuple_impl<3, std::vector<short, std::allocator<short> > >> = {
          <std::_Head_base<3, std::vector<short, std::allocator<short> >, false>> = {
            _M_head_impl = {
              <std::_Vector_base<short, std::allocator<short> >> = {
                _M_impl = {
                  <std::allocator<short>> = {
                    <__gnu_cxx::new_allocator<short>> = {<No data fields>}, <No data fields>},
                  members of std::_Vector_base<short, std::allocator<short> >::_Vector_impl:
                  _M_start = 0x616e90,
                  _M_finish = 0x616e94,
                  _M_end_of_storage = 0x616e94
                }
              }, <No data fields>}
          }, <No data fields>},
        <std::_Head_base<2, vec3_t, false>> = {
          _M_head_impl = {
            x = 5,
            y = 6,
            z = 7
          }
        }, <No data fields>},
      <std::_Head_base<1, double, false>> = {
        _M_head_impl = 3.1400000000000001
      }, <No data fields>},
    <std::_Head_base<0, int, false>> = {
      _M_head_impl = 100
    }, <No data fields>
  }, <No data fields>
}