set -x
circle -c info.cxx
circle -c get_args.cxx
circle -c get_args_cia.cxx
circle print.cxx && ./print
circle -c rebind.cxx
circle -c rotate.cxx
circle -c rotate2.cxx
circle -c is_specialization.cxx
circle -c is_specialization2.cxx
circle forward.cxx && ./forward
circle -c pairs.cxx
circle -c step.cxx
circle -c step2.cxx
circle -c power.cxx
circle -c power2.cxx
circle -c product.cxx
circle -c group.cxx
circle -c repeat.cxx
# circle -c life.cxx
circle -c sort.cxx
circle -c sort2.cxx
circle -c search.cxx
circle -c search_traits.cxx
circle variant.cxx && ./variant

# Traits
circle -c sort3.cxx
circle -c sort4.cxx
circle -c unique.cxx