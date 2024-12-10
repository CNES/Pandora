
py.extension_module(
  'refinement_cpp',
  ['bindings.cpp', 'refinement.cpp', 'vfit.cpp', 'quadratic.cpp'],
  include_directories: ['includes'],
  subdir: 'pandora',
  install: true,
  dependencies : [pybind11_dep, eigen_dep],
)