
py.extension_module(
  'aggregation_cpp',
  ['bindings.cpp', 'aggregation.cpp'],
  include_directories: ['includes'],
  subdir: 'pandora',
  install: true,
  dependencies : [pybind11_dep, eigen_dep],
)