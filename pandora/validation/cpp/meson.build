
py.extension_module(
  'validation_cpp',
  ['bindings.cpp', 'interpolated_disparity.cpp'],
  include_directories: ['includes'],
  subdir: 'pandora',
  install: true,
  dependencies : [pybind11_dep, eigen_dep],
)