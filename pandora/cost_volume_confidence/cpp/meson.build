
py.extension_module(
  'cost_volume_confidence_cpp',
  ['bindings.cpp', 'ambiguity.cpp', 'interval_bounds.cpp', 'risk.cpp'],
  include_directories: ['includes'],
  subdir: 'pandora',
  install: true,
  dependencies : [pybind11_dep, eigen_dep],
)