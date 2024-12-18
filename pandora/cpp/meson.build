
py.extension_module(
  'img_tools_cpp',
  ['bindings_img_tools.cpp', 'img_tools.cpp'],
  include_directories: ['includes'],
  subdir: 'pandora',
  install: true,
  dependencies : [pybind11_dep, eigen_dep],
)

py.extension_module(
  'interval_tools_cpp',
  ['bindings_interval_tools.cpp', 'interval_tools.cpp'],
  include_directories: ['includes'],
  subdir: 'pandora',
  install: true,
  dependencies : [pybind11_dep, eigen_dep],
)