project(
  'pandora',
  'cpp',
  version: run_command(
    'version.sh',
    'get-vcs',
  ).stdout().strip(),
  default_options: [
    'cpp_std=c++17',
  ],
)

meson.add_dist_script('version.sh', 'set-dist', meson.project_version())

py = import('python').find_installation(pure: false)

pybind11_dep = dependency('pybind11')
eigen_dep = dependency('eigen3')

install_subdir(
  'pandora',
  install_dir: py.get_install_dir() / 'pandora',
  strip_directory: true,
)

py.extension_module(
  'pandora_lib_c',
  'pandora/matching_cost/lib_c.cpp',
  install: true,
  dependencies : [pybind11_dep, eigen_dep],
)