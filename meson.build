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

subdir(
  'pandora/cpp'
)

subdir(
  'pandora/validation/cpp'
)

subdir(
  'pandora/aggregation/cpp'
)

subdir(
  'pandora/cost_volume_confidence/cpp'
)

subdir(
  'pandora/refinement/cpp'
)