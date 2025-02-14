project(
  'pandora',
  'cpp',
  version: run_command(
    'python',
    meson.source_root() / 'version.py',
    'get-vcs',
  ).stdout().strip(),
  default_options: [
    'cpp_std=c++17',
  ],
)

meson.add_dist_script('python', meson.source_root() / 'version.py', 'set-dist', meson.project_version())

py = import('python').find_installation(pure: false)

pybind11_dep = dependency('pybind11')
doctest_dep = dependency('doctest')
python_dep = py.dependency(embed:true)


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

subdir(
  'pandora/matching_cost/cpp'
)

subdir(
  'tests/test_cpp/'
)