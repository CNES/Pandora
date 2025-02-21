project(
    'pandora',
    'cpp',
    version: run_command('version.sh', 'get-vcs', check: true).stdout().strip(),
    default_options: ['cpp_std=c++17'],
    meson_version: '>=1.1',
)

meson.add_dist_script(
    meson.project_source_root() / 'version.sh',
    'set-dist',
    meson.project_version(),
)

py = import('python').find_installation(pure: false)

pybind11_dep = dependency('pybind11')
doctest_dep = dependency('doctest', required: get_option('build_cpp_tests'))
python_dep = py.dependency(embed: true, required: get_option('build_cpp_tests'))

install_subdir(
    'pandora',
    install_dir: py.get_install_dir() / 'pandora',
    strip_directory: true,
)

subdir('pandora/cpp')

subdir('pandora/validation/cpp')

subdir('pandora/aggregation/cpp')

subdir('pandora/cost_volume_confidence/cpp')

subdir('pandora/refinement/cpp')

subdir('pandora/matching_cost/cpp')

if get_option('build_cpp_tests').enabled()
    subdir('tests/test_cpp/')
endif
