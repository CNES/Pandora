
py.extension_module(
    'matching_cost_cpp',
    ['src/bindings.cpp', 'src/census.cpp'],
    include_directories: ['includes'],
    subdir: 'pandora/matching_cost/cpp',
    install: true,
    dependencies: [pybind11_dep],
)
