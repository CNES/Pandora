
matching_cost_includes = include_directories('includes')
matching_cost_src = files('src/matching_cost.cpp')

py.extension_module(
    'matching_cost_cpp',
    ['src/bindings.cpp', 'src/census.cpp', matching_cost_src],
    include_directories: matching_cost_includes,
    subdir: 'pandora/matching_cost/cpp',
    install: true,
    dependencies: [pybind11_dep],
)
