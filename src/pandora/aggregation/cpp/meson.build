
py.extension_module(
    'aggregation_cpp',
    ['src/bindings.cpp', 'src/aggregation.cpp'],
    include_directories: ['includes'],
    subdir: 'pandora/aggregation/cpp',
    install: true,
    dependencies: [pybind11_dep],
)
