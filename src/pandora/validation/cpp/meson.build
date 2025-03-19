
py.extension_module(
    'validation_cpp',
    ['src/bindings.cpp', 'src/interpolated_disparity.cpp'],
    include_directories: ['includes'],
    subdir: 'pandora/validation/cpp',
    install: true,
    dependencies: [pybind11_dep],
)
