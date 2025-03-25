
py.extension_module(
    'refinement_cpp',
    [
        'src/bindings.cpp',
        'src/refinement_tools.cpp',
        'src/refinement.cpp',
        'src/vfit.cpp',
        'src/quadratic.cpp',
    ],
    include_directories: ['includes'],
    subdir: 'pandora/refinement/cpp',
    install: true,
    dependencies: [pybind11_dep],
)
