
py.extension_module(
    'cost_volume_confidence_cpp',
    [
        'src/cost_volume_confidence_tools.cpp',
        'src/bindings.cpp',
        'src/ambiguity.cpp',
        'src/interval_bounds.cpp',
        'src/risk.cpp',
    ],
    include_directories: ['includes'],
    subdir: 'pandora/cost_volume_confidence/cpp',
    install: true,
    dependencies: [pybind11_dep],
)
