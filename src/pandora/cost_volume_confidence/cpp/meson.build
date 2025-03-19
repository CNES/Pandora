
cost_volume_confidence_includes = include_directories('includes')
cost_volume_confidence_src = files('src/cost_volume_confidence_tools.cpp')

py.extension_module(
    'cost_volume_confidence_cpp',
    [
        cost_volume_confidence_src,
        'src/bindings.cpp',
        'src/ambiguity.cpp',
        'src/interval_bounds.cpp',
        'src/risk.cpp',
    ],
    include_directories: cost_volume_confidence_includes,
    subdir: 'pandora/cost_volume_confidence/cpp',
    install: true,
    dependencies: [pybind11_dep],
)
