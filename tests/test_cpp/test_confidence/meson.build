test(
    'Confidence',
    executable(
        'test_confidence',
        ['test_confidence.cpp', cost_volume_confidence_src],
        include_directories: cost_volume_confidence_includes,
        dependencies: [doctest_dep, pybind11_dep, python_dep],
    ),
)
