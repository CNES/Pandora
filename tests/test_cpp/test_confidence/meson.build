test(
    'Confidence',
    executable(
        'test_confidence',
        [
            'test_confidence.cpp',
            '../../../pandora/cost_volume_confidence/cpp/src/cost_volume_confidence_tools.cpp',
        ],
        include_directories: ['../../../pandora/cost_volume_confidence/cpp/includes/'],
        dependencies: [doctest_dep, pybind11_dep, python_dep],
    ),
)
