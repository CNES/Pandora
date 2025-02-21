
py.extension_module(
    'img_tools_cpp',
    ['src/bindings_img_tools.cpp', 'src/img_tools.cpp'],
    include_directories: ['includes'],
    subdir: 'pandora/cpp',
    install: true,
    dependencies: [pybind11_dep],
)

py.extension_module(
    'interval_tools_cpp',
    ['src/bindings_interval_tools.cpp', 'src/interval_tools.cpp'],
    include_directories: ['includes'],
    subdir: 'pandora/cpp',
    install: true,
    dependencies: [pybind11_dep],
)
