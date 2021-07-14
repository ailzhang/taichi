import glob
import os
import sys
import subprocess
import shutil
import platform
from pathlib import Path

from setuptools import setup, Extension, find_packages
from setuptools.command.build_ext import build_ext

classifiers = [
    'Development Status :: 2 - Pre-Alpha',
    'Topic :: Software Development :: Compilers',
    'Topic :: Multimedia :: Graphics',
    'Topic :: Games/Entertainment :: Simulation',
    'Intended Audience :: Science/Research',
    'Intended Audience :: Developers',
    'License :: OSI Approved :: MIT License',
    'Programming Language :: Python :: 3.6',
    'Programming Language :: Python :: 3.7',
    'Programming Language :: Python :: 3.8',
    'Programming Language :: Python :: 3.9',
]

# data_files = glob.glob('python/lib/*')
# print(data_files)
packages = find_packages() + ['taichi.examples']
print(packages)

project_name = 'taichi'
version = '0.7.25'

def get_os_name():
    name = platform.platform()
    # in python 3.8, platform.platform() uses mac_ver() on macOS
    # it will return 'macOS-XXXX' instead of 'Darwin-XXXX'
    if name.lower().startswith('darwin') or name.lower().startswith('macos'):
        return 'osx'
    elif name.lower().startswith('windows'):
        return 'win'
    elif name.lower().startswith('linux'):
        return 'linux'
    assert False, "Unknown platform name %s" % name

class CMakeExtension(Extension):
    def __init__(self, name):
        Extension.__init__(self, name, sources=[])

class CMakeBuild(build_ext):
    def run(self):
        try:
            out = subprocess.check_output(['cmake', '--version'])
        except OSError:
            raise RuntimeError(
                "CMake must be installed to build the following extensions: " +
                ", ".join(e.name for e in self.extensions))
        # CMakeLists.txt is in the same directory as this setup.py file
        # FIXME: currently we run setup.py from python/ folder,
        #        this should be moved up to root folder.
        # cmake_list_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
        cmake_list_dir = os.path.abspath(os.path.dirname(__file__))
        self.build_temp = os.path.join(cmake_list_dir, 'build')
        print('AILING', self.build_temp)

        build_directory = os.path.abspath(self.build_temp)
        llvm_as_path = shutil.which('llvm-as')

        cmake_args = [
            '-DCMAKE_LIBRARY_OUTPUT_DIRECTORY=' + build_directory,
            '-DPYTHON_EXECUTABLE=' + sys.executable,
            '-DTI_VERSION_STRING=' + version,
            '-DLLVM_AS=' + llvm_as_path,
        ]

        cfg = 'Debug' if self.debug else 'Release'
        build_args = ['--config', cfg]

        cmake_args += ['-DCMAKE_BUILD_TYPE=' + cfg]

        # Assuming Makefiles
        build_args += ['--', '-j8']

        self.build_args = build_args

        env = os.environ.copy()
        #env['CXXFLAGS'] = '{} -DVERSION_INFO=\\"{}\\"'.format(
        #    env.get('CXXFLAGS', ''),
        #    self.distribution.get_version())
        if not os.path.exists(self.build_temp):
            os.makedirs(self.build_temp)

        print('-'*10, 'Running CMake prepare', '-'*40)
        subprocess.check_call(['cmake', cmake_list_dir] + cmake_args,
                              cwd=self.build_temp, env=env)

        print('-'*10, 'Building extensions', '-'*40)
        cmake_cmd = ['cmake', '--build', '.'] + self.build_args
        subprocess.check_call(cmake_cmd,
                              cwd=self.build_temp)

        shutil.rmtree('taichi/lib', ignore_errors=True)
        shutil.rmtree('taichi/tests', ignore_errors=True)
        shutil.rmtree('taichi/examples', ignore_errors=True)
        shutil.rmtree('taichi/assets', ignore_errors=True)

        os.makedirs('taichi/lib', exist_ok=True)
        if get_os_name() == 'linux':
            shutil.copy(os.path.join(self.build_temp, 'libtaichi_core.so'), 'taichi/lib/taichi_core.so')
        elif get_os_name() == 'osx':
            shutil.copy(os.path.join(self.build_temp, 'libtaichi_core.dylib'),
                        'taichi/lib/taichi_core.so')
        else:
            shutil.copy('../runtimes/RelWithDebInfo/taichi_core.dll',
                        'taichi/lib/taichi_core.pyd')

        runtime_dir = 'taichi/csrc/runtime/llvm'
        for f in os.listdir(runtime_dir):
            if f.startswith('runtime_') and f.endswith('.bc'):
                print(f"Fetching runtime file {f}")
                shutil.copy(os.path.join(runtime_dir, f), 'taichi/lib')

        shutil.copytree('examples', './taichi/examples')
        shutil.copytree('external/assets', './taichi/assets')

setup(name=project_name,
      packages=packages,
      version=version,
      description='The Taichi Programming Language',
      author='Taichi developers',
      author_email='yuanmhu@gmail.com',
      url='https://github.com/taichi-dev/taichi',
      install_requires=[
          'numpy',
          'pybind11>=2.5.0',
          'sourceinspect>=0.0.4',
          'colorama',
          'astor',
      ],
      # data_files=[('lib', data_files)],
      data_files=[('lib', glob.glob('taichi/lib/*'))],
      keywords=['graphics', 'simulation'],
      license='MIT',
      include_package_data=True,
      entry_points={
          'console_scripts': [
              'ti=taichi.main:main',
          ],
      },
      classifiers=classifiers,
      ext_modules=[CMakeExtension('taichi_core')],
      cmdclass=dict(build_ext=CMakeBuild),
      has_ext_modules=lambda: True)

# Note: this is a template setup.py used by python/build.py
