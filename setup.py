#!/usr/bin/env python
from __future__ import division, print_function, absolute_import

from os.path import join

def configuration(parent_package='',top_path=None):
    from numpy.distutils.misc_util import Configuration
    from numpy.distutils.system_info import get_info
    config = Configuration('expokitpy', parent_package, top_path)


    expokit_src = [join('expokit','expokit.f'),
                   join('expokit','blas.f'),
                   join('expokit','lapack.f'),
                  ]
    config.add_library('expokit', 
                        sources=expokit_src)
    # dop
    config.add_extension('_expokit',
                         sources=['expokit.pyf'],
                         libraries=['expokit'],
                         depends=expokit_src)


    return config

if __name__ == '__main__':
    from numpy.distutils.core import setup
    setup(**configuration(top_path='').todict())
