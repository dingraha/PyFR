# -*- coding: utf-8 -*-

from pyfr.parallel_partitioners.base import BaseParallelPartitioner
from pyfr.parallel_partitioners.parmetis import ParMETISPartitioner
from pyfr.util import subclass_where


def get_parallel_partitioner(name, *args, **kwargs):
    return subclass_where(BaseParallelPartitioner, name=name)(*args, **kwargs)
