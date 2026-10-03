"""One bounded real-NumPy CPU check; retained create-only synthetic fixtures."""
import argparse
import json
import os
from pathlib import Path
import resource
import sys
import unittest

from io_contract import save


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--output-dir',required=True)
    args=parser.parse_args()
    directory=Path(args.output_dir)
    directory.mkdir(parents=True,exist_ok=False)
    assert os.environ.get('CUDA_VISIBLE_DEVICES')==''
    assert all(os.environ.get(name)=='1' for name in
               ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'))
    assert 'torch' not in sys.modules
    os.environ['STEP58_CPU_EVIDENCE_DIR']=str(directory/'fixtures')
    import numpy
    import test_retry_serialization
    suite=unittest.defaultTestLoader.loadTestsFromModule(test_retry_serialization)
    result=unittest.TextTestRunner(verbosity=2).run(suite)
    receipt={'status':'PASS_SOURCE_NUMPY_WRAPPER_DRAIN_NOT_SERVICE' if result.wasSuccessful()
             else 'FAIL_CPU_SERIALIZATION_FIXTURE',
             'tests_run':result.testsRun,'failures':len(result.failures),'errors':len(result.errors),
             'python':sys.version,'executable':sys.executable,'numpy':numpy.__version__,
             'torch_imported':'torch' in sys.modules,'cuda_visible_devices':os.environ['CUDA_VISIBLE_DEVICES'],
             'max_rss_kib_linux':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
             'gpu_parts':'model/attention/counter/allocator/synchronize explicitly stubbed',
             'actual_parts':'NumPy scalar + pinned native source AST + installed Runtime wrapper/event/full drain/io save',
             'not_proved':['actual service readiness/admission','candidate float/graph behavior','W2 numerics/performance']}
    assert not receipt['torch_imported']
    save(directory/'RESULT.json',receipt)
    print(json.dumps(receipt,sort_keys=True))
    return 0 if result.wasSuccessful() else 1


if __name__=='__main__':
    raise SystemExit(main())
