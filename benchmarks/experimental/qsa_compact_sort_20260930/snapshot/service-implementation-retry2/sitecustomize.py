"""Opt-in study shim; preserve production shim and stage source gate."""
import os
import runpy
from pathlib import Path

if os.environ.get('STEP58_RAW_GENERATION_PATH'):
    import logging
    from logging.handlers import RotatingFileHandler
    path = Path(os.environ['STEP58_RAW_GENERATION_PATH'])
    path.parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger('localai.raw_generation')
    logger.setLevel(logging.INFO)
    logger.propagate = False
    if not logger.handlers:
        handler = RotatingFileHandler(path, maxBytes=64 * 1024**2, backupCount=7)
        handler.setFormatter(logging.Formatter('%(message)s'))
        logger.addHandler(handler)
        os.chmod(path, 0o600)

if not os.environ.get('STEP58_SERVICE_CONFIG') and os.environ.get('STEP58_ORIGINAL_SHIM'):
    original = Path(os.environ['STEP58_ORIGINAL_SHIM']).resolve()
    assert original != Path(__file__).resolve()
    runpy.run_path(str(original), run_name='step58_original_sitecustomize')

if os.environ.get('STEP58_SERVICE_CONFIG'):
    try:
        import json
        from io_contract import require_hash, save, validate_config
        cfg = validate_config(json.loads(
            Path(os.environ['STEP58_SERVICE_CONFIG']).read_text()))
        original = Path(cfg['original_shim']).resolve()
        assert original != Path(__file__).resolve(), 'shim self-chain'
        require_hash(original, cfg['original_shim_sha256'])
        runpy.run_path(str(original), run_name='step58_original_sitecustomize')
        from service_hook import install_early
        install_early(cfg)
    except BaseException as error:
        # Python normally swallows sitecustomize exceptions: fail startup.
        from io_contract import save
        output = Path(cfg['output_dir']) if 'cfg' in globals() else (
            Path(os.environ['STEP58_SERVICE_CONFIG']).parent)
        save(output / f'hook-load-error-pid{os.getpid()}.json',
             {'status': 'ERROR', 'error': repr(error)})
        os._exit(78)
