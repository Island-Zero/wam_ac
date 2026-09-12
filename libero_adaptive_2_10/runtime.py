"""Shared I/O and external dependency configuration; no legacy experiment imports."""
import json,os,sys
from pathlib import Path


def write_json(path, data):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(data,indent=2,allow_nan=False)+'\n');tmp.replace(path)


def configure_environment(config):
    up=Path(config['fastwam_root']);libero=Path(config['libero_root'])
    sys.path[:0]=[str(up/'experiments/libero'),str(up),str(up/'src'),str(libero)]
    values=dict(LIBERO_CONFIG_PATH=config['libero_config_path'],DIFFSYNTH_MODEL_BASE_PATH=config['model_base_path'],DIFFSYNTH_DOWNLOAD_SOURCE='modelscope',MUJOCO_GL='egl',PYOPENGL_PLATFORM='egl',TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD='1',TOKENIZERS_PARALLELISM='false')
    os.environ.update(values)


def method_config(name, config):
    if name=='adaptive':return dict(name=name,mode='adaptive',metric='relative_l2',threshold=config['adaptive_threshold'],horizon=10)
    if name in ('10-step','2-step'):return dict(name=name,mode='fixed',nfe=int(name.split('-')[0]))
    raise ValueError(f'Unknown method {name}')


def load_config(path):
    """Resolve resource paths relative to the configuration file."""
    path=Path(path).expanduser().resolve()
    config=json.loads(path.read_text())
    for key in ['fastwam_root','libero_root','checkpoint','dataset_stats','libero_config_path','model_base_path']:
        resource=Path(os.path.expandvars(config[key])).expanduser()
        if not resource.is_absolute():resource=path.parent/resource
        config[key]=str(resource.resolve())
    return config
