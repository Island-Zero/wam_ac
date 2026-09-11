"""Persistent model; native task loop and environment reuse, seed 42 per task."""
import argparse, importlib, json, os, random, time, traceback, sys
from pathlib import Path
from runtime import configure_environment, write_json


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--campaign',type=Path,required=True);ap.add_argument('--worker',type=int,required=True);args=ap.parse_args()
    out=args.campaign;write=write_json
    settings=json.loads((out/'run_config.json').read_text())['config']
    configure_environment(settings)
    import numpy as np
    import torch
    from hydra import compose, initialize_config_dir
    from hydra.utils import instantiate
    from omegaconf import OmegaConf
    from router import Router
    native=importlib.import_module('eval_libero_single')
    torch.set_num_threads(2);torch.set_num_interop_threads(2)
    up=Path(settings['fastwam_root'])
    with initialize_config_dir(version_base='1.3',config_dir=str(up/'configs')):
        cfg=compose(config_name='sim_libero',overrides=['task=libero_uncond_2cam224_1e-4',f'ckpt={settings["checkpoint"]}',f'EVALUATION.dataset_stats_path={settings["dataset_stats"]}',f'output_dir={out}',f'EVALUATION.output_dir={out}',f'seed={settings["seed"]}',f'EVALUATION.sigma_shift={settings["sigma_shift"]}','EVALUATION.num_inference_steps=10','EVALUATION.replan_steps=10'])
    native.set_global_seed(settings["seed"],get_worker_init_fn=False)
    model=instantiate(cfg.model,model_dtype=native._mixed_precision_to_model_dtype(cfg.mixed_precision),device='cuda')
    native._load_model_checkpoint(model,str(cfg.ckpt));model=model.to('cuda').eval()
    processor=instantiate(cfg.data.train.processor).eval()
    processor.set_normalizer_from_stats(native.load_dataset_stats_from_json(str(cfg.EVALUATION.dataset_stats_path)))
    # Restore the post-initialization state for every task, as in a fresh native process.
    rng=(random.getstate(),np.random.get_state(),torch.get_rng_state(),torch.cuda.get_rng_state_all())
    router=Router(model)
    OmegaConf.save(cfg,out/f'worker_{args.worker}_config.yaml',resolve=True)
    native.save_rollout_video=lambda *a,**k:None
    episode_fn=native.run_single_episode;env_fn=native.get_libero_env
    environments=[];context={}
    def get_env(*a,**kw):
        env,desc=env_fn(*a,**kw);environments.append(env);return env,desc
    def episode(*a,**kw):
        router.configure(context['job']['method'],context['job']['method'].get('horizon',10))
        result=episode_fn(*a,**kw)
        idx=kw['episode_idx'];rows=router.rows
        path=context['path']/f'episode_{idx:02d}.json'
        write(path,dict(trial_idx=idx,success=bool(result[0]),chunks=len(rows),forwards=sum(r['actual_nfe'] for r in rows),rows=rows))
        router.tensors=[]
        return result
    native.get_libero_env=get_env;native.run_single_episode=episode
    write(out/f'worker_{args.worker}_ready.json',dict(pid=os.getpid(),start_ticks=Path(f'/proc/{os.getpid()}/stat').read_text().split()[21],time=time.time()))
    while not (out/'STOP').exists():
        claim=None
        for q in sorted((out/'queue').glob('*.json')):
            dest=out/'running'/q.name
            try:q.rename(dest);claim=dest;break
            except FileNotFoundError:continue
        if claim is None:time.sleep(2);continue
        job=json.loads(claim.read_text());path=out/'tasks'/job['id'];path.mkdir(parents=True,exist_ok=True)
        context.update(job=job,path=path);started=time.time()
        try:
            random.setstate(rng[0]);np.random.set_state(rng[1]);torch.set_rng_state(rng[2]);torch.cuda.set_rng_state_all(rng[3])
            cfg.EVALUATION.task_suite_name=job['suite'];cfg.EVALUATION.task_id=job['task_id'];cfg.EVALUATION.num_trials=job['trials']
            print('START',job['id'],flush=True)
            suite=native.benchmark.get_benchmark_dict()[job['suite']]()
            states=suite.get_task_init_states(job['task_id'])
            if len(states)<job['trials']:raise ValueError('Insufficient distinct initial states')
            import hashlib
            write(path/'job.json',dict(job,init_hashes=[hashlib.sha256(np.asarray(x).tobytes()).hexdigest() for x in states[:job['trials']]]))
            r=native.run_single_task(suite.get_task(job['task_id']),states,model,processor,cfg,path,path,action_horizon=32,input_w=448,input_h=224,model_device='cuda')
            eps=[json.loads((path/f'episode_{i:02d}.json').read_text()) for i in range(job['trials'])]
            assert sorted(r['success_episodes']+r['failure_episodes'])==list(range(job['trials']))
            write(path/'result.json',dict(job,**r,total_episodes=job['trials'],chunks=sum(e['chunks'] for e in eps),forwards=sum(e['forwards'] for e in eps),duration=time.time()-started))
            claim.rename(out/'done'/claim.name)
            print('DONE',job['id'],r['successes'],flush=True)
        except Exception:
            write(path/'error.json',dict(error=traceback.format_exc(),job=job))
            claim.rename(out/'failed'/claim.name)
            traceback.print_exc()
            # Fail closed on sampler, numerical or evaluation errors; manager reports them.
            raise
        finally:
            for env in environments:env.close()
            environments.clear();router.rows=[];router.tensors=[]
if __name__=='__main__':main()
