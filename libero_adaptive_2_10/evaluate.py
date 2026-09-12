"""Resumable task queue for adaptive, 10-step and 2-step LIBERO evaluation."""
import argparse,fcntl,hashlib,json,math,os,signal,subprocess,sys,time,traceback
from pathlib import Path
from runtime import write_json,method_config,load_config
from summarize import summarize
ROOT=Path(__file__).resolve().parent
SUITES=['libero_spatial','libero_object','libero_goal','libero_10']
METHODS=['adaptive','10-step','2-step']


def parse_args():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',type=Path,default=ROOT/'configs/local.json')
    p.add_argument('--methods',nargs='+',choices=METHODS,default=METHODS)
    p.add_argument('--workers',type=int,choices=[1,2,3],default=3)
    p.add_argument('--gpu',default='0')
    p.add_argument('--output',type=Path,default=ROOT/'outputs/libero_2000')
    p.add_argument('--suites',nargs='+',choices=SUITES,default=SUITES)
    p.add_argument('--task-ids',nargs='+',type=int,choices=range(10),default=list(range(10)))
    p.add_argument('--trials',type=int,default=50)
    p.add_argument('--detach',action='store_true',help='Run independently of the terminal')
    p.add_argument('--dry-run',action='store_true',help='Validate paths and print plan without GPU initialization')
    p.add_argument('--retry-failed',action='store_true',help='Archive failed tasks and retry from trial 0')
    return p.parse_args()


def make_plan(args):
    config=load_config(args.config)
    for k in ['fastwam_root','libero_root','checkpoint','dataset_stats','libero_config_path','model_base_path']:
        if not Path(config[k]).exists():raise ValueError(f'Missing {k}: {config[k]}')
    if not 1<=args.trials<=50:raise ValueError('trials must be 1..50')
    if not math.isfinite(config['adaptive_threshold']) or config['adaptive_threshold']<0:raise ValueError('threshold must be finite and nonnegative')
    jobs=[]
    for name in dict.fromkeys(args.methods):
        for suite in dict.fromkeys(args.suites):
            for task in dict.fromkeys(args.task_ids):
                jobs.append(dict(id=f'{name}__{suite}_{task}',method=method_config(name,config),suite=suite,task_id=task,trials=args.trials))
    return dict(config=config,methods=list(dict.fromkeys(args.methods)),suites=list(dict.fromkeys(args.suites)),task_ids=list(dict.fromkeys(args.task_ids)),trials=args.trials,jobs=jobs)


def alive(pid,ticks=None):
    try:
        s=Path(f'/proc/{pid}/stat').read_text().split()
        return s[2]!='Z' and (ticks is None or s[21]==ticks)
    except FileNotFoundError:return False


def run(args,plan):
    out=args.output.resolve();out.mkdir(parents=True,exist_ok=True)
    with (out/'controller.lock').open('w') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        return run_locked(args,plan,out)


def run_locked(args,plan,out):
    for p in out.glob('worker_*_ready.json'):
        w=json.loads(p.read_text())
        if alive(w['pid'],w['start_ticks']):raise RuntimeError('Existing worker still alive; wait before resuming')
    # A previous controller may have died before a worker wrote its ready marker.
    for p in out.glob('worker_*_pid.json'):
        w=json.loads(p.read_text())
        if alive(w['pid'],w['start_ticks']):raise RuntimeError('Existing worker process still alive')
    cfg_path=out/'run_config.json'
    if cfg_path.exists() and json.loads(cfg_path.read_text())!=plan:raise ValueError('Output contains a different experiment; use a new --output')
    write_json(cfg_path,plan)
    for d in ['queue','running','done','failed','tasks','archive']:(out/d).mkdir(exist_ok=True)
    failed=list((out/'failed').glob('*.json'))
    if failed and not args.retry_failed:raise RuntimeError('Inspect failed/; then use --retry-failed to retry explicitly')
    for p in list((out/'running').glob('*.json'))+failed:
        key=p.stem;task=out/'tasks'/key
        if task.exists() and not (task/'result.json').exists():task.rename(out/'archive'/f'{key}_{time.time_ns()}')
        p.unlink()
    for job in plan['jobs']:
        key=job['id']
        if (out/'tasks'/key/'result.json').exists():continue
        write_json(out/'queue'/f'{key}.json',job)
    (out/'STOP').unlink(missing_ok=True)
    if (out/'ERROR.json').exists():(out/'ERROR.json').rename(out/'archive'/f'error_{time.time_ns()}.json')
    if (out/'provenance.json').exists():
        (out/'provenance.json').rename(out/'archive'/f'provenance_{time.time_ns()}.json')
    write_json(out/'provenance.json',dict(time=time.time(),python=sys.executable,source_sha256={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in ROOT.glob('*.py')},external_evaluator_sha256=hashlib.sha256((Path(plan['config']['fastwam_root'])/'experiments/libero/eval_libero_single.py').read_bytes()).hexdigest()))
    workers=[];interrupted=False
    def stop(signum,frame):
        nonlocal interrupted
        interrupted=True;(out/'STOP').touch()
    signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop)
    def progress(stage):
        rows=summarize(out)
        done=sum((out/'tasks'/j['id']/'result.json').exists() for j in plan['jobs'])
        write_json(out/'status.json',dict(stage=stage,completed_tasks=done,total_tasks=len(plan['jobs']),completed_episodes=len(list((out/'tasks').glob('*/episode_*.json'))),total_episodes=sum(j['trials'] for j in plan['jobs']),workers_started=len(workers),time=time.time()))
        return done
    try:
        if progress('starting')==len(plan['jobs']):
            progress('complete');write_json(out/'completion.json',dict(complete=True,tasks=len(plan['jobs']),episodes=sum(j['trials'] for j in plan['jobs']),time=time.time()));return
        env=dict(os.environ,CUDA_VISIBLE_DEVICES=args.gpu,OMP_NUM_THREADS='2',MKL_NUM_THREADS='2')
        for i in range(args.workers):
            if all((out/'tasks'/j['id']/'result.json').exists() for j in plan['jobs']):break
            if interrupted or (out/'STOP').exists():break
            ready=out/f'worker_{i}_ready.json';ready.unlink(missing_ok=True)
            with (out/f'worker_{i}.log').open('a') as f:
                w=subprocess.Popen([sys.executable,'-u',str(ROOT/'worker.py'),'--campaign',str(out),'--worker',str(i)],env=env,cwd=ROOT,stdout=f,stderr=subprocess.STDOUT)
            workers.append(w)
            write_json(out/f'worker_{i}_pid.json',dict(pid=w.pid,start_ticks=Path(f'/proc/{w.pid}/stat').read_text().split()[21]))
            deadline=time.time()+1200
            while not ready.exists():
                if interrupted or (out/'STOP').exists():break
                if any(p.poll() is not None for p in workers):raise RuntimeError('Worker exited during startup; inspect logs')
                if time.time()>deadline:raise RuntimeError('Worker initialization timed out')
                progress('loading_workers');time.sleep(10)
        while not interrupted and not (out/'STOP').exists():
            done=progress('evaluating')
            if list((out/'failed').glob('*.json')):raise RuntimeError('Evaluation task failed; inspect task error.json')
            if done==len(plan['jobs']):break
            if any(w.poll() is not None for w in workers):raise RuntimeError('Worker exited before queue completion')
            time.sleep(15)
        (out/'STOP').touch()
        # Graceful stop waits for currently claimed tasks; workers retain environment history.
        while any(w.poll() is None for w in workers):
            progress('stopping_workers');time.sleep(5)
        done=progress('stopped')
        if done==len(plan['jobs']):
            progress('complete')
            write_json(out/'completion.json',dict(complete=True,tasks=done,episodes=sum(j['trials'] for j in plan['jobs']),time=time.time()))
    except Exception:
        write_json(out/'ERROR.json',dict(error=traceback.format_exc(),time=time.time()));raise
    finally:
        (out/'STOP').touch()
        while any(w.poll() is None for w in workers):
            time.sleep(2)
        summarize(out)


def main():
    args=parse_args();plan=make_plan(args)
    if args.dry_run:
        print(json.dumps(dict(methods=plan['methods'],tasks=len(plan['jobs']),episodes=sum(j['trials'] for j in plan['jobs']),workers=args.workers,config=plan['config']),indent=2));return
    if args.detach:
        out=args.output.resolve();out.mkdir(parents=True,exist_ok=True)
        argv=[x for x in sys.argv[1:] if x!='--detach']
        with (out/'controller.log').open('a') as f:
            p=subprocess.Popen([sys.executable,'-u',str(ROOT/'evaluate.py'),*argv],stdin=subprocess.DEVNULL,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
        (out/'controller_pid.txt').write_text(str(p.pid)+'\n');print(f'Controller PID {p.pid}; output {out}');return
    run(args,plan)
if __name__=='__main__':main()
