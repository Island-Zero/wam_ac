import importlib.util,json,unittest,os
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import numpy as np
import torch
from metrics import action_readouts
from router import Router
from runtime import method_config,load_config
ROOT=Path(__file__).resolve().parents[1]
UP=Path(load_config(Path(os.environ.get('LIBERO_EVAL_CONFIG',str(ROOT/'configs/local.json'))))['fastwam_root'])
spec=importlib.util.spec_from_file_location('native_scheduler',UP/'src/fastwam/models/wan22/schedulers/scheduler_continuous.py')
scheduler=importlib.util.module_from_spec(spec);spec.loader.exec_module(scheduler)

class Model:
    def __init__(self):
        self.infer_action_scheduler=scheduler.WanContinuousFlowMatchScheduler()
        self.mot=SimpleNamespace(forward_action_with_video_cache=lambda z,t:.1*z*z+.02*t/1000)
    def _predict_action_noise_with_cache(self,latents_action,timestep_action,**kwargs):
        return self.mot.forward_action_with_video_cache(latents_action,timestep_action)
    def infer_action(self,*,num_inference_steps,seed,sigma_shift=None):
        z=torch.randn((1,32,7),generator=torch.Generator().manual_seed(seed))*.1
        ts,ds=self.infer_action_scheduler.build_inference_schedule(num_inference_steps,z.device,z.dtype,shift_override=sigma_shift)
        for t,d in zip(ts,ds):z=self.infer_action_scheduler.step(self._predict_action_noise_with_cache(latents_action=z,timestep_action=t.reshape(1)),d,z)
        return dict(action=z[0].float())

class EvaluationTests(unittest.TestCase):
    @patch('torch.cuda.synchronize',lambda:None)
    def test_native_baselines_and_adaptive_branches(self):
        for shift in [3.,5.]:
            m=Model();kw=dict(seed=42,sigma_shift=shift,num_inference_steps=10)
            expected={k:m.infer_action(**dict(kw,num_inference_steps=k))['action'] for k in [2,10]}
            router=Router(m)
            for name in ['2-step','10-step']:
                method=method_config(name,{})
                router.configure(method)
                self.assertTrue(torch.equal(m.infer_action(**kw)['action'],expected[method['nfe']]))
                self.assertEqual(router.rows[0]['total_counted_forwards'],method['nfe'])
            for threshold,k,nfe in [(100,2,2),(-1,10,11)]:
                router.configure(dict(mode='adaptive',metric='relative_l2',threshold=threshold))
                self.assertTrue(torch.equal(m.infer_action(**kw)['action'],expected[k]))
                self.assertEqual(router.rows[0]['actual_nfe'],nfe)
                self.assertEqual(router.rows[0]['total_counted_forwards'],nfe)
    @patch('torch.cuda.synchronize',lambda:None)
    def test_sampler_audit(self):
        m=Model();router=Router(m);router.configure(dict(mode='shadow'))
        m.infer_action(seed=42,num_inference_steps=10,sigma_shift=5.)
        self.assertEqual(router.rows[0]['sampler_audit']['a10_native_max_abs'],0)
        self.assertEqual(router.rows[0]['total_counted_forwards'],33)
    def test_relative_l2_magnitude_and_horizon(self):
        a=np.ones((32,7));b=2*a;r=action_readouts(a,b)
        self.assertAlmostEqual(r['relative_l2'],1/np.sqrt(2.5))
        self.assertAlmostEqual(r['cosine'],0)
        b=a.copy();b[10:]=8
        self.assertEqual(action_readouts(a,b,10)['relative_l2'],0)
        self.assertGreater(action_readouts(a,b,32)['relative_l2'],0)
        self.assertEqual(action_readouts(a*0,a*0)['relative_l2'],0)
        with self.assertRaises(ValueError):action_readouts(a,np.full_like(a,np.nan))
if __name__=='__main__':unittest.main()
