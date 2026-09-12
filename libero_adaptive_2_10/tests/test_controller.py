import json,tempfile,unittest,os
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
from evaluate import make_plan,run
from runtime import load_config
ROOT=Path(__file__).resolve().parents[1]

class ControllerTests(unittest.TestCase):
    def args(self,out):
        return SimpleNamespace(config=Path(os.environ.get('LIBERO_EVAL_CONFIG',str(ROOT/'configs/local.json'))),methods=['2-step'],suites=['libero_spatial'],task_ids=[0],trials=1,output=out,retry_failed=False,workers=3,gpu='0')
    def test_config_paths_relative_to_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/'settings.json'
            keys=['fastwam_root','libero_root','checkpoint','dataset_stats','libero_config_path','model_base_path']
            p.write_text(json.dumps({k:'resources/'+k for k in keys}))
            config=load_config(p)
            for k in keys:self.assertEqual(config[k],str(Path(tmp)/'resources'/k))

    def test_nonfinite_threshold_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            a=self.args(Path(tmp));c=json.loads(a.config.read_text());c['adaptive_threshold']=float('nan')
            a.config=Path(tmp)/'config.json';a.config.write_text(json.dumps(c))
            with self.assertRaises(ValueError):make_plan(a)
    def test_completed_resume_preserves_counts_without_loading(self):
        with tempfile.TemporaryDirectory() as tmp:
            out=Path(tmp);a=self.args(out);plan=make_plan(a);job=plan['jobs'][0]
            p=out/'tasks'/job['id'];p.mkdir(parents=True)
            (p/'result.json').write_text(json.dumps(dict(job,total_episodes=1,successes=1,chunks=7,forwards=14)))
            with patch('evaluate.subprocess.Popen',side_effect=AssertionError('must not load model')):
                run(a,plan)
            r=json.loads((out/'completion.json').read_text())
            self.assertEqual(r['tasks'],1);self.assertEqual(r['episodes'],1)
if __name__=='__main__':unittest.main()
