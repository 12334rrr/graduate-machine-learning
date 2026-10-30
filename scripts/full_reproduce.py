"""Run the complete repaired experiment matrix sequentially, with immutable input/source snapshots."""
from pathlib import Path
import csv,hashlib,json,os,shutil,subprocess,sys,time,traceback
ROOT=Path(__file__).resolve().parents[1]
RUN=ROOT/'full_reproduction_20261029'
PY1='/opt/anaconda3/envs/hw1-unet-medseg/bin/python'
PY2='/opt/anaconda3/envs/hw2-sam-transformer/bin/python'
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
 return h.hexdigest()
def write(path,obj):
 tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(obj,ensure_ascii=False,indent=2));tmp.replace(path)
def prepare():
 RUN.mkdir(exist_ok=True)
 if (RUN/'source_manifest.json').exists():return
 for proj in ['hw1_unet','hw2_sam']:
  shutil.copytree(ROOT/proj/'src',RUN/proj/'src',ignore=shutil.ignore_patterns('__pycache__'),dirs_exist_ok=True)
  for folder in ['checkpoints','logs','figures','predictions','progress']:(RUN/proj/'outputs'/folder).mkdir(parents=True,exist_ok=True)
  (RUN/proj/'reports').mkdir(exist_ok=True);(RUN/proj/'data').mkdir(exist_ok=True)
 for rel in ['hw1_unet/data/processed','hw2_sam/data/raw','hw2_sam/data/embeddings']:
  dest=RUN/rel
  if not dest.exists():dest.symlink_to(ROOT/rel,target_is_directory=True)
 shutil.copy2(ROOT/'hw1_unet/reports/data_stats.json',RUN/'hw1_unet/reports/data_stats.json')
 files=[p for proj in ['hw1_unet','hw2_sam'] for p in (RUN/proj/'src').glob('**/*.py')]
 write(RUN/'source_manifest.json',{str(p.relative_to(RUN)):sha(p) for p in files})
 write(RUN/'input_manifest.json',{str(p.relative_to(ROOT)):sha(p) for p in [ROOT/'hw2_sam/data/raw/sam_vit_b_01ec64.pth']+list((ROOT/'hw1_unet/data/processed').glob('*_ids.txt'))})
def matrix():
 jobs=[]
 def add(name,proj,python,args):jobs.append(dict(name=name,project=proj,command=[python,'-u']+args))
 add('E2_retrain20','hw2_sam',PY2,['src/e2_finetune.py','--tag','e2_box','--epochs','20','--batch','8','--max-minutes','100000','--num-workers','0'])
 add('E3_retrain6','hw2_sam',PY2,['src/e3_lora.py','--tag','e3_lora','--epochs','6','--batch','1','--accum','8','--train-limit','600','--val-limit','100','--max-minutes','100000'])
 cfgs=[dict(model=m,seed=s,tag=f'{m}_s{s}') for s in [42,2026,7] for m in ['unet','resunet','attn_unet']]
 cfgs += [dict(model='unet',tag=t,**kw) for t,kw in [('loss_bce',dict(loss='bce')),('loss_dice',dict(loss='dice')),('size192',dict(size=192)),('size384',dict(size=384))]]
 cfgs += [dict(model='attn_unet',tag='attn_'+a,attn_where=a) for a in ['skip','bottleneck']]
 cfgs += [dict(model=m,tag=f'cv_{m}_f{k}',fold=k) for m in ['attn_unet','resunet'] for k in range(5)]
 for cfg in cfgs:
  tag=cfg['tag'];train=['src/train.py','--epochs','12','--patience','6','--max-minutes','100000','--num-workers','0']
  for k,v in cfg.items():train += ['--'+k.replace('_','-'),str(v)]
  add('train_'+tag,'hw1_unet',PY1,train)
  ev=['src/evaluate.py','--tag',tag,'--split','test','--size',str(cfg.get('size',256))]
  if 'fold' in cfg:ev+=['--fold',str(cfg['fold'])]
  add('eval_'+tag,'hw1_unet',PY1,ev)
 for stage in ['e1','e2','e3']:
  for prompt in ['gt','pert-20','pert-10','pert10','pert20','point1','point3','point5','auto','auto10']:
   args=['src/evaluate_sam.py','--stage',stage,'--tag',{'e1':'e1','e2':'e2_box','e3':'e3_lora'}[stage],'--prompt',prompt,'--split','test']
   if prompt.startswith('auto'):args+=['--unet-tag','unet_s42']
   add('eval_'+stage+'_'+prompt,'hw2_sam',PY2,args)
 return jobs
def main():
 prepare();jobs=matrix();logdir=RUN/'pipeline_logs';logdir.mkdir(exist_ok=True)
 state_path=RUN/'pipeline_status.json';state=json.loads(state_path.read_text()) if state_path.exists() else dict(started_at=time.strftime('%Y-%m-%d %H:%M:%S'),completed=[])
 write(RUN/'experiment_plan.json',jobs)
 state.update(status='running',pid=os.getpid(),total_jobs=len(jobs),run_root=str(RUN))
 current=None
 try:
  for i,job in enumerate(jobs):
   if job['name'] in state['completed']:continue
   # Failed interrupted training is never silently resumed with an altered RNG trajectory.
   if job['name'].startswith(('train_','E2_','E3_')):
    ck=RUN/job['project']/'outputs/checkpoints'
    tag=job['command'][job['command'].index('--tag')+1]
    if (ck/f'{tag}_last.pth').exists():raise RuntimeError(f'Interrupted run {tag}; review before restart to preserve reproducibility')
   state.update(current_job=job['name'],job_index=i+1,job_started_at=time.time(),updated_at=time.time());write(state_path,state)
   print(time.strftime('%F %T'),job['name'],flush=True)
   with (logdir/(job['name']+'.log')).open('w') as log:
    current=subprocess.Popen(job['command'],cwd=RUN/job['project'],stdout=log,stderr=subprocess.STDOUT)
    while current.poll() is None:
     state.update(child_pid=current.pid,updated_at=time.time());write(state_path,state);time.sleep(15)
    if current.returncode:raise RuntimeError(f'{job["name"]} failed ({current.returncode}); inspect {log.name}')
   # Record the exact weight identity used/created, with the source manifest identity.
   ckdir=RUN/job['project']/'outputs/checkpoints'
   weights={p.name:sha(p) for p in ckdir.glob('*_best.pth')}
   write(logdir/(job['name']+'_evidence.json'),dict(command=job['command'],completed_at=time.time(),source_manifest_sha256=sha(RUN/'source_manifest.json'),best_weights=weights))
   state['completed'].append(job['name']);write(state_path,state)
  state.update(status='experiments_complete',current_job=None,finished_at=time.time());write(state_path,state)
 except BaseException as ex:
  if current and current.poll() is None:current.terminate();current.wait()
  state.update(status='failed',error=str(ex),updated_at=time.time());write(state_path,state);raise
if __name__=='__main__':main()
