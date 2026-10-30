"""Evaluate remaining E3 prompts with one exact, on-device encoder result per image.
Each prompt keeps its own seed-42 RNG stream, identical to evaluate_sam.py.
No pretrained/E2 cache or reduced precision embedding serialization is used.
"""
from pathlib import Path
import argparse, hashlib, json, os, sys, time
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[1]
RUN=ROOT/'full_reproduction_20261029'
sys.path.insert(0,str(RUN/'hw2_sam/src'))
from common import EVAL_SIZE, autocast_ctx, get_device, set_seed, supports_amp, write_progress
from evaluate_sam import load_stage_weights, get_embedding, build_prompt
from sam_dataset import ISICRaw
from sam_utils import build_sam, encode_and_decode, to_1024_box, resize_logits_to_eval
from metrics import MetricAccumulator, dice_score
PROMPTS=['pert-20','pert-10','pert10','pert20','point1','point3','point5','auto','auto10']
def atomic(p,d):
 q=p.with_suffix('.tmp');q.write_text(json.dumps(d,ensure_ascii=False,indent=2));q.replace(p)
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
@torch.no_grad()
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--limit',type=int);ap.add_argument('--verify-reuse',action='store_true');args=ap.parse_args()
 set_seed(42);device=get_device();amp=supports_amp(device)
 sam=load_stage_weights(build_sam(device),'e3','e3_lora',device);sam.eval()
 ds=ISICRaw('test',size=EVAL_SIZE);n=min(args.limit or len(ds),len(ds))
 boxes=json.loads((RUN/'hw1_unet/outputs/predictions/unet_s42_test_boxes.json').read_text())
 accs={p:MetricAccumulator() for p in PROMPTS};rngs={p:np.random.default_rng(42) for p in PROMPTS}
 samples={p:[] for p in PROMPTS};times={p:[] for p in PROMPTS};fallback={p:0 for p in PROMPTS}
 statuspath=RUN/'pipeline_status.json';state=json.loads(statuspath.read_text());start=time.time()
 full=args.limit is None
 if full:
  state.update(pid=os.getpid(),child_pid=os.getpid(),status='running',current_job='eval_e3_remaining_prompts',updated_at=time.time(),error=None);atomic(statuspath,state)
 checks=[]
 def decode(emb,box,pts):
  with autocast_ctx(device,enabled=amp):
   low,_=encode_and_decode(sam,emb,box_1024=to_1024_box(box) if box is not None else None,points_1024=(pts*4).astype(np.float32) if pts is not None else None)
  logits=resize_logits_to_eval(low,EVAL_SIZE)
  return (torch.sigmoid(logits.float())>0.5).cpu().numpy()[0,0].astype(np.uint8)
 try:
  for i in range(n):
   sid,img,mask=ds[i];t=time.time();emb=get_embedding(sam,'e3','test',sid,img,device,use_cache=False)
   if device.type=='mps':torch.mps.synchronize()
   enc_ms=(time.time()-t)*1000
   # Independently compare with the already completed single-prompt GT evaluation.
   if i<3:
    ref=json.loads((RUN/'hw2_sam/outputs/logs/eval_e3_gt_test.json').read_text())['per_sample'][i]
    dice=dice_score(decode(emb,build_prompt('gt',mask,EVAL_SIZE,np.random.default_rng(42))[0],None),mask)
    assert sid==ref['id'] and abs(dice-ref['dice'])<1e-12,(sid,dice,ref)
   for p in PROMPTS:
    box,pts=build_prompt(p,mask,EVAL_SIZE,rngs[p],boxes,sid)
    if box is None and pts is None:box=np.array([0,0,255,255],dtype=np.float32);fallback[p]+=1
    t=time.time();pred=decode(emb,box,pts);times[p].append(enc_ms+(time.time()-t)*1000)
    if args.verify_reuse and i==0:
     fresh=get_embedding(sam,'e3','test',sid,img,device,use_cache=False);other=decode(fresh,box,pts)
     assert torch.equal(emb,fresh) and np.array_equal(pred,other),p
     checks.append(dict(prompt=p,embedding_bitwise_equal=True,prediction_bitwise_equal=True))
    accs[p].add(pred,mask,compute_hd=True);samples[p].append(dict(id=sid,dice=float(dice_score(pred,mask))))
    if p=='gt' or p=='auto10':pass
    # Retain actual repaired case masks, not historical gallery outputs.
    if i<8 and p in ['point1','auto','auto10']:
     dest=RUN/'hw2_sam/outputs/predictions/repaired_cases';dest.mkdir(exist_ok=True)
     np.save(dest/f'{sid}_{p}.npy',pred)
   if (i+1)%10==0 or i+1==n:
    print(f'E3 prompts {i+1}/{n}; {time.time()-start:.1f}s',flush=True)
    write_progress('e3-suite-test',phase='eval',it=i+1,total=n,stage='e3',detail='E3九种提示共享同图在线编码')
    if full:state.update(updated_at=time.time(),suite_samples=i+1,suite_total=n);atomic(statuspath,state)
  outdir=RUN/'hw2_sam/outputs/logs' if full else ROOT/'.refine_build/suite_verification';outdir.mkdir(exist_ok=True)
  weight=sha(RUN/'hw2_sam/outputs/checkpoints/e3_lora_best.pth')
  for p in PROMPTS:
   metrics=accs[p].summary();metrics.update(latency_ms_per_img=float(np.mean(times[p])),no_prompt_fallback=fallback[p])
   name=f'eval_e3_{p}_test'+('_unet_s42' if p.startswith('auto') else '')
   out=dict(stage='e3',tag='e3_lora',prompt=p,split='test',n=n,metrics=metrics,per_sample=samples[p],seed=42,revision='aligned-cache-full-prompt-v2',checkpoint_sha256=weight,encoder_reuse='exact online E3 embedding per image; independent per-prompt RNG',latency_note='online encoder time + individual prompt decoder time; excludes metrics and other prompts')
   atomic(outdir/(name+'.json'),out)
   if full:
    job='eval_e3_'+p
    if job not in state['completed']:state['completed'].append(job)
    atomic(RUN/'pipeline_logs'/f'{job}_evidence.json',dict(command=sys.argv,completed_at=time.time(),source_sha256=sha(Path(__file__)),best_weights={'e3_lora_best.pth':weight},n=n,seed=42))
  atomic(outdir/'encoder_reuse_verification.json',dict(checks=checks,gt_reference_first3_equal=True,n=n,checkpoint_sha256=weight))
  if full:
   assert len(state['completed'])==24,state['completed']
   state.update(status='experiments_complete',current_job=None,finished_at=time.time(),updated_at=time.time());atomic(statuspath,state)
  print('COMPLETE',n,len(PROMPTS),flush=True)
 except BaseException as ex:
  if full:state.update(status='failed',error=str(ex),updated_at=time.time());atomic(statuspath,state)
  raise
if __name__=='__main__':main()
