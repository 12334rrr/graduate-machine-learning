from pathlib import Path
import sys,torch,numpy as np,json,ast,tempfile,types
R=Path.cwd();sys.path.insert(0,str(R/'hw2_sam/src'))
import e2_finetune as e2
from sam_utils import build_sam
from evaluate_sam import load_stage_weights
from common import CKPT_DIR
results={}
# The cached embedding and asymmetric target must retain the same orientation, even with training enabled.
m=torch.zeros(256,256,dtype=torch.uint8);m[15:45,60:110]=1
emb=torch.arange(256*64*64).reshape(256,64,64).half()
a,b,g=e2.make_batch(([emb],[m],['synthetic']),np.random.default_rng(42),True)
assert torch.equal(a[0],emb.float()) and torch.equal(g[0,0],m.float())
assert 'flip_emb_and_box' not in (R/'hw2_sam/src/e2_finetune.py').read_text()
results['E2_cache_target_orientation']='passed'
# Load the actual saved E3 checkpoint and compare every trained prompt tensor.
sam=build_sam(torch.device('cpu'));sam=load_stage_weights(sam,'e3','e3_lora',torch.device('cpu'))
ck=torch.load(CKPT_DIR/'e3_lora_best.pth',map_location='cpu',weights_only=False)
assert all(torch.equal(v,sam.prompt_encoder.state_dict()[k]) for k,v in ck['prompt_encoder'].items())
results['E3_actual_checkpoint_prompt_all_tensors']='passed'
# Exercise actual persistence block with improve and non-improve epochs, reading saved metadata.
for rel,marker,score,var in [('hw1_unet/src/train.py','        improved = vsum','vsum','best_dice'),('hw2_sam/src/e2_finetune.py','        improved = vs[','vs','best'),('hw2_sam/src/e3_lora.py','        improved = vs[','vs','best')]:
 s=(R/rel).read_text();start=s.index(marker);end=s.index('        if '+('bad_epochs' if 'hw1' in rel else 'row["cum_min"]'),start)
 block=s[start:end];import textwrap
 saved=[]
 class Obj:
  def state_dict(self):return {}
 fake=types.SimpleNamespace(save=lambda state,path:saved.append((str(path),dict(state))))
 scope=dict(torch=fake,vsum={'dice':.8},vs={'dice':.8},best_dice=.7,best=.7,bad_epochs=2,epoch=1,model=Obj(),optimizer=Obj(),scheduler=Obj(),history=[],args=types.SimpleNamespace(lora_r=8,lora_alpha=16),ckpt_last='last',ckpt_best='best',ck_last='last',ck_best='best',sam=types.SimpleNamespace(mask_decoder=Obj(),prompt_encoder=Obj()),lora_state_dict=lambda sam:{})
 exec(textwrap.dedent(block),scope)
 assert saved[0][1]['best_dice']==.8
 if 'hw1' in rel:assert saved[0][1]['bad_epochs']==0
 saved.clear();scope[score]={'dice':.6};exec(textwrap.dedent(block),scope)
 assert len(saved)==1 and saved[0][1]['best_dice']==.8
 if 'hw1' in rel:assert saved[0][1]['bad_epochs']==1
 results[rel+'_persistence']='passed'
for rel in ['hw1_unet/src/train.py','hw2_sam/src/e2_finetune.py','hw2_sam/src/e3_lora.py','hw2_sam/src/evaluate_sam.py','hw2_sam/src/run_all.py']:ast.parse((R/rel).read_text())
results['five_source_syntax']='passed'
# Test completion via actual train_loop with mocked subprocess and fixture CSV.
import run_all
with tempfile.TemporaryDirectory() as td:
 run_all.LOG_DIR=Path(td);run_all.CKPT_DIR=Path(td);run_all.subprocess.run=lambda *a,**k:types.SimpleNamespace(returncode=0)
 for n,script in [(6,'e3_lora.py'),(20,'e2_finetune.py')]:
  (Path(td)/'fixed.csv').write_text('epoch\n'+str(n)+'\n')
  assert run_all.train_loop(script,'fixed',['--epochs',str(n)],max_segments=1)
 (Path(td)/'fixed.csv').write_text('epoch\n17\n')
 try:run_all.train_loop('e2_finetune.py','fixed',['--epochs','20'],max_segments=1)
 except RuntimeError:pass
 else:raise AssertionError('incomplete training accepted')
results['stage_completion_6_20_and_incomplete17']='passed'
p=R.parent/'验证记录/修复回归检查.json';p.write_text(json.dumps(results,ensure_ascii=False,indent=2));print(results)
