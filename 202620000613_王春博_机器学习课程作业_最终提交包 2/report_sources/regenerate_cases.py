from pathlib import Path
import json,sys,hashlib
import numpy as np,torch
import matplotlib;matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
R=Path.cwd();F=R/'submission_refined';sys.path.insert(0,str(R/'hw2_sam/src'))
from sam_dataset import ISICRaw
from sam_utils import build_sam,gt_box_from_mask,to_1024_box,encode_and_decode,resize_logits_to_eval
from evaluate_sam import load_stage_weights,get_embedding
from common import get_device,autocast_ctx,supports_amp
from metrics import dice_score
old=json.loads((F/'案例复核.json').read_text());ids=old['ids'];ds=ISICRaw('test',256);samples=[ds[ds.ids.index(k)] for k in ids];device=get_device();results={}
plt.rcParams.update({'font.sans-serif':['Arial Unicode MS','Heiti TC'],'axes.unicode_minus':False})
for st,tag in [('e1','e1'),('e2','e2_box'),('e3','e3_lora')]:
 sam=load_stage_weights(build_sam(device),st,tag,device);sam.eval();results[st]={}
 ref=json.loads(next((R/'hw2_sam/outputs/logs').glob(f'eval_{st}_gt_test*.json')).read_text());scores={z['id']:z['dice'] for z in ref.get('per_sample',[])}
 with torch.no_grad():
  for sid,img,gt in samples:
   emb=get_embedding(sam,st,'test',sid,img,device,st!='e3')
   with autocast_ctx(device,enabled=supports_amp(device)):log,_=encode_and_decode(sam,emb,box_1024=to_1024_box(gt_box_from_mask(gt)))
   pd=(torch.sigmoid(resize_logits_to_eval(log).float())>.5).cpu().numpy()[0,0].astype(np.uint8);d=float(dice_score(pd,gt));results[st][sid]={'mask':pd,'dice':d}
   if scores:assert abs(d-scores[sid])<1e-12,(st,sid,d,scores[sid])
 del sam
 if device.type=='mps':torch.mps.empty_cache()
fig,axs=plt.subplots(3,5,figsize=(9,6.8));colors=['#2878B5','#21A08B','#E39D3E']
for row,(sid,img,gt) in enumerate(samples):
 axs[row,0].imshow(img);b=gt_box_from_mask(gt);axs[row,0].add_patch(Rectangle((b[0],b[1]),b[2]-b[0],b[3]-b[1],fill=False,color='#21A08B',lw=1));axs[row,0].set_title(sid+'\n原图与GT框',fontsize=11)
 axs[row,1].imshow(gt,cmap='gray',vmin=0,vmax=1);axs[row,1].set_title('真实标签',fontsize=12)
 for col,st in enumerate(['e1','e2','e3'],2):
  z=results[st][sid];axs[row,col].imshow(z['mask'],cmap='gray',vmin=0,vmax=1);axs[row,col].set_title(f"{st.upper()} Dice={z['dice']:.4f}",fontsize=12,color=colors[col-2])
for ax in axs.flat:ax.axis('off')
fig.tight_layout(pad=.7);fig.savefig(F/'figures/hw2/cases_verified.png',dpi=300,bbox_inches='tight');plt.close(fig)
sh=__import__('shutil');sh.copy2(F/'figures/hw2/cases_verified.png',R/'hw2_sam/outputs/figures/case_sam_gt_test.png')
old.update(sam_loader='修复后完整恢复LoRA、prompt encoder和decoder；E2/E3为2026-10-29新训练best',sam={st:{sid:z['dice'] for sid,z in rec.items()} for st,rec in results.items()},verified_against_full_test_json=True)
old['sam_checkpoint_sha256']={st:hashlib.sha256((R/f'hw2_sam/outputs/checkpoints/{tag}_best.pth').read_bytes()).hexdigest() for st,tag in [('e2','e2_box'),('e3','e3_lora')]}
(F/'案例复核.json').write_text(json.dumps(old,ensure_ascii=False,indent=2));print(json.dumps(old,ensure_ascii=False))
