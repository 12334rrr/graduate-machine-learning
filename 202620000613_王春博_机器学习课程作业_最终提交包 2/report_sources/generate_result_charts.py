from pathlib import Path
import json,csv
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
R=Path(__file__).resolve().parents[1]/'repository';D=R.parent/'figures/analysis';D.mkdir(exist_ok=True)
plt.rcParams.update({'font.sans-serif':['Arial Unicode MS','Songti SC','Heiti TC'],'axes.unicode_minus':False,'font.size':12,'axes.spines.top':False,'axes.spines.right':False,'axes.edgecolor':'#A5B3C2','axes.labelcolor':'#21364B','text.color':'#21364B','grid.color':'#E2EAF0','figure.facecolor':'white','axes.facecolor':'white','svg.fonttype':'none'})
C=['#2878B5','#21A08B','#E39D3E']; sources={}
def load(p):return json.loads((R/p).read_text())
def hw1(tag,split='test'):return load(f'hw1_unet/outputs/logs/eval_{tag}_{split}.json')['metrics']
def hw2(st,p):
 f=next((R/'hw2_sam/outputs/logs').glob(f'eval_{st}_{p}_test*.json'));return json.loads(f.read_text())['metrics']
def save(fig,n,src):
 fig.tight_layout(pad=1.5);fig.savefig(D/f'{n}.png',dpi=300,bbox_inches='tight');fig.savefig(D/f'{n}.svg',bbox_inches='tight');plt.close(fig);sources[n]=src
models=['unet','resunet','attn_unet'];names=['U-Net','ResUNet','Attention U-Net']
fig,axs=plt.subplots(1,2,figsize=(10.5,4.5));x=np.arange(3)
for k,metric in enumerate(['dice','iou']):
 vals=np.array([[hw1(f'{m}_s{s}')[metric] for s in [42,2026,7]] for m in models]);b=axs[0].bar(x+(k-.5)*.32,vals.mean(1),.32,yerr=vals.std(1,ddof=1),capsize=4,color=C[k],label=metric.upper())
 axs[0].bar_label(b,fmt='%.3f',padding=4,fontsize=10)
vals=np.array([[hw1(f'{m}_s{s}')['hd95'] for s in [42,2026,7]] for m in models]);b=axs[1].bar(x,vals.mean(1),.55,yerr=vals.std(1,ddof=1),capsize=4,color=C[2]);axs[1].bar_label(b,fmt='%.1f',padding=4)
for a in axs:a.set_xticks(x,names,rotation=12);a.grid(axis='y',zorder=0);a.set_axisbelow(True)
axs[0].set_ylim(0,1);axs[0].legend(loc='upper left',frameon=False);axs[0].set_title('区域重合  三种子均值与样本标准差');axs[1].set_ylabel('HD95 / 256网格像素');axs[1].set_title('边界距离  空预测不进入HD95均值');axs[1].set_ylim(0,125)
save(fig,'model_comparison','9份正式test评测JSON；ddof=1')
fig,axs=plt.subplots(1,3,figsize=(10.7,4.2))
groups=[(['loss_bce','loss_dice','unet_s42'],['BCE','Dice','混合'],'损失函数'),(['size192','unet_s42','size384'],['192','256','384'],'输入分辨率'),(['attn_skip','attn_bottleneck','attn_unet_s42'],['g1/g2','g4','g1–g4'],'门控位置')]
for a,(tags,labs,title) in zip(axs,groups):
 b=a.bar(range(3),[hw1(t)['dice'] for t in tags],color=[C[0],C[2],'#98A6B5'] if title=='损失函数' else [C[0],'#98A6B5',C[2]] if title=='输入分辨率' else [C[0],C[2],'#98A6B5'],width=.6);a.bar_label(b,fmt='%.4f',padding=4,fontsize=10);a.set_xticks(range(3),labs);a.set_ylim(0,1);a.set_title(title);a.set_ylabel('Dice');a.grid(axis='y');a.set_axisbelow(True)
save(fig,'ablation_dice','对应9个设置的test JSON；单种子42，默认组复用')
fig,ax=plt.subplots(figsize=(9.7,4.1))
for j,m in enumerate(['resunet','attn_unet']):
 v=[hw1(f'cv_{m}_f{k}',f'fold{k}')['dice'] for k in range(5)];ax.plot(range(5),v,'o-',color=C[j],label=names[models.index(m)]+f'  {np.mean(v):.4f} ± {np.std(v,ddof=1):.4f}')
ax.set_xticks(range(5),[f'折{k}' for k in range(5)]);ax.set_ylim(0,1);ax.set_ylabel('留出折 Dice');ax.legend(frameon=False,loc='lower left');ax.grid();save(fig,'cv_dice','10份fold评测JSON，非独立嵌套CV')
fig,axs=plt.subplots(1,3,figsize=(10.7,4.1))
for a,m,name in zip(axs,models,names):
 for j,s in enumerate([42,2026,7]):
  h=load(f'hw1_unet/outputs/logs/{m}_s{s}_summary.json')['history'];a.plot([z['epoch'] for z in h],[z['dice'] for z in h],'o-',ms=3,lw=1.4,color=C[j],ls=['-','--','-.'][j],label=f'seed {s}')
 a.set_title(name);a.set_ylim(0,1);a.set_xlabel('实际轮次');a.set_ylabel('验证 Dice');a.grid();a.legend(fontsize=9,frameon=False)
save(fig,'all_seed_curves','9份summary.history；原始点直线相连，不进行平滑')
fig,axs=plt.subplots(1,2,figsize=(10.4,4.1))
for j,m in enumerate(models):
 h=load(f'hw1_unet/outputs/logs/{m}_s42_summary.json')['history'];ep=[z['epoch'] for z in h]
 axs[0].plot(ep,[z['lr'] for z in h],'o-',color=C[j],label=names[j],ms=3);axs[1].plot(ep,[z['epoch_sec'] for z in h],'o-',color=C[j],label=names[j],ms=3)
for a in axs:a.set_xlabel('实际轮次');a.grid();a.legend(fontsize=9,frameon=False)
axs[0].set_ylabel('记录学习率');axs[0].ticklabel_format(style='sci',axis='y',scilimits=(0,0));axs[1].set_ylabel('单轮耗时 / 秒');save(fig,'lr_epoch_cost','种子42 summary.history；不使用重置的cum_min求训练总时长')
prompts=['gt','pert-20','pert-10','pert10','pert20','point1','point3','point5','auto','auto10'];labels=['GT框','内缩20%','内缩10%','外扩10%','外扩20%','1点','3点','5点','自动框','自动框+10%'];stages=['e1','e2','e3']
fig,ax=plt.subplots(figsize=(10.8,3.8));v=np.array([[hw2(s,p)['dice'] for p in prompts] for s in stages]);im=ax.imshow(v,cmap='GnBu',vmin=0,vmax=1,aspect='auto');ax.set_xticks(range(10),labels,rotation=25,ha='right');ax.set_yticks(range(3),[s.upper() for s in stages]);ax.spines[['left','bottom']].set_visible(False)
for y in range(3):
 for x in range(10):ax.text(x,y,f'{v[y,x]:.4f}',ha='center',va='center',color='white' if v[y,x]>.65 else '#21364B',fontsize=11)
fig.colorbar(im,ax=ax,pad=.02,label='测试 Dice');save(fig,'prompt_matrix','30份test JSON；各n=1000；E2/E3为2026-10-29完整修复后复评')
fig,axs=plt.subplots(3,1,figsize=(10.5,7.4),sharex=True)
for k,(a,s) in enumerate(zip(axs,stages)):
 x=np.arange(10);a.bar(x-.18,[hw2(s,p)['sensitivity'] for p in prompts],.36,color=C[0],label='Sensitivity');a.bar(x+.18,[hw2(s,p)['specificity'] for p in prompts],.36,color=C[1],label='Specificity');a.set_ylim(0,1.08);a.set_ylabel(s.upper());a.grid(axis='y');a.set_axisbelow(True)
axs[0].legend(ncol=2,loc='lower left',frameon=False);axs[-1].set_xticks(range(10),labels,rotation=20,ha='right');save(fig,'sens_spec','30份test JSON；逐图平均，sens_valid=1000，spec_valid=998')
fig,axs=plt.subplots(3,1,figsize=(10.5,7.4),sharex=True)
for a,s,col in zip(axs,stages,C):
 b=a.bar(range(10),[hw2(s,p)['latency_ms_per_img'] for p in prompts],color=col,width=.65);a.bar_label(b,fmt='%.1f',padding=2,fontsize=8.5);a.set_ylabel(s.upper()+' / ms');a.grid(axis='y');a.set_axisbelow(True);a.margins(y=.25)
axs[-1].set_xticks(range(10),labels,rotation=20,ha='right');save(fig,'latency','30份test JSON；各行独立纵轴，E1/E2含缓存读取，E3在线编码，非端到端同路径基准')
fig,ax=plt.subplots(figsize=(9.8,4.7));x=np.arange(3)
for j,p in enumerate(['gt','auto','auto10']):
 b=ax.bar(x+(j-1)*.24,[hw2(s,p)['dice'] for s in stages],.24,color=C[j],label={'gt':'GT框（模拟交互）','auto':'U-Net自动框','auto10':'自动框+10%'}[p]);ax.bar_label(b,fmt='%.4f',padding=3,fontsize=10)
ax.axhline(hw1('unet_s42')['dice'],color='#687889',ls='--',label='上游 unet_s42 = 0.6062');ax.set_xticks(x,['E1','E2','E3']);ax.set_ylim(0,1.12);ax.set_ylabel('测试 Dice');ax.legend(ncol=2,fontsize=10,frameon=False,loc='upper right');ax.grid(axis='y');ax.set_axisbelow(True);save(fig,'cascade_results','9份SAM JSON与unet_s42 JSON；三个阶段均使用相同上游unet_s42框')
(D/'sources.json').write_text(json.dumps(sources,ensure_ascii=False,indent=2));print('new charts',len(sources))
