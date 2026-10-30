from pathlib import Path
import json,csv,re,numpy as np,shutil
R=Path.cwd();F=R/'submission_refined';RUN=R/'full_reproduction_20261029';s=(R/'.refine_build/previous_delivery/机器学习课程作业_精简报告.md').read_text()
audit=json.loads((RUN/'checkpoint_audit.json').read_text());assert json.loads((RUN/'pipeline_status.json').read_text())['status']=='experiments_complete'
def get(st,p):
 hits=list((R/'hw2_sam/outputs/logs').glob(f'eval_{st}_{p}_test*.json'));assert len(hits)==1
 return json.loads(hits[0].read_text())['metrics']
def repl(a,b,new):
 global s
 start=s.index(a);end=s.index(b,start);s=s[:start]+new+'\n\n'+s[end:]
def tab(num,rows):
 global s
 pattern=rf'(^表{num}　[^\n]+\n\n)(?:\|[^\n]*\n)+'
 match=re.search(pattern,s,re.M);assert match,num
 oldrows=match.group(0).splitlines();head='\n'.join(oldrows[:4])+'\n'
 s=s[:match.start()]+head+'\n'.join('| '+' | '.join(map(str,row))+' |' for row in rows)+'\n'+s[match.end():]
# Concise main text gives actual final outcomes; detailed matrices remain in the appendix.
s=s.replace('SAM历史结果显示提示来源明显影响性能，GT框成绩不能直接代表自动分割能力','修复后E2在GT框下Dice为0.9181、自动框下为0.6918，提示来源明显影响性能').replace('历史评测与修复后的验证分别归档','问题实验已重新训练并完成全量复评，原结果单独归档')
repl('### 4.4 实现修复与证据边界','## 五 实验结果分析','''### 4.4 实现修复与完整复验

针对原4.8节的问题，E2取消缓存特征翻转，保持嵌入、框与GT同坐标；E3完整恢复LoRA、提示编码器、解码器及缩放参数；训练器先更新best与耐心计数再保存last，编排按实际目标轮数验收。E2从官方权重完成20轮，E3按600张训练、100张验证的既定配置完成6轮，两者各10种提示均完成1000张测试评测。384输入实验也重新训练并全量评测；其余24组卷积实验与E1核对后保留，cv_resunet_f3只修正滞后的汇总字段。

新E2、E3的best分别位于第9、2轮，384输入best位于第2轮，均与日志峰值及last中的best记录一致。原缺失权重以新训练解决，原结果另存历史目录；正式结果记录样本ID、命令、源码与权重哈希。E3多提示评测逐图复用当前模型的在线编码，每类提示独立维护随机序列，并核对重新编码后的预测逐像素一致；未使用原版缓存替代适配后编码。''')
s=s.replace('192输入优于384，但二者训练成本与优化轨迹不同，且HD95网格不同','384输入修复后Dice由0.4756变为0.5810，仍低于192输入的0.6391；两者训练成本、优化轨迹与HD95网格不同')
repl('图12将三阶段和十类提示','![](figures/analysis/prompt_matrix.png)','''图12将三阶段和十类提示置于同一矩阵。E2的GT框Dice为0.9181，高于E1的0.8691及E3的0.8970；提示内缩会遗漏边缘，改用点提示也改变了可用的定位信息。本轮E2的点数增加改善重合，是否普遍成立仍需多种子验证。E3只用600张训练图像，而E2使用2594张，因此阶段差异不能单独归因于LoRA机制。''')
s=s.replace('三阶段十种提示的历史Dice矩阵','三阶段十种提示的最终Dice矩阵').replace('每格n=1000；E2/E3为修复前历史记录，详见4.4节。','每格n=1000；E2/E3为本次修复后完整复评，E1核对后保留。')
e3auto=get('e3','auto')['dice'];e3auto10=get('e3','auto10')['dice'];e2auto=get('e2','auto')['dice'];e2auto10=get('e2','auto10')['dice']
repl('图13显示GT框到自动框','![](figures/analysis/cascade_results.png)',f'''图13显示GT框到自动框的性能落差。固定上游unet_s42的Dice为0.6062，E2与E3自动框级联分别为{e2auto:.4f}和{e3auto:.4f}；自动化能力应在这一路径下讨论。外扩10%后E2为{e2auto10:.4f}、E3为{e3auto10:.4f}，不能预设外扩必然改善结果。GT框及前景点由真实标签生成，只代表模拟交互条件。E1/E2读取原版缓存，E3在线编码，记录时延不作为统一部署速度比较。''')
s=s.replace('GT框与自动框条件的历史比较','GT框与自动框条件的最终比较')
repl('图15对照SAM','![](figures/hw2/cases_verified.png)','''图15使用新选定权重重新生成同一组三个样本的GT框预测，直观对照不同适配方式的边缘覆盖。图中每个Dice直接由展示的二值掩膜计算，并与正式全测试JSON中相同ID的记录核对。该案例补充总体均值，不能代表全体样本的误差分布。''')
s=s.replace('同样本与GT框下的SAM历史预测','同样本与GT框下的SAM最终预测').replace('图像ID与U-Net一致；E3历史加载限制见4.4节。','图像ID与U-Net一致；E2/E3使用本次新训练的best，完整恢复各训练模块。')
s=s.replace('代码修复之后仍需独立重训和完整评测，历史缺失权重不能通过改写报告弥补。','本次以独立重训和完整评测解决权重缺失，并核对选中轮次与日志峰值；结论随新证据更新。')
s=s.replace('GT 框下三阶段历史测试结果','GT 框下三阶段最终测试结果').replace('三阶段十类提示的历史 Dice','三阶段十类提示的最终 Dice').replace('自动框级联与上游模型的历史结果','自动框级联与上游模型的最终结果')
# Only replace the affected input-resolution row; unaffected runs keep their recorded results.
m=json.loads((R/'hw1_unet/outputs/logs/eval_size384_test.json').read_text())['metrics'];z=audit['size384']
s=re.sub(r'^\| 384 输入 \|[^\n]+',f"| 384 输入 | {m['dice']:.4f} | {m['iou']:.4f} | {m['hd95']:.2f} | {m['empty_pred']} / {m['hd95_valid']} | {z['actual_epochs']} | {z['train_minutes']:.1f} |",s,flags=re.M)
s=re.sub(r'^\| size384 \|[^\n]+',f"| size384 | 42 | 384 | bce_dice | — | {z['actual_epochs']} | 早停6 | {z['best_epoch']} |",s,flags=re.M)
tab(7,[[st.upper(),f"{(m:=get(st,'gt'))['dice']:.4f}",f"{m['iou']:.4f}",f"{m['hd95']:.2f}",f"{m['sensitivity']:.4f}",f"{m['specificity']:.4f}",{'e1':'0','e2':'4.06','e3':'4.36'}[st]] for st in ['e1','e2','e3']])
tab(8,[[st.upper(),(m:=get(st,'gt'))['empty_pred'],m['hd95_valid'],m['sens_valid'],m['spec_valid'],f"{m['latency_ms_per_img']:.2f}"] for st in ['e1','e2','e3']])
rows=[]
for tag in ['e2_box','e3_lora']:
 z=audit[tag];hist=list(csv.DictReader((R/f'hw2_sam/outputs/logs/{tag}.csv').open()));rows.append([tag,z['actual_epochs'],z['best_epoch'],f"{z['val_best']:.4f}",f"{z['train_minutes']:.2f}",f"{z['mean_epoch_sec']:.1f}",f"{np.median([float(r['epoch_sec']) for r in hist]):.1f}"])
tab(9,rows)
prompts=['gt','pert-10','pert-20','pert10','pert20','point1','point3','point5','auto','auto10']
tab(10,[[p]+[f"{get(st,p)['dice']:.4f}" for st in ['e1','e2','e3']] for p in prompts])
m=json.loads((R/'hw1_unet/outputs/logs/eval_unet_s42_test.json').read_text())['metrics'];rows=[['unet_s42','无提示',f"{m['dice']:.4f}",f"{m['iou']:.4f}",f"{m['hd95']:.2f}",m['empty_pred'],m['hd95_valid']]]
for st in ['e1','e2','e3']:
 for p in ['auto','auto10']:
  m=get(st,p);rows.append([st.upper(),p,f"{m['dice']:.4f}",f"{m['iou']:.4f}",f"{m['hd95']:.2f}",m['empty_pred'],m['hd95_valid']])
tab(11,rows)
tab(13,[[st,p,f"{(m:=get(st,p))['iou']:.4f}",f"{m['hd95']:.2f}",m['empty_pred'],m['hd95_valid'],m['sens_valid'],m['spec_valid']] for st in ['e1','e2','e3'] for p in prompts])
repl('修复后的源文件随repository交付','### B.2 获取代码并查看结果','''已完成E2 20轮、E3 6轮与384输入问题实验的独立训练，随后完成E2/E3各10类提示与384输入的1000张测试评测。训练best轮次、日志峰值和last元数据一致；其余卷积权重核对通过，未扩大重跑范围。回归检查及逐像素编码复用核验通过。完整命令、日志、逐样本结果和哈希见验证记录/完整复验及repository/RESULTS_RELEASE.json；历史文件单独归档。''')
s=s.replace('重算历史汇总并打开门户','重算最终汇总并打开门户')
s=s.replace('完整实验应在独立副本使用新tag，E2重新从官方权重训练，E3完整恢复prompt后重评；不要续接受旧监督影响的E2状态。新评测使用--output-dir隔离输出。数据、嵌入和大权重不随提交包分发。','本次正式命令见repository/QUICKSTART.md。需要重训时使用独立副本和新输出目录，完整训练采用连续运行避免分段续跑改变随机轨迹。提交包提供E2/E3及384输入的推理权重，原始数据、官方SAM基础权重和嵌入需按说明准备。')
s=s.replace('历史页面中的简化参数描述以正文和原始记录为准。','截图重新采集于正式结果更新后的页面；配色、提示矩阵和三级联比较与正文一致。')
s=s.replace('原始JSON/CSV位于repository各实验outputs/logs。','正式JSON/CSV位于repository各实验outputs/logs；E2/E3及384输入已替换为本次复验，其他结果保留。')
(F/'机器学习课程作业_精简报告.md').write_text(s)
print('Updated report from verified final metrics',len(s))
