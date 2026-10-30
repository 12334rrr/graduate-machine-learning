"""Repair only evidence-confirmed affected runs; retain all unaffected experiments."""
from pathlib import Path
import csv,json,os,subprocess,sys,time
import full_reproduce as runner
RUN=runner.RUN
STATUS=RUN/'pipeline_status.json'
def alive(pid):
 try:os.kill(pid,0);return True
 except ProcessLookupError:return False
def main():
 s=json.loads(STATUS.read_text());pid=s['child_pid'];s.update(pid=os.getpid(),status='running',scope='E2 E3及size384；cv_resunet_f3仅修正汇总元数据，其余沿用')
 # Preserve the already-running corrected E2 training process.
 while alive(pid):
  s.update(current_job='E2_retrain20',child_pid=pid,updated_at=time.time(),total_jobs=24);runner.write(STATUS,s);time.sleep(15)
 rows=list(csv.DictReader((RUN/'hw2_sam/outputs/logs/e2_box.csv').open()))
 if not rows or int(rows[-1]['epoch'])!=20:raise RuntimeError('E2未完成20轮，检查日志后处理，禁止标为完成')
 s['completed']=['E2_retrain20'];runner.write(STATUS,s)
 alljobs=runner.matrix()
 wanted={'E2_retrain20','E3_retrain6','train_size384','eval_size384'}
 wanted.update(j['name'] for j in alljobs if j['name'].startswith(('eval_e2_','eval_e3_')))
 jobs=[j for j in alljobs if j['name'] in wanted]
 runner.matrix=lambda:jobs
 runner.main()
if __name__=='__main__':main()
