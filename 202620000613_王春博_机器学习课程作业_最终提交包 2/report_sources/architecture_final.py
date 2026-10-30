from pathlib import Path
R=Path.cwd()
s=(R/'.polish_build/diagrams.py').read_text().replace("F=Path('submission_polished/figures')","F=Path('submission_refined/figures')").replace("B='#2167a6';G='#2c8167';O='#a46a30';T='#257a82'","B='#2878B5';G='#21A08B';O='#D59231';T='#5D71A8'").replace("'#eef5fc','#eff8f2','#fff5e9','#eef8f9'","'#ECF4FC','#EDF8F4','#FFF4E4','#F0F2FA'").replace('共享 A  8×768','层内 A  8×768').replace('r=8，α=16；B 的 k 输出行被屏蔽。','r=8，α=16；各层A/B独立，层内q/v共用A；B的k输出行被屏蔽。')
exec(compile(s,'diagrams_final','exec'))
