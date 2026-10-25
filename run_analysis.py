"""Run selected analyses in dependency order. Without --run, print the commands."""
from pathlib import Path
import argparse, os, subprocess, sys, shutil

ROOT=Path(__file__).resolve().parent
GROUPS={
    'panel':['01_build_panel.py'],
    'storage':['03_extract_storage_events.py','04_build_landmarks.py'],
    'references':['08_twin_v2_profile.py','11_node_localization.py'],
    'injections':['09b_drywet_plane.py'],
    'rain':['12_rain_build_align.py'],
    'intervention':['13_retrofit_rain_causal.py','15_offset_and_headloss.py','18_causal_sensitivity.py','22_overlap_uncertainty.py'],
    'warning':['14_warning_with_rain.py','16_warning_robustness.py','16b_anchor_bootstrap.py','19_figure_data.py','20_si_data.py','21_calibration.py','17_adaptive_feedback.py'],
}
def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--steps',default=','.join(GROUPS))
    ap.add_argument('--sensor-dir',type=Path)
    ap.add_argument('--rain-dir',type=Path)
    ap.add_argument('--run',action='store_true')
    a=ap.parse_args();selected=a.steps.split(',')
    if set(selected)-GROUPS.keys():ap.error('Unknown steps: '+str(set(selected)-GROUPS.keys()))
    env=os.environ.copy()
    if a.sensor_dir:env['BELLINGE_SENSOR_DIR']=str(a.sensor_dir.resolve())
    if a.rain_dir:env['BELLINGE_RAIN_DIR']=str(a.rain_dir.resolve())
    if a.run:
        for key,step in [('BELLINGE_SENSOR_DIR','panel'),('BELLINGE_RAIN_DIR','rain')]:
            if step in selected and not Path(env.get(key,'__missing__')).is_dir():ap.error(key+' must be an existing data directory')
        for d in ['processed','results/figdata','results_corrected','logs']:(ROOT/d).mkdir(parents=True,exist_ok=True)
    for group,names in GROUPS.items():
        if group not in selected:continue
        for name in names:
            cmd=[sys.executable,str(ROOT/'scripts'/name)]
            if name=='03_extract_storage_events.py':cmd+=['--panel',str(ROOT/'processed/panel_5min.parquet'),'--output-dir',str(ROOT/'results')]
            if name=='13_retrofit_rain_causal.py':cmd+=['0'] # Verified unshifted rainfall clock.
            print(' '.join(cmd),flush=True)
            if a.run:
                with (ROOT/'logs'/f'{Path(name).stem}.log').open('w',encoding='utf-8') as log:
                    subprocess.run(cmd,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
                if name=='08_twin_v2_profile.py':
                    p=ROOT/'processed/twin2_G71F04R_Level2.parquet'
                    shutil.copyfile(p,p.with_name('twin2_G71F04R_Level2_frozen.parquet'))
if __name__=='__main__':main()
