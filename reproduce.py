"""Run the acoustic-energy calculations, figures and numerical checks."""
from pathlib import Path
import argparse
import subprocess
import sys

ROOT=Path(__file__).resolve().parent
CODE=ROOT/'code'
ANALYSES=['revision_analysis.py','empirical_analysis.py','intermittent_analysis.py',
          'restart_analysis.py','component_analysis.py','robustness_analysis.py','tolerance_analysis.py']
TESTS=['test_model','test_revision','test_extended','test_restart','test_robustness']

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check-only',action='store_true',help='Check the model and supplied reference results without rerunning every sweep.')
    args=parser.parse_args()
    for folder in ['extended_results','revised_results','revised_figures']:
        (CODE/folder).mkdir(exist_ok=True)
    log_path=CODE/'run.log'
    with log_path.open('w',encoding='utf-8') as log:
        def run(label,arguments):
            print(label,flush=True)
            log.write('\n'+label+'\n');log.flush()
            result=subprocess.run([sys.executable,*arguments],cwd=CODE,stdout=log,stderr=subprocess.STDOUT)
            if result.returncode:
                log.flush()
                print('\n'.join(log_path.read_text(encoding='utf-8',errors='replace').splitlines()[-30:]))
                raise SystemExit(f'Calculation failed; see {log_path}')
        if not args.check_only:
            for script in ANALYSES:run('Running '+script,[script])
        run('Running 60 numerical checks',['-m','unittest','-v',*TESTS])
        run('Checking capacitance sensitivity',['-c','from tolerance_analysis import calculate,check; check(calculate()); print("Capacitance checks passed")'])
    print('Completed successfully. Detailed output: '+str(log_path),flush=True)

if __name__=='__main__':main()
