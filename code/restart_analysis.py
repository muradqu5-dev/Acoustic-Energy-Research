"""Exact hybrid storage model with brownout, restart cost and standby policies.

Restart costs are storage-side assumptions, not measured component data.
The original constant-power electrical source is retained for comparability.
"""
from pathlib import Path
import csv,json,math
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from model import Scenario
from intermittent_analysis import advance,time_to

ROOT=Path(__file__).resolve().parent
OUT=ROOT/'extended_results'
PERIODS=[None,10,60,300,1200,3600]
COSTS=[0,25,100,200,300]

def window(s,period=None,boot_j=0.,gated=False,duration=21600.,initial=None):
    eh=.5*s.capacitance_f*s.high_voltage_v**2
    el=.5*s.capacitance_f*s.low_voltage_v**2
    if not math.isfinite(boot_j) or not 0<=boot_j<eh-el:
        raise ValueError('Restart cost must fit strictly inside usable storage window')
    if s.post_event_voltage_v() is None:raise ValueError('Task cannot fit storage window')
    e,awake=(el,True) if initial is None else initial
    initial_e=e;cost=s.event_energy_j/s.output_efficiency
    lam=0. if math.isinf(s.leakage_resistance_ohm) else 2/(s.capacitance_f*s.leakage_resistance_ohm)
    ledger=dict(source_J=0.,converted_J=0.,standby_J=0.,leakage_J=0.,task_J=0.,restart_J=0.)
    t=0.;events=[];boots=[];below=0.;minimum=e;maximum=e
    while t<duration-1e-8:
        if period is None:pin=s.source_power_w;end=duration
        else:
            cycle=math.floor((t+1e-8)/period);phase=t-cycle*period
            on=phase<period/2-1e-8
            pin=2*s.source_power_w if on else 0.
            end=min((cycle+(.5 if on else 1))*period,duration)
        while t<end-1e-9:
            pb=0. if gated and not awake else s.standby_power_w
            q=s.input_efficiency*pin-pb
            if awake and e<=el+1e-16 and q-lam*e<0:
                awake=False
                continue
            remaining=end-t
            if e<=1e-17 and q<=0:
                ledger['source_J']+=pin*remaining
                ledger['converted_J']+=s.input_efficiency*pin*remaining
                ledger['standby_J']+=s.input_efficiency*pin*remaining
                below+=remaining;e=0.;minimum=0.;t=end
                continue
            high=time_to(e,eh,q,lam)
            low=time_to(e,el,q,lam) if awake and q-lam*e<0 else math.inf
            zero=time_to(e,0.,q,lam) if q<0 else math.inf
            dt=min(remaining,high,low,zero)
            before=e;after=max(0.,advance(e,q,lam,dt))
            ledger['source_J']+=pin*dt;ledger['converted_J']+=s.input_efficiency*pin*dt
            ledger['standby_J']+=pb*dt
            ledger['leakage_J']+=max(0.,q*dt-(after-before))
            if before<el and after<el:below+=dt
            elif before<el<=after:below+=min(dt,time_to(before,el,q,lam))
            elif before>=el>after:below+=max(0.,dt-time_to(before,el,q,lam))
            e=after;t+=dt;minimum=min(minimum,e);maximum=max(maximum,e)
            if high<=dt+1e-8:
                if not awake:
                    awake=True;e=eh-boot_j;boots.append(t);ledger['restart_J']+=boot_j
                    if boot_j==0:
                        e-=cost;events.append(t);ledger['task_J']+=cost
                else:
                    e=eh-cost;events.append(t);ledger['task_J']+=cost
                minimum=min(minimum,e)
            elif low<=dt+1e-8:
                e=el;awake=False
            elif zero<=dt+1e-8:e=0.;awake=False
    residual=initial_e+ledger['converted_J']-e-sum(ledger[k] for k in ['standby_J','leakage_J','task_J','restart_J'])
    return dict(state=(e,awake),events=events,boots=boots,tasks=len(events),restarts=len(boots),
        tasks_per_hour=len(events)*3600/duration,max_gap_s=float(max(np.diff(events),default=float('nan'))),
        below_low_percent=100*below/duration,min_voltage_V=math.sqrt(2*minimum/s.capacitance_f),
        max_voltage_V=math.sqrt(2*maximum/s.capacitance_f),initial_energy_J=initial_e,
        residual_J=residual,ledger=ledger)

def simulate(period=None,boot_uj=0.,gated=False,s=None):
    s=Scenario() if s is None else s
    warm=window(s,period,boot_uj*1e-6,gated,3600)
    return window(s,period,boot_uj*1e-6,gated,21600,warm['state'])

def independent_rk4(s,period,boot_j,gated=False,duration=3600.,dt=.05):
    """Fixed-step reference: no calls to the event solver or its transition helpers."""
    e=.5*s.capacitance_f*s.low_voltage_v**2;el=e
    eh=.5*s.capacitance_f*s.high_voltage_v**2;awake=True
    lam=2/(s.capacitance_f*s.leakage_resistance_ohm)
    cost=s.event_energy_j/s.output_efficiency;events=[];boots=[]
    for n in range(round(duration/dt)):
        t=n*dt
        p=s.source_power_w if period is None else (2*s.source_power_w if (t+1e-8)%period<period/2 else 0.)
        pb=0. if gated and not awake else s.standby_power_w
        q=s.input_efficiency*p-pb
        if awake and e<=el+1e-16 and q-lam*e<0:
            awake=False;pb=0. if gated else s.standby_power_w;q=s.input_efficiency*p-pb
        f=lambda x:q-lam*x
        a=f(e);b=f(e+dt*a/2);c=f(e+dt*b/2);d=f(e+dt*c)
        e=max(0.,e+dt*(a+2*b+2*c+d)/6)
        if awake and e<el:awake=False
        if e>=eh:
            if not awake:
                awake=True;e-=boot_j;boots.append(t+dt)
                if boot_j==0:e-=cost;events.append(t+dt)
            else:e-=cost;events.append(t+dt)
    return dict(events=events,boots=boots,state=(e,awake))

def export():
    rows=[]
    for gated in [False,True]:
        for p in PERIODS:
            for boot in COSTS:
                r=simulate(p,boot,gated)
                rows.append(dict(policy='Power-gated' if gated else 'Continuous standby',period_s=p,
                    boot_uJ=boot,tasks_6h=r['tasks'],tasks_h=r['tasks_per_hour'],restarts=r['restarts'],
                    max_gap_s=r['max_gap_s'],below_low_percent=r['below_low_percent'],
                    first_3h=sum(x<=10800 for x in r['events']),last_3h=sum(x>10800 for x in r['events']),
                    residual_J=r['residual_J'],**r['ledger']))
    with (OUT/'restart_comparison.csv').open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=rows[0]);w.writeheader();w.writerows(rows)
    (OUT/'restart_summary.json').write_text(json.dumps(rows,indent=2),encoding='utf-8')
    checks=[]
    for gated in [False,True]:
        for p in [None,300,1200,3600]:
            exact=window(Scenario(),p,100e-6,gated,duration=7200)
            ref=independent_rk4(Scenario(),p,100e-6,gated,duration=7200)
            checks.append(dict(policy='gated' if gated else 'continuous',period_s=p,boot_uJ=100,
                exact_tasks=len(exact['events']),rk4_tasks=len(ref['events']),exact_restarts=len(exact['boots']),
                rk4_restarts=len(ref['boots']),max_event_error_s=max((abs(a-b) for a,b in zip(exact['events'],ref['events'])),default=0)))
    (OUT/'restart_verification.json').write_text(json.dumps(checks,indent=2),encoding='utf-8')
    plt.rcParams.update({'font.family':'serif','font.serif':['Times New Roman','DejaVu Serif'],'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    fig,axs=plt.subplots(1,2,figsize=(7,3.1))
    for p,ls,col,label in [(None,':','.6','Steady'),(300,'-.','.4','300 s cycle'),(1200,'-','black','1200 s cycle'),(3600,'--','.25','3600 s cycle')]:
        rr=[r for r in rows if r['period_s']==p and r['policy']=='Continuous standby']
        for ax,key in [(axs[0],'tasks_h'),(axs[1],'max_gap_s')]:
            ax.plot([r['boot_uJ'] for r in rr],[r[key] for r in rr],ls,marker='o',markersize=3,color=col,label=label)
    for ax in axs:ax.set_xlabel('Restart energy from storage (µJ)');ax.grid(alpha=.18)
    axs[0].set_ylabel('Completed tasks per hour');axs[1].set_ylabel('Longest intertask gap (s)')
    handles,labels=axs[0].get_legend_handles_labels()
    fig.legend(handles,labels,fontsize=8,frameon=False,loc='upper center',ncol=4)
    fig.tight_layout(rect=(0,0,1,.92))
    for ext in ['png','pdf']:fig.savefig(ROOT/'revised_figures'/f'restart_sensitivity.{ext}',dpi=320,bbox_inches='tight')
    return rows

if __name__=='__main__':
    results=export()
    print(json.dumps([r for r in results if r['period_s'] in [None,1200,3600] and r['boot_uJ'] in [0,100,300]],indent=2))
