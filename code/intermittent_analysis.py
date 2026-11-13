"""Exact event-driven storage simulation for equal-energy electrical inputs."""
from dataclasses import replace
from pathlib import Path
import csv,json,math
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from model import Scenario

ROOT=Path(__file__).resolve().parent
OUT=ROOT/'extended_results';OUT.mkdir(exist_ok=True)
FIG=ROOT/'revised_figures'

def advance(e,q,lam,dt):
    if lam==0:return e+q*dt
    return e+(q/lam-e)*(-math.expm1(-lam*dt))

def time_to(e,target,q,lam):
    if abs(e-target)<1e-17:return 0.0
    if lam==0:
        t=(target-e)/q if q else math.inf
        return t if t>=0 else math.inf
    eq=q/lam
    if (target>e and eq<=target) or (target<e and eq>=target):return math.inf
    return math.log((e-eq)/(target-eq))/lam

def run_window(s,period_s=None,duty=.5,duration_s=21600,initial_energy_j=None,record=False):
    if period_s is not None and (period_s<=0 or not 0<duty<1):raise ValueError('Invalid burst schedule')
    if s.post_event_voltage_v() is None:raise ValueError('Task exceeds usable energy window')
    e=.5*s.capacitance_f*s.low_voltage_v**2 if initial_energy_j is None else initial_energy_j
    if e<0:raise ValueError('Negative initial energy')
    initial=e;eh=.5*s.capacitance_f*s.high_voltage_v**2
    el=.5*s.capacitance_f*s.low_voltage_v**2
    cost=s.event_energy_j/s.output_efficiency
    lam=0.0 if math.isinf(s.leakage_resistance_ohm) else 2/(s.leakage_resistance_ohm*s.capacitance_f)
    t=0.0;events=[];trace=[(0.0,math.sqrt(2*e/s.capacitance_f))]
    ledger={'source_J':0.,'converted_J':0.,'standby_J':0.,'leakage_J':0.,'task_storage_J':0.,'below_low_s':0.,'empty_s':0.}
    min_e=e;max_e=e
    while t<duration_s-1e-9:
        if period_s is None:pin=s.source_power_w;end=duration_s
        else:
            cycle=math.floor((t+1e-8)/period_s)
            phase=t-cycle*period_s
            if phase<period_s*duty-1e-8:
                pin=s.source_power_w/duty;end=min((cycle+duty)*period_s,duration_s)
            else:pin=0.;end=min((cycle+1)*period_s,duration_s)
        q=s.input_efficiency*pin-s.standby_power_w
        while t<end-1e-9:
            remaining=end-t
            if e<=1e-17 and q<=0:
                ledger['source_J']+=pin*remaining
                ledger['converted_J']+=s.input_efficiency*pin*remaining
                ledger['standby_J']+=s.input_efficiency*pin*remaining
                ledger['below_low_s']+=remaining;ledger['empty_s']+=remaining
                e=0.;min_e=0.;t=end
                if record:trace.append((t,0.))
                continue
            hit=time_to(e,eh,q,lam)
            zero=time_to(e,0.,q,lam) if q<0 else math.inf
            dt=min(remaining,hit,zero)
            if dt<0:raise ArithmeticError('Negative time step')
            start=e;new=max(0.,advance(e,q,lam,dt))
            ledger['source_J']+=pin*dt;ledger['converted_J']+=s.input_efficiency*pin*dt
            ledger['standby_J']+=s.standby_power_w*dt
            # Integrate leakage analytically from the exact energy balance.
            leak=q*dt-(new-start)
            ledger['leakage_J']+=max(0.,leak)
            if start<el and new<el:ledger['below_low_s']+=dt
            elif start<el<=new:ledger['below_low_s']+=min(dt,time_to(start,el,q,lam))
            elif start>=el>new:ledger['below_low_s']+=max(0.,dt-time_to(start,el,q,lam))
            if record and dt>0:
                for td in np.linspace(0,dt,max(2,int(dt/10)+1))[1:]:
                    trace.append((t+float(td),math.sqrt(2*max(0,advance(start,q,lam,float(td)))/s.capacitance_f)))
            t+=dt;e=new;min_e=min(min_e,e);max_e=max(max_e,e)
            if hit<=dt+1e-8:
                e=eh-cost;events.append(t);ledger['task_storage_J']+=cost;min_e=min(min_e,e)
                if record:trace.append((t,math.sqrt(2*e/s.capacitance_f)))
            elif zero<=dt+1e-8:e=0.;min_e=0.
    ledger['energy_balance_error_J']=initial+ledger['converted_J']-ledger['standby_J']-ledger['leakage_J']-ledger['task_storage_J']-e
    return {'end_energy_J':e,'initial_energy_J':initial,'events':events,'trace':trace,
        'tasks':len(events),'tasks_per_hour':len(events)*3600/duration_s,
        'longest_intertask_s':max(np.diff(events),default=float('nan')),
        'min_voltage_V':math.sqrt(2*max(0,min_e)/s.capacitance_f),
        'max_voltage_V':math.sqrt(2*max_e/s.capacitance_f),
        'below_low_percent':100*ledger['below_low_s']/duration_s,'ledger':ledger}

def simulate(period_s=None,s=None,duty=.5,record=False):
    s=Scenario() if s is None else s
    warm=run_window(s,period_s,duty,3600)
    result=run_window(s,period_s,duty,21600,warm['end_energy_J'],record)
    result['period_s']=period_s;result['duty']=1. if period_s is None else duty
    result['warmup_s']=3600;result['observation_s']=21600
    return result

def rk4_check(s,period_s,duration_s=3600,dt=.1):
    """Independent fixed-step integrator; tasks triggered at step ends."""
    e=.5*s.capacitance_f*s.low_voltage_v**2;eh=.5*s.capacitance_f*s.high_voltage_v**2
    cost=s.event_energy_j/s.output_efficiency;lam=2/(s.capacitance_f*s.leakage_resistance_ohm)
    count=0;events=[]
    for k in range(round(duration_s/dt)):
        t=k*dt
        p=s.source_power_w if period_s is None else (2*s.source_power_w if ((t+1e-7)%period_s)<period_s/2 else 0.)
        q=s.input_efficiency*p-s.standby_power_w
        def f(x):return q-lam*x
        a=f(e);b=f(e+dt*a/2);c=f(e+dt*b/2);d=f(e+dt*c)
        e=max(0.,e+dt*(a+2*b+2*c+d)/6)
        if e>=eh:
            e-=cost;count+=1;events.append(t+dt)
    return count,events,e

def export():
    periods=[None,10,60,300,1200,3600]
    sims=[simulate(p,record=p in [None,1200]) for p in periods]
    rows=[]
    for r in sims:
        rows.append({'schedule':'Steady' if r['period_s'] is None else f"Burst T={r['period_s']} s",
            'period_s':r['period_s'],'tasks_6h':r['tasks'],'tasks_per_hour':r['tasks_per_hour'],
            'longest_intertask_s':float(r['longest_intertask_s']),'min_voltage_V':r['min_voltage_V'],
            'below_low_percent':r['below_low_percent'],
            **{k:float(v) for k,v in r['ledger'].items()}})
    with (OUT/'intermittent_comparison.csv').open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=rows[0]);w.writeheader();w.writerows(rows)
    # First and final three-hour rates expose slow start/phase effects.
    for row,s in zip(rows,sims):
        row['first_3h_tasks']=sum(t<=10800 for t in s['events'])
        row['last_3h_tasks']=sum(t>10800 for t in s['events'])
    (OUT/'intermittent_summary.json').write_text(json.dumps(rows,indent=2),encoding='utf-8')
    plt.rcParams.update({'font.family':'serif','font.serif':['Times New Roman','DejaVu Serif'],'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    fig,axes=plt.subplots(2,1,figsize=(6.8,3.7),sharex=True,gridspec_kw={'height_ratios':[1,2]})
    tt=np.linspace(0,3600,1801)
    axes[0].plot(tt/60,np.full_like(tt,4),'--',color='.5',label='Steady: 4 µW')
    axes[0].step(tt/60,np.where(tt%1200<600,8,0),where='post',color='black',label='Burst: 8 / 0 µW')
    axes[0].set_ylabel('Input (µW)');axes[0].set_ylim(-.4,11)
    axes[0].legend(frameon=False,fontsize=8,ncol=2,loc='upper right')
    for r,ls,col,label in [(sims[0],'--','.5','Steady'),(sims[4],'-','black','20-minute burst cycle')]:
        tr=np.array(r['trace']);tr=tr[tr[:,0]<=3600]
        axes[1].plot(tr[:,0]/60,tr[:,1],ls,color=col,lw=1,label=label)
    axes[1].axhline(2,color='.5',ls=':',lw=.8);axes[1].axhline(3.3,color='.5',ls=':',lw=.8)
    axes[1].set_ylabel('Storage voltage (V)');axes[1].set_xlabel('Time after warm-up (min)')
    axes[1].legend(frameon=False,fontsize=8,loc='lower right')
    for ax in axes:ax.grid(axis='y',alpha=.18);ax.set_xlim(0,60)
    fig.tight_layout(pad=.5)
    for ext in ['png','pdf']:fig.savefig(FIG/f'intermittent_input.{ext}',dpi=320,bbox_inches='tight')
    plt.close(fig)
    checks=[]
    for period in [None,60,300,1200,3600]:
        exact=run_window(Scenario(),period,duration_s=3600)
        count,events,_=rk4_check(Scenario(),period,dt=.1)
        checks.append({'schedule':'Steady' if period is None else f'{period} s cycle',
            'exact_tasks':exact['tasks'],'rk4_tasks':count,
            'max_event_difference_s':max(abs(a-b) for a,b in zip(events,exact['events']))})
    with (OUT/'event_verification.csv').open('w',newline='',encoding='utf-8') as f:
        writer=csv.DictWriter(f,fieldnames=checks[0]);writer.writeheader();writer.writerows(checks)
    return rows

if __name__=='__main__':print(json.dumps(export(),indent=2))
