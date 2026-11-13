"""Operating-voltage control and source-model robustness; explicit scenario assumptions."""
from pathlib import Path
from dataclasses import replace
import json,csv,math
import numpy as np
from scipy.integrate import solve_ivp,quad
from scipy.optimize import brentq
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from model import Scenario
from restart_analysis import simulate
from empirical_analysis import fit_one,energy_model,voltage_model

ROOT=Path(__file__).resolve().parent
OUT=ROOT/'extended_results'

def source_window(s,voc,pref,period=None,boot=0.,duration=21600.,initial=None,method='DOP853',max_step=60.):
    """Thevenin source, constant input efficiency, ideal shutdown below VL.
    pref is pre-conversion source power at Vref=2.8 V during steady excitation.
    Bursts double source conductance for half a period; source is isolated when off.
    """
    c=s.capacitance_f;vh=s.high_voltage_v;vl=s.low_voltage_v
    task=s.event_energy_j/s.output_efficiency
    if task>.5*c*(vh*vh-vl*vl)+1e-15 or boot>=.5*c*(vh*vh-vl*vl):raise ValueError('Energy window too small')
    if voc<=2.8 or pref<=0:raise ValueError('Invalid source')
    rs=2.8*(voc-2.8)/pref
    v,awake=(vl,True) if initial is None else initial
    initial_e=.5*c*v*v;t=0.;events=[];boots=[]
    ledger=np.zeros(3) # pre-conversion source, capacitor leakage, standby
    task_j=boot_j=0.;cache={}
    while t<duration-1e-7:
        if period is None:g=1.;end=duration
        else:
            cyc=math.floor((t+1e-7)/period);phase=t-cyc*period
            on=phase<period/2-1e-7;g=2. if on else 0.
            end=min(duration,(cyc+(.5 if on else 1))*period)
        while t<end-1e-7:
            pb=s.standby_power_w if awake else 0.
            def rhs(tt,y):
                vv=max(float(y[0]),1e-15)
                pin=g*max(0.,vv*(voc-vv))/rs
                leak=vv*vv/s.leakage_resistance_ohm
                dv=(s.input_efficiency*pin-leak-pb)/(c*vv)
                return [dv,pin,leak,pb]
            if awake and v<=vl+1e-10 and rhs(t,[v])[0]<0:
                awake=False;continue
            def high(tt,y):return y[0]-vh
            high.terminal=True;high.direction=1
            def low(tt,y):return y[0]-vl
            low.terminal=True;low.direction=-1
            ev=[high,low] if awake else [high]
            key=(round(v,13),awake,g);cached=cache.get(key)
            if cached is not None and cached[0]<=end-t+1e-8:
                dt,yy,hit=cached
            else:
                sol=solve_ivp(rhs,(0,end-t),[v,0.,0.,0.],method=method,events=ev,rtol=2e-10,
                    atol=[1e-11,1e-14,1e-14,1e-14],max_step=max_step)
                if not sol.success:raise RuntimeError(sol.message)
                dt=float(sol.t[-1]);yy=sol.y[:,-1]
                hit='high' if len(sol.t_events[0]) else ('low' if awake and len(sol.t_events[1]) else None)
                if hit is not None:cache[key]=(dt,yy,hit)
            v=float(yy[0]);ledger+=yy[1:];t+=dt
            if hit=='high':
                if awake:
                    v=math.sqrt(vh*vh-2*task/c);events.append(t);task_j+=task
                else:
                    awake=True;v=math.sqrt(vh*vh-2*boot/c);boots.append(t);boot_j+=boot
                    if boot==0:v=math.sqrt(vh*vh-2*task/c);events.append(t);task_j+=task
            elif hit=='low':v=vl;awake=False
    final_e=.5*c*v*v
    residual=initial_e+s.input_efficiency*ledger[0]-final_e-ledger[1]-ledger[2]-task_j-boot_j
    return dict(state=(v,awake),events=events,boots=boots,tasks=len(events),tasks_h=len(events)*3600/duration,
        max_gap_s=float(max(np.diff(events),default=float('nan'))),source_J=float(ledger[0]),
        leakage_J=float(ledger[1]),standby_J=float(ledger[2]),task_J=task_j,restart_J=boot_j,
        residual_J=float(residual),source_R_Mohm=rs/1e6,pref_uW=pref*1e6)

def source_run(s,voc,pref,period=None,boot=0.,method='DOP853',max_step=60.):
    warm=source_window(s,voc,pref,period,boot,3600.,method=method,max_step=max_step)
    return source_window(s,voc,pref,period,boot,21600.,warm['state'],method,max_step)

def matched_source(s,voc,period,boot):
    target=s.source_power_w*21600
    def mismatch(logpref):
        return source_run(s,voc,math.exp(logpref),period,boot)['source_J']/target-1
    x=brentq(mismatch,math.log(.5e-6),math.log(20e-6),xtol=2e-9)
    result=source_run(s,voc,math.exp(x),period,boot)
    return dict(voc_V=voc,VH_V=s.high_voltage_v,period_s=period,boot_uJ=boot*1e6,
        **{k:v for k,v in result.items() if k not in ['state','events','boots']})

def threshold_controls():
    rows=[]
    for v in [3.3,2.8,2.55]:
        s=replace(Scenario(),high_voltage_v=v)
        for period in [None,1200]:
            for boot in [0,100]:
                r=simulate(period,boot,False,s)
                rows.append(dict(VH_V=v,period_s=period,boot_uJ=boot,tasks_h=r['tasks_per_hour'],
                    max_gap_s=r['max_gap_s'],task_margin_uJ=(s.usable_event_energy_j-s.event_energy_j)*1e6,
                    required_C_uF=2*s.event_energy_j/(s.output_efficiency*(v*v-s.low_voltage_v**2))*1e6,
                    source_J=r['ledger']['source_J'],residual_J=r['residual_J']))
    return rows

def fit_split_checks():
    with (OUT/'published_trace.csv').open(encoding='utf-8',newline='') as f:
        rows=list(csv.DictReader(f))
    t=np.array([float(x['time_s']) for x in rows]);y=np.array([float(x['voltage_v']) for x in rows])
    out=[]
    for split in [200,300,450]:
        fit=t<=split;check=~fit
        for name,fn in [('Energy',energy_model),('Voltage',voltage_model)]:
            a,tau=fit_one(fn,t[fit],y[fit]);p=fn(t,a,tau)
            out.append(dict(split_s=split,model=name,nfit=int(sum(fit)),ncheck=int(sum(check)),A_V=float(a),tau_s=float(tau),
                fit_rmse_V=float(np.sqrt(np.mean((p[fit]-y[fit])**2))),check_rmse_V=float(np.sqrt(np.mean((p[check]-y[check])**2)))))
    return out

def acoustic_budget():
    target=Scenario().critical_source_power_w
    # Plane-progressive wave approximation, rho*c=400 Pa s/m, normal incidence.
    rows=[]
    for spl in [70,80,90,100]:
        intensity=1e-12*10**(spl/10)
        rows.append(dict(SPL_dB=spl,intensity_W_m2=intensity,
            area_for_4uW_cm2=[4e-6/(intensity*eta)*1e4 for eta in [1.,.1,.01]],
            area_for_threshold_10pct_cm2=target/(intensity*.1)*1e4))
    spls=[dict(area_cm2=a,efficiency=eta,SPL_threshold_dB=10*math.log10(target/(1e-12*a*1e-4*eta)))
        for a in [4,20,50] for eta in [.01,.1,1.]]
    area=math.pi*(.0821/2)**2;pin=.01*area
    return dict(rows=rows,thresholds=spls,duan_area_cm2=area*1e4,duan_incident_uW=pin*1e6,
        duan_resistive_efficiency=8.45e-6/pin,duan_reported_storage_efficiency=2.39e-6/pin,
        duan_endpoint_power_uW=.5*.001*2.165**2/690*1e6,
        duan_endpoint_efficiency=(.5*.001*2.165**2/690)/pin,
        duan_implied_area_cm2=12.04e-6/.01*1e4)

def leakage_ranges():
    rows=[]
    for rmeg in [.1,.33,.55,1,5,10,50,100]:
        for pb in [.86875,2.3125]:
            for eta in [.25,.5,.75]:
                s=replace(Scenario(),leakage_resistance_ohm=rmeg*1e6,standby_power_w=pb*1e-6,input_efficiency=eta,
                    event_energy_j=6.90525e-6 if pb<1 else 12.705e-6)
                rows.append(dict(R_Mohm=rmeg,standby_uW=pb,eta_in=eta,required_source_uW=s.critical_source_power_w*1e6,
                    refill_at_8uW_s=None if math.isinf(replace(s,source_power_w=8e-6).fixed_event_recharge_time_s())
                    else replace(s,source_power_w=8e-6).fixed_event_recharge_time_s()))
    return rows

def steady_source_refill(voc,pref,s):
    rs=2.8*(voc-2.8)/pref
    start=s.post_event_voltage_v()
    if start is None:return None
    def dv(v):return (s.input_efficiency*v*(voc-v)/rs-s.standby_power_w-v*v/s.leakage_resistance_ohm)/(s.capacitance_f*v)
    if min(dv(float(v)) for v in np.linspace(start,s.high_voltage_v,101))<=0:return None
    return quad(lambda v:1/dv(v),start,s.high_voltage_v,epsabs=1e-8,epsrel=1e-10)[0]

def export():
    controls=threshold_controls();splits=fit_split_checks();acoustic=acoustic_budget();leak=leakage_ranges()
    rows=[]
    for v in [3.3,2.55]:
        for voc in [3.6,5.,10.]:
            for p in [None,1200]:
                for boot in [0.,100e-6]:
                    row=matched_source(replace(Scenario(),high_voltage_v=v),voc,p,boot);rows.append(row)
                    print('source',v,voc,p,boot,row['tasks_h'],flush=True)
    cp=[]
    for v in [3.3,2.55]:
        for p in [None,1200]:
            for boot in [0,100]:
                r=simulate(p,boot,True,replace(Scenario(),high_voltage_v=v))
                cp.append(dict(VH_V=v,period_s=p,boot_uJ=boot,tasks_h=r['tasks_per_hour'],max_gap_s=r['max_gap_s']))
    sensor=[]
    for voc in [3.6,5.,10.]:
        for pin in [4e-6,8e-6]:
            # Match power at VH for a local replenishment test, not the time-average matching above.
            rs=3.3*(voc-3.3)/pin;pref=2.8*(voc-2.8)/rs
            for label,pb,e in [('Typical',.86875e-6,6.90525e-6),('Limit',2.3125e-6,12.705e-6)]:
                s=replace(Scenario(),leakage_resistance_ohm=5e6,standby_power_w=pb,event_energy_j=e)
                sensor.append(dict(voc_V=voc,power_at_high_uW=pin*1e6,case=label,refill_s=steady_source_refill(voc,pref,s)))
    data=dict(threshold=controls,split_checks=splits,acoustic=acoustic,leakage=leak,voltage_source=rows,constant_power_gated=cp,source_sensor=sensor)
    (OUT/'robustness_summary.json').write_text(json.dumps(data,indent=2),encoding='utf-8')
    for name,items in [('threshold_controls',controls),('fit_split_checks',splits),('source_robustness',rows),('leakage_ranges',leak)]:
        with (OUT/(name+'.csv')).open('w',newline='',encoding='utf-8') as f:
            w=csv.DictWriter(f,fieldnames=items[0]);w.writeheader();w.writerows(items)
    plt.rcParams.update({'font.family':'serif','font.serif':['Times New Roman','DejaVu Serif'],'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    fig,axs=plt.subplots(1,2,figsize=(7,3.0))
    for per,col,marker,label in [(None,'black','o','Steady'),(1200,'.5','s','1200 s cycle')]:
        rr=[r for r in controls if r['period_s']==per and r['boot_uJ']==0]
        for ax,key in [(axs[0],'tasks_h'),(axs[1],'max_gap_s')]:ax.plot([r['VH_V'] for r in rr],[r[key] for r in rr],marker=marker,color=col,label=label)
    axs[0].set_ylabel('Completed tasks per hour');axs[1].set_ylabel('Longest intertask gap (s)')
    for ax in axs:ax.set_xlabel('Upper storage threshold (V)');ax.grid(alpha=.2)
    axs[0].legend(frameon=False);fig.tight_layout()
    for ext in ['png','pdf']:fig.savefig(ROOT/'revised_figures'/('threshold_control.'+ext),dpi=300,bbox_inches='tight')
    return data

if __name__=='__main__':
    data=export();print('Saved robustness analyses')
