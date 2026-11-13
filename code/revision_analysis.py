"""Generate reproducible results and publication figures for the revised study."""
from dataclasses import replace
from pathlib import Path
import csv
import json
import math
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from model import Scenario, write_csv

ROOT=Path(__file__).resolve().parent
OUT=ROOT/'revised_results'
FIG=ROOT/'revised_figures'
OUT.mkdir(exist_ok=True); FIG.mkdir(exist_ok=True)
BASE=Scenario()
TAU=1000.0

def fixed_decay(capacitance_f, baseline=BASE, tau_s=TAU):
    if tau_s <= 0 or not math.isfinite(tau_s):
        raise ValueError('Decay time must be finite and positive.')
    return replace(baseline,capacitance_f=capacitance_f,leakage_resistance_ohm=tau_s/capacitance_f)

def maximum_capacitance(baseline=BASE,tau_s=TAU):
    return max(0.0,baseline.surplus_w*tau_s/baseline.high_voltage_v**2)

def rk4_endpoint(s,steps):
    t=s.recharge_time_s()
    if not math.isfinite(t): raise ValueError('Threshold is unreachable')
    e=0.5*s.capacitance_f*s.low_voltage_v**2
    h=t/steps
    def f(x):
        return s.surplus_w-2*x/(s.capacitance_f*s.leakage_resistance_ohm)
    for _ in range(steps):
        a=f(e);b=f(e+h*a/2);c=f(e+h*b/2);d=f(e+h*c)
        e+=h*(a+2*b+2*c+d)/6
    return e

def results():
    summary={'baseline':{
        'source_power_uW':4,'capacitance_uF':100,'efficiency_in':.5,'efficiency_out':.8,
        'standby_uW':.5,'leakage_Mohm':10,'low_V':2,'high_V':3.3,'event_uJ':100,
        'input_after_fixed_loss_uW':BASE.surplus_w*1e6,
        'leakage_at_high_uW':BASE.leakage_at_high_w*1e6,
        'usable_energy_uJ':BASE.usable_event_energy_j*1e6,
        'energy_removed_uJ':BASE.event_energy_j/BASE.output_efficiency*1e6,
        'min_capacitance_uF':BASE.minimum_capacitance_f*1e6,
        'critical_source_power_uW':BASE.critical_source_power_w*1e6,
        'full_window_charge_s':BASE.recharge_time_s(),
        'event_refill_s':BASE.fixed_event_recharge_time_s(),
        'after_event_V':BASE.post_event_voltage_v(),
        'no_leak_event_refill_s':replace(BASE,leakage_resistance_ohm=math.inf).fixed_event_recharge_time_s()},
        'fixed_decay':{'tau_s':TAU,'max_capacitance_uF_exclusive':maximum_capacitance()*1e6}}
    rows=[]
    for c in [10,22,47,100,220,470,1000]:
        a=replace(BASE,capacitance_f=c*1e-6);b=fixed_decay(c*1e-6)
        rows.append({'C_uF':c,'usable_energy_uJ':a.usable_event_energy_j*1e6,
            'Rfixed_full_charge_s':a.recharge_time_s(),
            'Rfixed_event_refill_s':a.fixed_event_recharge_time_s(),
            'taufixed_resistance_Mohm':b.leakage_resistance_ohm/1e6,
            'taufixed_critical_power_uW':b.critical_source_power_w*1e6,
            'taufixed_full_charge_s':b.recharge_time_s(),
            'taufixed_event_refill_s':b.fixed_event_recharge_time_s()})
    write_csv(OUT/'capacitance_comparison.csv',rows)
    power=[]
    for p in [.5,1,2,3,3.178,4,8,8.45,16]:
        s=replace(BASE,source_power_w=p*1e-6)
        power.append({'source_uW':p,'equilibrium_V':s.equilibrium_voltage_v,
            'charge_s':s.recharge_time_s(),'event_refill_s':s.fixed_event_recharge_time_s()})
    write_csv(OUT/'power_comparison.csv',power)
    losses=[]
    for eta in [.25,.5,.75]:
        for q in [.1,.5,1.0]:
            s=replace(BASE,input_efficiency=eta,standby_power_w=q*1e-6)
            losses.append({'eta_in':eta,'standby_uW':q,'power_threshold_uW':s.critical_source_power_w*1e6,
                'charge_s_at_4uW':s.recharge_time_s()})
    write_csv(OUT/'loss_comparison.csv',losses)
    verify=[]
    for name,s in [('Baseline',BASE),('8 uW source',replace(BASE,source_power_w=8e-6)),
                   ('No leakage',replace(BASE,leakage_resistance_ohm=math.inf)),
                   ('47 uF fixed decay',fixed_decay(47e-6))]:
        for n in [10,100,1000]:
            e=rk4_endpoint(s,n)
            exact=.5*s.capacitance_f*s.high_voltage_v**2
            verify.append({'case':name,'steps':n,'endpoint_relative_energy_error':abs(e-exact)/exact,
                'endpoint_voltage_V':math.sqrt(2*e/s.capacitance_f)})
    write_csv(OUT/'numerical_verification.csv',verify)
    (OUT/'summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    return summary

def figures():
    plt.rcParams.update({'font.family':'serif','font.serif':['Liberation Serif','DejaVu Serif'],
        'mathtext.fontset':'stix','font.size':10,'axes.labelsize':11,'legend.fontsize':9,
        'axes.spines.top':False,'axes.spines.right':False,'axes.linewidth':.65,
        'xtick.direction':'out','ytick.direction':'out','pdf.fonttype':42})
    fig,ax=plt.subplots(figsize=(6.5,3.15),layout='constrained')
    tt=np.linspace(0,1200,301)
    for p,style,color in [(2,':','0.5'),(4,'-','0.05'),(8,'--','0.3')]:
        s=replace(BASE,source_power_w=p*1e-6)
        ax.plot(tt,[s.voltage_at_s(float(t)) for t in tt],style,color=color,lw=1.7,label=f'{p} µW input')
    ax.axhline(3.3,color='0.5',lw=.7,ls='-.')
    ax.annotate('3.3 V threshold',xy=(1180,3.3),xytext=(1180,3.02),ha='right',fontsize=9)
    ax.set(xlim=(0,1200),ylim=(1.8,6),xlabel='Time (s)',ylabel='Capacitor voltage (V)')
    ax.legend(frameon=False,loc='upper left');ax.grid(axis='y',lw=.4,color='.87')
    fig.savefig(FIG/'charging.pdf');fig.savefig(FIG/'charging.png',dpi=300);plt.close(fig)

    fig,axs=plt.subplots(1,2,figsize=(6.5,3.0),layout='constrained')
    xx=np.linspace(36.285,1000,250)
    full=[];event=[]
    for c in xx:
        s=replace(BASE,capacitance_f=float(c)*1e-6)
        full.append(s.recharge_time_s()/60);event.append(s.fixed_event_recharge_time_s()/60)
    axs[0].plot(xx,full,color='.1',label='Full voltage window',lw=1.4)
    axs[0].plot(xx,event,'--',color='.35',label='One 100 µJ event',lw=1.4)
    axs[0].set(xlim=(0,1000),ylim=(0,90),xlabel='Capacitance (µF)',ylabel='Recharge time (min)',title='(a) Fixed leakage resistance')
    axs[0].legend(frameon=False,fontsize=8,loc='upper left')
    xx=np.linspace(10,300,250)
    pp=[fixed_decay(float(c)*1e-6).critical_source_power_w*1e6 for c in xx]
    axs[1].plot(xx,pp,color='.1',lw=1.5,label='Required source power')
    axs[1].axhline(4,color='.4',ls='--',lw=1.1,label='Available power: 4 µW')
    axs[1].axvline(BASE.minimum_capacitance_f*1e6,color='.55',ls=':',lw=1)
    axs[1].axvline(maximum_capacitance()*1e6,color='.55',ls=':',lw=1)
    axs[1].axvspan(BASE.minimum_capacitance_f*1e6,maximum_capacitance()*1e6,color='.92')
    axs[1].text(87,7.8,'Energy-feasible\ninterval',ha='center',fontsize=8)
    axs[1].set(xlim=(0,300),ylim=(0,9),xlabel='Capacitance (µF)',ylabel='Source power (µW)',title='(b) Fixed self-discharge time')
    axs[1].legend(frameon=False,fontsize=8,loc='lower right')
    for ax in axs:ax.grid(axis='y',lw=.4,color='.87');ax.title.set_fontsize(10)
    fig.savefig(FIG/'capacitance.pdf');fig.savefig(FIG/'capacitance.png',dpi=300);plt.close(fig)

if __name__=='__main__':
    print(json.dumps(results(),indent=2));figures()
