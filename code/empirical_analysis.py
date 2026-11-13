"""Retrospective model comparison using a digitized published charging trace."""
from pathlib import Path
import csv
import hashlib
import json
import numpy as np
from PIL import Image
from scipy.optimize import least_squares
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parent
SOURCE=ROOT/'source_data'/'duan_figure5.png'
OUT=ROOT/'extended_results';OUT.mkdir(exist_ok=True)
FIG=ROOT/'revised_figures'

def energy_model(t, plateau, tau):
    return plateau*np.sqrt(-np.expm1(-np.asarray(t)/tau))

def voltage_model(t, plateau, tau):
    return plateau*(-np.expm1(-np.asarray(t)/tau))

def digitize():
    """Read median red-line height near each chosen time; original embedded image."""
    im=np.asarray(Image.open(SOURCE).convert('RGB')).astype(float)
    # Coordinates refer to the original 2050 x 1148 pixel Figure 5 image.
    x0,x1000,y0,y25=1545.0,2014.0,433.5,47.5
    points=[]
    for t in range(50,901,50):
        x=x0+t*(x1000-x0)/1000
        cols=range(round(x)-1,round(x)+2)
        ys=[]
        for col in cols:
            rgb=im[60:434,col,:]
            mask=(rgb[:,0]>150)&(rgb[:,0]-rgb[:,1]>65)&(rgb[:,0]-rgb[:,2]>40)
            found=np.where(mask)[0]+60
            if len(found):ys.append(float(np.median(found)))
        if not ys:raise ValueError(f'No red curve at {t} seconds')
        y=float(np.median(ys))
        points.append(dict(time_s=t,pixel_x=x,pixel_y=y,
            voltage_v=(y0-y)*2.5/(y0-y25),split='fit' if t<=300 else 'check'))
    with (OUT/'published_trace.csv').open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=points[0]);w.writeheader();w.writerows(points)
    meta={'source':'Duan et al. (2023), DOI 10.1002/aelm.202300348, Figure 5(c)',
      'url':'https://eprints.gla.ac.uk/302236/2/302236.pdf',
      'source_pdf_page':7,'embedded_image':'Im6.jpg','image_size':[2050,1148],
      'image_sha256':hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
      'axis_pixels':{'t0':x0,'t1000':x1000,'V0':y0,'V2_5':y25},
      'digitization_tolerance_V':0.02,'digitization_tolerance_s':5,
      'reported_conditions':{'capacitance_F':0.001,'sound_frequency_Hz':200,'sound_level_dB':100},
      'method':'Median red-line height in three columns; 18 readings spaced 50 s; no raw observations or replicates supplied.'}
    (OUT/'digitization_metadata.json').write_text(json.dumps(meta,indent=2),encoding='utf-8')
    return points

def fit_one(fn,t,y):
    # Log parameters keep both positive. Multiple starts check optimization stability.
    fits=[least_squares(lambda z:fn(t,*np.exp(z))-y,np.log(start),
          bounds=(np.log([.2,1]),np.log([20,100000])),xtol=1e-12,ftol=1e-12,gtol=1e-12)
          for start in [(2.2,150),(3,600),(5,1500)]]
    fit=min(fits,key=lambda f:float(np.sum(f.fun**2)))
    return np.exp(fit.x)

def analyse():
    points=digitize()
    t=np.array([r['time_s'] for r in points],float)
    y=np.array([r['voltage_v'] for r in points])
    train=t<=300;check=~train
    rows=[];pred=[]
    for name,fn in [('Energy balance',energy_model),('Voltage-limited',voltage_model)]:
        pars=fit_one(fn,t[train],y[train]);v=fn(t,*pars)
        row={'model':name,'plateau_V':float(pars[0]),'time_constant_s':float(pars[1]),
          'fit_RMSE_V':float(np.sqrt(np.mean((v[train]-y[train])**2))),
          'check_RMSE_V':float(np.sqrt(np.mean((v[check]-y[check])**2))),
          'check_max_abs_error_V':float(np.max(np.abs(v[check]-y[check]))),
          'prediction_900s_V':float(fn(900,*pars)),
          'parameter_at_bound':bool(np.any(np.isclose(pars,[20,100000],rtol=.001)))}
        if name=='Energy balance':
            row['effective_net_power_uW']=0.001*pars[0]**2/(2*pars[1])*1e6
            row['effective_resistance_kohm']=2*pars[1]/.001/1000
        # Deterministic perturbation, not a statistical confidence interval.
        stress=[]
        for dy in [-.02,0,.02]:
            for dt in [-5,0,5]:
                p=fit_one(fn,t[train]+dt,y[train]+dy)
                stress.append(float(np.sqrt(np.mean((fn(t[check]+dt,*p)-(y[check]+dy))**2))))
        row['digitization_stress_RMSE_range_V']=[min(stress),max(stress)]
        rows.append(row)
        for ti,yi,vi in zip(t,y,v):pred.append({'model':name,'time_s':ti,'digitized_voltage_V':yi,'model_voltage_V':vi,'residual_V':vi-yi,'split':'fit' if ti<=300 else 'check'})
    with (OUT/'published_comparison.csv').open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=pred[0]);w.writeheader();w.writerows(pred)
    (OUT/'empirical_summary.json').write_text(json.dumps(rows,indent=2),encoding='utf-8')
    plt.rcParams.update({'font.family':'serif','font.serif':['Times New Roman','DejaVu Serif'],'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    fig,ax=plt.subplots(1,2,figsize=(6.8,2.9),gridspec_kw={'width_ratios':[1.35,1]})
    ax[0].plot(t[train],y[train],'o',color='black',markersize=4,label='Published curve: fitting points')
    ax[0].plot(t[check],y[check],'o',mfc='white',mec='black',markersize=4,label='Published curve: checking points')
    tt=np.linspace(0,900,500)
    for i,(r,fn) in enumerate(zip(rows,[energy_model,voltage_model])):
        ax[0].plot(tt,fn(tt,r['plateau_V'],r['time_constant_s']),['--','-'][i],color=['.5','black'][i],label=r['model'])
        ax[1].plot(t,fn(t,r['plateau_V'],r['time_constant_s'])-y,['--','-'][i],color=['.5','black'][i],label=r['model'])
    for a in ax:a.axvline(300,color='.6',lw=.8,ls=':');a.set_xlabel('Time (s)');a.grid(axis='y',alpha=.18)
    ax[0].set_ylabel('Capacitor voltage (V)');ax[0].legend(frameon=False,fontsize=7,loc='lower right')
    ax[1].axhspan(-.02,.02,color='.9');ax[1].axhline(0,color='.6',lw=.7)
    ax[1].set_ylabel('Model minus digitized voltage (V)')
    ax[0].set_title('(a) Charging trace and fitted models',fontsize=10)
    ax[1].set_title('(b) Prediction errors',fontsize=10)
    fig.tight_layout(pad=.5)
    for ext in ['png','pdf']:fig.savefig(FIG/f'published_comparison.{ext}',dpi=320,bbox_inches='tight')
    plt.close(fig)
    return rows

if __name__=='__main__':print(json.dumps(analyse(),indent=2))
