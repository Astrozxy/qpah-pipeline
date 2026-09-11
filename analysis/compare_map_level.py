import os, time, json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import SymLogNorm, TwoSlopeNorm, LogNorm
from astropy.io import fits
from astropy.wcs import WCS
from scipy import stats

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))          # repo 根目录
ROOT = os.path.join(_ROOT, 'data')
CACHE = os.path.join(_ROOT, 'sanity_cache')
OUT = os.path.join(_ROOT, 'sanity_output')
os.makedirs(OUT, exist_ok=True)
T0=time.time()
def log(*a): print('[%.0fs]'%(time.time()-T0), *a, flush=True)

hq=fits.open(ROOT+'/M31_S350_110_SSS_110_Model_All_qpah.fits')
wout=WCS(hq[0].header); shape=hq[0].data.shape; hq.close()
ny,nx=shape
yy,xx=np.mgrid[0:ny,0:nx]
ra_g,dec_g=wout.pixel_to_world_values(xx.astype(np.float64),yy.astype(np.float64))
log('grid coords ready', shape)

def load_native(fn, cube=False):
    if cube:
        p=CACHE+'/CO_mom0.npy'
        if os.path.exists(p): d=np.load(p)
        else:
            hd=fits.open(ROOT+'/'+fn); d=np.nansum(hd[0].data,axis=0); hd.close(); np.save(p,d)
        hd=fits.open(ROOT+'/'+fn); w=WCS(hd[0].header).celestial; hd.close()
    else:
        hd=fits.open(ROOT+'/'+fn); d=hd[0].data.astype(np.float64,copy=False); w=WCS(hd[0].header); hd.close()
    return d,w

def old_center_sample(wcs_in, data):
    xf,yf=wcs_in.world_to_pixel_values(ra_g,dec_g)
    xi=np.rint(xf).astype(np.int64); yi=np.rint(yf).astype(np.int64)
    nrow,ncol=data.shape
    ok=(xi>=0)&(xi<ncol)&(yi>=0)&(yi<nrow)
    out=np.full(shape,np.nan)
    out[ok]=data[yi[ok],xi[ok]]
    return out

maps_def={
 'CO':   dict(fn='m31_co10_native.fits', cube=True),
 'HI':   dict(fn='m31_C+D+tp_hilores_120pc_strict_mom0.fits', cube=False),
 'HI_ew':dict(fn='m31_C+D+tp_hilores_120pc_strict_ew.fits', cube=False),
 'SFR':  dict(fn='M31-SFR.fits', cube=False),
 'dust': dict(fn='M31_dust_NH2_sm24.fits', cube=False),
}
summary={}
diffmaps={}
for k,cfg in maps_def.items():
    t=time.time()
    d,w=load_native(cfg['fn'],cfg.get('cube',False))
    old=old_center_sample(w,d)
    new=np.load(CACHE+'/'+k+'_ex_mean.npy'); cov=np.load(CACHE+'/'+k+'_ex_cov.npy')
    log(k,'loaded & old-sampled %.0fs'%(time.time()-t))
    # intersection masks
    m05=np.isfinite(old)&np.isfinite(new)&(cov>0.5)
    m99=np.isfinite(old)&np.isfinite(new)&(cov>0.999)
    diff=new-old
    # relative residual on |new| meaningful pixels
    scale=np.nanpercentile(np.abs(new[m05]),99)
    rel=diff/np.maximum(np.abs(new), scale*1e-3)
    entry={'n_cov>0.5':int(m05.sum()),'n_cov>0.999':int(m99.sum())}
    for tag,m in [('>0.5',m05),('>0.999',m99)]:
        d5,d16,d50,d84,d95=np.nanpercentile(diff[m],[5,16,50,84,95])
        a50,a90=np.nanpercentile(np.abs(diff[m]),[50,90])
        r16,r50,r84=np.nanpercentile(rel[m],[16,50,84])
        entry[tag]={'diff_pct':[float(x) for x in (d5,d16,d50,d84,d95)],
                    'absdiff_p50_p90':[float(a50),float(a90)],
                    'rel_pct':[float(x) for x in (r16,r50,r84)]}
    # positive-only log10 comparison (SFR/HI/dust-like)
    pos=(m05)&(old>0)&(new>0)
    if pos.sum()>100:
        ldiff=np.log10(new[pos])-np.log10(old[pos])
        l50=np.median(ldiff); l16,l84=np.percentile(ldiff,[16,84])
        entry['log10(new/old) pos n=%d'%pos.sum()]={'pct':[float(x) for x in (l16,l50,l84)],
            'median|dlog|':float(np.median(np.abs(ldiff)))}
        rr=stats.spearmanr(np.log10(new[pos]),np.log10(old[pos])).statistic
        entry['spearman_log_pos']=float(rr)
    summary[k]=entry
    diffmaps[k]=(diff,m05,m99)
    log(k,'m05=%d m99=%d |diff| p50=%.4g p90=%.4g   rel p16/p50/p84=%.3g/%.3g/%.3g'
        %(m05.sum(),m99.sum(),entry['>0.5']['absdiff_p50_p90'][0],entry['>0.5']['absdiff_p50_p90'][1],
          entry['>0.5']['rel_pct'][0],entry['>0.5']['rel_pct'][1],entry['>0.5']['rel_pct'][2]))

json.dump(summary,open(OUT+'/old_vs_new_stats.json','w'),indent=1)
log('stats saved')

# ---------- fig4: residual maps (new-old) ----------
fig,axes=plt.subplots(2,3,figsize=(16,9),constrained_layout=True)
for ax,(k,(diff,m05,m99)) in zip(axes.flat,diffmaps.items()):
    v=np.nanpercentile(np.abs(diff[m05]),99)
    if v==0: v=1
    cmap='coolwarm'
    im=ax.imshow(diff,origin='lower',cmap=cmap,norm=TwoSlopeNorm(vmin=-v,vcenter=0,vmax=v),aspect='auto')
    ax.set_title(k+': new-old (cov>0.5 px=%d)'%m05.sum())
    fig.colorbar(im,ax=ax,shrink=0.85)
    d_=diff[~np.isnan(diff)]
    ax.set_facecolor('0.85')
fig.suptitle('Residual maps: reproject_exact block minus old center-sample')
fig.savefig(OUT+'/fig4_diff_maps.png',dpi=140); plt.close(fig)

# ---------- fig5: new vs old scatter 2D hist ----------
fig,axes=plt.subplots(2,3,figsize=(16,9),constrained_layout=True)
for ax,(k,(diff,m05,m99)) in zip(axes.flat,diffmaps.items()):
    # reload old properly
    d,w=load_native(maps_def[k]['fn'],maps_def[k].get('cube',False))
    old=old_center_sample(w,d)
    new=np.load(CACHE+'/'+k+'_ex_mean.npy')
    mm=m05 & (old>0)&(new>0)
    if mm.sum()>500:
        if k=='SFR':
            # SFR 动态范围约 5 个数量级：对数据先取 log10 再 hexbin（避免 set_xscale 扭曲 binning）
            x=np.log10(old[mm]); y=np.log10(new[mm])
            v=np.concatenate([x,y])
            lo=np.nanpercentile(v,0.1); hi=np.nanpercentile(v,99.9)
            hb=ax.hexbin(x,y,gridsize=100,bins='log',cmap='viridis',mincnt=1,
                         extent=(lo,hi,lo,hi))
            fig.colorbar(hb,ax=ax,shrink=0.85)
            ax.plot([lo,hi],[lo,hi],'r--',lw=1)
            ax.set_xlim(lo,hi); ax.set_ylim(lo,hi)
            t0,t1=int(np.ceil(lo)),int(np.floor(hi))
            ticks=list(range(t0,t1+1))
            ax.set_xticks(ticks); ax.set_yticks(ticks)
            ax.set_xticklabels([r'$10^{%d}$'%t for t in ticks])
            ax.set_yticklabels([r'$10^{%d}$'%t for t in ticks])
            ax.set_xlabel('old (center sample), SFR'); ax.set_ylabel('new (exact block), SFR')
        else:
            x=old[mm]; y=new[mm]
            hb=ax.hexbin(x,y,gridsize=80,bins='log',cmap='viridis',mincnt=1)
            fig.colorbar(hb,ax=ax,shrink=0.85)
            lim=(min(np.nanmin(x),np.nanmin(y)),max(np.nanmax(x),np.nanmax(y)))
            ax.plot(lim,lim,'r--',lw=1)
            ax.set_xlabel('old (center sample)'); ax.set_ylabel('new (exact block)')
    else:
        ax.scatter(old[m05],new[m05],s=0.2)
        ax.set_xlabel('old (center sample)'); ax.set_ylabel('new (exact block)')
    ax.set_title(k)
fig.suptitle('new vs old per-pixel (positive values)')
fig.savefig(OUT+'/fig5_new_vs_old_scatter.png',dpi=140); plt.close(fig)
log('figures saved'); log('ALL DONE')
