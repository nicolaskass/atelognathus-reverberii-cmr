"""
cmr_robust_design.py  v2
=========================
Bayesian Pollock Robust Design CMR — Atelognathus reverberii
Laguna Azul, Meseta de Somuncurá, 2019-2020 (austral spring-summer)

STUDY DESIGN
------------
3 primary sessions (population open between, closed within):

  Primary 1  Oct 2019  [austral spring, frogs emerging from underground]:
      4 secondary occasions: 08, 09, 10, 11 Oct
  Primary 2  Nov 2019  [late spring / early summer, peak surface activity]:
      5 secondary occasions: 24, 26, 27, 28, 29 Nov
  Primary 3  Feb 2020  [late summer, activity declining]:
      5 secondary occasions: 01, 02, 03, 04, 05 Feb

  NOTE on 25-Nov: zero-effort weather day, absent from dataset (correct).
  NOTE on 10-Oct: 6 real captures despite adverse conditions; included
      (as in R's robustd.0 analysis) because captures are real events.

BIOLOGICAL CONTEXT
------------------
Laguna Azul is an endorheic basin; the population is geographically
closed. Low apparent survival between primary sessions (especially
phi_12, Nov->Feb) is interpreted as frogs returning to underground
refugia rather than mortality. Cei (1969) proposed that A. reverberii
lives primarily underground and exhibits explosive surface activity
tied to rainfall events rather than a fixed breeding season.

PARAMETERS
----------
phi_01  : apparent survival / detectability, Oct->Nov (~7 weeks)
phi_12  : apparent survival / detectability, Nov->Feb (~10 weeks)
p[t]    : per-secondary-occasion detection probability (session-specific)

PRIORS
------
logit(phi_t)  ~ Normal(2.0, 2.0)   -> mean phi ~0.88, broad
logit(p[t])   ~ Normal(-2.0, 1.5)  -> mean p ~0.12, weakly informative

Prior justification: Rolón et al. (2025, Aquat. Conserv.) report
annual phi ~0.96 for congeneric A. patagonicus; incommensurable
timescale and burrowing life-history motivate a broad prior.

MCMC: Adaptive Metropolis-Hastings, 4 chains, 3000 warmup + 5000 draws.

COMPARISON
----------
MLE equivalent: robustd.0(X, dfreq=FALSE, vt=c(4,5,5), vm="M0") in R.
"""

import argparse, warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy import stats
from scipy.optimize import minimize
from scipy.special import expit as sigmoid, logit

warnings.filterwarnings('ignore')

# ── Constants ─────────────────────────────────────────────────────────────────

J      = np.array([4, 5, 5])
T      = 3
STARTS = [0, 4, 9]
LABELS_SHORT = ['Oct 2019', 'Nov 2019', 'Feb 2020']
SP     = r'$\it{Atelognathus\ reverberii}$'

PALETTE = {
    'phi': ['#1b7837', '#762a83'],
    'p':   ['#4393c3', '#d6604d', '#f4a582'],
    'pd':  ['#2166ac', '#b2182b', '#e08214'],
    'N':   ['#006837', '#8e0152', '#b35806'],
}
plt.rcParams.update({'font.family': 'serif', 'font.size': 10,
                     'axes.spines.top': False, 'axes.spines.right': False})

PARAM_NAMES = ['phi_01', 'phi_12',
               'p_oct', 'p_nov', 'p_feb',
               'pd_oct', 'pd_nov', 'pd_feb',
               'N_oct', 'N_nov', 'N_feb']

PARAM_LABELS = {
    'phi_01': r'$\phi_{Oct \to Nov}$',
    'phi_12': r'$\phi_{Nov \to Feb}$',
    'p_oct':  r'$p_{Oct}$',  'p_nov': r'$p_{Nov}$', 'p_feb': r'$p_{Feb}$',
    'pd_oct': r'$p^*_{Oct}$','pd_nov': r'$p^*_{Nov}$','pd_feb': r'$p^*_{Feb}$',
    'N_oct':  r'$\hat{N}_{Oct}$','N_nov': r'$\hat{N}_{Nov}$',
    'N_feb':  r'$\hat{N}_{Feb}$',
}


# ── Data loading ──────────────────────────────────────────────────────────────

def load_data(filepath):
    fp = Path(filepath)
    if fp.suffix.lower() == '.csv':
        df   = pd.read_csv(fp)
        id_c = df.columns[0]
        dcol = [c for c in df.columns if c != id_c]
    else:
        df = pd.read_excel(fp,
                           sheet_name='Historial de capturauras con fe', header=1)
        df.columns = [str(c) for c in df.columns]
        df.rename(columns={df.columns[0]: 'ID'}, inplace=True)
        df['ID'] = pd.to_numeric(df['ID'], errors='coerce')
        df = df.dropna(subset=['ID'])
        id_c = 'ID'
        dcol = [c for c in df.columns if c != id_c]

    def is1(v): return str(v).strip() == '1'
    X = np.array([[1 if is1(df[c].iloc[i]) else 0 for c in dcol]
                  for i in range(len(df))], dtype=int)

    if X.shape[1] != J.sum():
        raise ValueError(f"Expected {J.sum()} occasions, got {X.shape[1]}")

    primary = np.zeros((len(X), T), dtype=int)
    within  = np.zeros((len(X), T), dtype=int)
    for t in range(T):
        s, e = STARTS[t], STARTS[t] + J[t]
        primary[:, t] = (X[:, s:e].sum(axis=1) > 0).astype(int)
        within[:, t]  = X[:, s:e].sum(axis=1)

    cap      = primary.sum(axis=1) > 0
    prim_cap = primary[cap]
    with_cap = within[cap]
    n_t = prim_cap.sum(axis=0)
    K_t = with_cap.sum(axis=0)
    JNK = J * n_t - K_t

    pats = {}
    for row in prim_cap:
        k = tuple(int(x) for x in row)
        pats[k] = pats.get(k, 0) + 1

    return dict(n_t=n_t, K_t=K_t, JNK=JNK, patterns=pats,
                N_cap=int(cap.sum()))


# ── Log-posterior ─────────────────────────────────────────────────────────────

def log_posterior(theta, data):
    phi01 = sigmoid(theta[0]); phi12 = sigmoid(theta[1])
    p     = sigmoid(theta[2:5])
    n_t, K_t, JNK, pats = data['n_t'], data['K_t'], data['JNK'], data['patterns']
    eps = 1e-12
    pd  = np.clip(1.0 - (1.0 - p)**J, eps, 1-eps)
    p   = np.clip(p, eps, 1-eps)

    # Priors
    lp  = stats.norm.logpdf(theta[0], 2.0, 2.0)
    lp += stats.norm.logpdf(theta[1], 2.0, 2.0)
    lp += stats.norm.logpdf(theta[2], -2.0, 1.5)
    lp += stats.norm.logpdf(theta[3], -2.0, 1.5)
    lp += stats.norm.logpdf(theta[4], -2.0, 1.5)

    # Within-session (zero-truncated binomial)
    lp += float(np.sum(K_t*np.log(p) + JNK*np.log(1-p) - n_t*np.log(pd)))

    # CJS between sessions
    chi2 = 1.0
    chi1 = float(np.clip(1.0 - phi12*pd[2], eps, 1.0))
    chi0 = float(np.clip((1-phi01) + phi01*(1-pd[1])*chi1, eps, 1.0))
    lp01 = np.log(phi01+eps);  lp12 = np.log(phi12+eps)
    lpd  = np.log(pd);         l1pd = np.log(1-pd+eps)
    lchi = [np.log(chi0), np.log(chi1), np.log(chi2)]

    for (y0,y1,y2), cnt in pats.items():
        if cnt == 0: continue
        ys = [y0,y1,y2]
        f  = next(t for t,y in enumerate(ys) if y==1)
        l  = max(t for t,y in enumerate(ys) if y==1)
        ll = 0.0
        for t in range(f+1, l+1):
            ll += (lp01 if t==1 else lp12)
            ll += lpd[t] if ys[t]==1 else l1pd[t]
        ll += lchi[l]
        lp += cnt*ll
    return float(lp)


# ── MAP ───────────────────────────────────────────────────────────────────────

def find_map(data, n_restarts=25, seed=42):
    rng  = np.random.default_rng(seed)
    best = None; best_lp = -np.inf

    def neg(t):
        v = log_posterior(t, data)
        return -v if np.isfinite(v) else 1e10

    for _ in range(n_restarts):
        x0 = rng.normal([2.0,0.0,-2.0,-2.0,-2.0],[1,1,.5,.5,.5])
        try:
            r = minimize(neg, x0, method='L-BFGS-B',
                         options={'maxiter':5000,'ftol':1e-14})
            if np.isfinite(r.fun) and -r.fun > best_lp:
                best_lp = -r.fun; best = r.x
        except: pass
    return best, best_lp


# ── Adaptive MH MCMC ──────────────────────────────────────────────────────────

def run_chain(theta0, data, n_iter, seed=0):
    rng  = np.random.default_rng(seed)
    D    = len(theta0)
    sd   = 2.38**2 / D
    samp = np.zeros((n_iter, D))
    lps  = np.zeros(n_iter)
    th   = theta0.copy()
    lp_c = log_posterior(th, data)
    cov  = np.eye(D)*0.05
    nacc = 0
    for i in range(n_iter):
        if i >= 500 and i % 100 == 0:
            past = samp[max(0,i-4000):i]
            if past.shape[0] > D+1:
                cov = sd*np.cov(past.T) + 1e-8*np.eye(D)
        tp = rng.multivariate_normal(th, cov)
        lp_p = log_posterior(tp, data)
        if np.log(rng.uniform()) < lp_p - lp_c:
            th=tp; lp_c=lp_p; nacc+=1
        samp[i]=th; lps[i]=lp_c
    return samp, lps, nacc/n_iter


def run_mcmc(data, n_chains=4, n_warmup=3000, n_sample=5000, seed=42):
    print("Finding MAP...")
    map_th, map_lp = find_map(data, seed=seed)
    phi01 = sigmoid(map_th[0]); phi12 = sigmoid(map_th[1])
    p     = sigmoid(map_th[2:5]); pd_ = 1-(1-p)**J
    N_    = data['n_t']/pd_
    print(f"  MAP lp={map_lp:.2f}  phi_01={phi01:.4f}  phi_12={phi12:.4f}")
    print(f"  p=[{p[0]:.3f},{p[1]:.3f},{p[2]:.3f}]  N=[{N_[0]:.0f},{N_[1]:.0f},{N_[2]:.0f}]")

    print(f"\nRunning {n_chains} chains x {n_warmup+n_sample:,} iterations...")
    rng = np.random.default_rng(seed)
    all_s, all_lp = [], []
    for c in range(n_chains):
        th0 = map_th + rng.normal(0, 0.15, 5)
        s, lp_, ar = run_chain(th0, data, n_warmup+n_sample, seed=seed+c+1)
        all_s.append(s[n_warmup:]); all_lp.append(lp_[n_warmup:])
        print(f"  Chain {c+1}: acceptance={ar:.3f}")
    return np.stack(all_s), np.stack(all_lp), map_th


# ── Transforms and diagnostics ────────────────────────────────────────────────

def transform(samps, n_t):
    phi01 = sigmoid(samps[...,0]); phi12 = sigmoid(samps[...,1])
    p     = sigmoid(samps[...,2:5])
    pd    = 1-(1-p)**J
    N     = n_t.astype(float)/pd
    return np.concatenate([phi01[...,None], phi12[...,None], p, pd, N], axis=-1)


def rhat(chains):
    M,N = chains.shape
    B = N*chains.mean(1).var(ddof=1)
    W = chains.var(1,ddof=1).mean()
    V = (N-1)/N*W + B/N
    return float(np.sqrt(V/W)) if W>0 else np.nan


def ess(chain):
    N=len(chain); x=chain-chain.mean()
    acf=np.correlate(x,x,'full')[N-1:]; acf/=acf[0]
    cut=next((i for i in range(1,len(acf)) if acf[i]+acf[i-1]<0),len(acf))
    return max(float(N/(1+2*acf[1:cut].sum())),1.0)


def summarize(samps, n_t):
    nat  = transform(samps, n_t)
    C,S,_= nat.shape
    flat = nat.reshape(C*S,-1)
    rows = []
    for j,name in enumerate(PARAM_NAMES):
        col = flat[:,j]
        r   = rhat(nat[:,:,j])
        e   = int(np.mean([ess(nat[c,:,j]) for c in range(C)]))
        rows.append(dict(
            Parameter=name, Mean=col.mean(), SD=col.std(),
            pct2_5=np.percentile(col,2.5), Median=np.median(col),
            pct97_5=np.percentile(col,97.5), Rhat=r, ESS=e
        ))
    df = pd.DataFrame(rows)
    print("\n=== POSTERIOR SUMMARY ===")
    print(df.rename(columns={'pct2_5':'2.5%','pct97_5':'97.5%'})
            .to_string(index=False, float_format='{:.3f}'.format))
    return df


# ── Figures ───────────────────────────────────────────────────────────────────

def fig_main(samps, n_t, outdir):
    nat  = transform(samps, n_t)
    flat = nat.reshape(-1, nat.shape[-1])
    fig  = plt.figure(figsize=(14,10))
    gs   = gridspec.GridSpec(3,4,hspace=0.5,wspace=0.42)

    # phi_01 and phi_12
    for i, (color, title) in enumerate([
        (PALETTE['phi'][0], r'$\phi_{Oct\to Nov}$  (spring survival / detectability)'),
        (PALETTE['phi'][1], r'$\phi_{Nov\to Feb}$  (summer surface-activity retention'),
    ]):
        ax = fig.add_subplot(gs[0, i*2:i*2+2])
        s  = flat[:,i]
        ax.hist(s, bins=60, color=color, alpha=0.75, density=True)
        med,lo,hi = np.median(s), np.percentile(s,2.5), np.percentile(s,97.5)
        ax.axvline(med,color='k',lw=1.8,label=f'Median={med:.3f}\n[{lo:.3f},{hi:.3f}]')
        ax.axvline(lo,color='k',lw=.8,ls='--'); ax.axvline(hi,color='k',lw=.8,ls='--')
        ax.set_xlabel(title,fontsize=9); ax.set_ylabel('Density',fontsize=9)
        ax.legend(fontsize=8,frameon=False)

    # p and pd
    ax_p = fig.add_subplot(gs[1,0:2])
    for t in range(3):
        s = flat[:,2+t]
        ax_p.hist(s,bins=50,color=PALETTE['p'][t],alpha=0.65,density=True,
                  label=f'{LABELS_SHORT[t]} ({np.median(s):.3f})')
    ax_p.set_xlabel(r'$p_t$ per secondary occasion',fontsize=10)
    ax_p.set_ylabel('Density',fontsize=9)
    ax_p.set_title('Per-occasion detection',fontsize=10)
    ax_p.legend(fontsize=8,frameon=False)

    ax_pd = fig.add_subplot(gs[1,2:4])
    for t in range(3):
        s = flat[:,5+t]
        ax_pd.hist(s,bins=50,color=PALETTE['pd'][t],alpha=0.65,density=True,
                   label=f'{LABELS_SHORT[t]} ({np.median(s):.3f})')
    ax_pd.set_xlabel(r"$p^*_t=1-(1-p_t)^{J_t}$",fontsize=10)
    ax_pd.set_ylabel('Density',fontsize=9)
    ax_pd.set_title('Session-level detection',fontsize=10)
    ax_pd.legend(fontsize=8,frameon=False)

    # N forest + marginals
    ax_N = fig.add_subplot(gs[2,0:2])
    for t in range(3):
        s   = flat[:,8+t]
        med = np.median(s); lo=np.percentile(s,2.5); hi=np.percentile(s,97.5)
        ax_N.errorbar(med,t,xerr=[[med-lo],[hi-med]],fmt='o',
                      color=PALETTE['N'][t],ms=9,capsize=6,lw=2,
                      label=f'{LABELS_SHORT[t]}: {med:.0f} [{lo:.0f}–{hi:.0f}]')
        ax_N.axvline(n_t[t],ymin=(t+.1)/3.5,ymax=(t+.9)/3.5,
                     color=PALETTE['N'][t],lw=1,ls=':')
    ax_N.set_yticks(range(3)); ax_N.set_yticklabels(LABELS_SHORT,fontsize=9)
    ax_N.set_xlabel(r'$\hat{N}$ (Horvitz-Thompson)',fontsize=10)
    ax_N.set_title('Population size\n[dotted = observed n]',fontsize=9)
    ax_N.legend(fontsize=7.5,frameon=False)

    ax_h = fig.add_subplot(gs[2,2:4])
    for t in range(3):
        s = flat[:,8+t]
        ax_h.hist(s,bins=60,color=PALETTE['N'][t],alpha=0.6,density=True,
                  label=LABELS_SHORT[t])
        ax_h.axvline(np.median(s),color=PALETTE['N'][t],lw=1.8)
    ax_h.set_xlabel(r'$\hat{N}$',fontsize=10); ax_h.set_ylabel('Density',fontsize=9)
    ax_h.set_title('Population marginal posteriors',fontsize=10)
    ax_h.legend(fontsize=8,frameon=False)

    fig.suptitle(f'Bayesian Robust Design — {SP}\n'
                 f'Laguna Azul, Somuncurá Plateau — austral spring–summer 2019–2020',
                 fontsize=11)
    p = outdir/'fig_cmr_posterior.png'; fig.savefig(p,dpi=200); plt.close(fig)
    return p


def fig_trace(samps, n_t, outdir):
    C,S,_ = samps.shape
    nat   = transform(samps, n_t)
    lbls  = [r'$\mathrm{logit}(\phi_{01})$', r'$\mathrm{logit}(\phi_{12})$',
             r'$\mathrm{logit}(p_{Oct})$', r'$\mathrm{logit}(p_{Nov})$',
             r'$\mathrm{logit}(p_{Feb})$']
    cols  = ['#1b7837','#762a83','#2166ac','#d6604d','#e08214']
    fig,axes = plt.subplots(5,2,figsize=(12,13))
    for j in range(5):
        ax_t,ax_d = axes[j]
        for c in range(C):
            ax_t.plot(samps[c,:,j],color=cols[c],alpha=.45,lw=.4,
                      label=f'Chain {c+1}' if j==0 else '')
        ax_d.hist(nat[:,:,j].reshape(-1),bins=60,color=cols[j],alpha=.7,density=True)
        ax_t.set_ylabel(lbls[j],fontsize=9)
        ax_d.set_xlabel(list(PARAM_LABELS.values())[j],fontsize=9)
        if j==0:
            ax_t.legend(fontsize=7,frameon=False)
            ax_t.set_title('Trace (logit scale)',fontsize=10)
            ax_d.set_title('Marginal posterior (natural scale)',fontsize=10)
    fig.suptitle(f'MCMC diagnostics — {SP}',fontsize=11)
    fig.tight_layout()
    p = outdir/'fig_cmr_trace.png'; fig.savefig(p,dpi=200); plt.close(fig)
    return p


def fig_prior_post(samps, n_t, outdir):
    nat  = transform(samps, n_t)
    flat = nat.reshape(-1, nat.shape[-1])
    priors = [(2.0,2.0),(2.0,2.0),(-2.0,1.5),(-2.0,1.5),(-2.0,1.5)]
    xlims  = [(0,1),(0,1),(0,.6),(0,.6),(0,.6)]
    cols   = ['#1b7837','#762a83','#2166ac','#d6604d','#e08214']
    fig,axes = plt.subplots(1,5,figsize=(16,4))
    for j,(ax,col,(mu,sg),xl) in enumerate(zip(axes,cols,priors,xlims)):
        x  = np.linspace(xl[0]+1e-5, xl[1]-1e-5, 400)
        lx = logit(x)
        lp = stats.norm.logpdf(lx,mu,sg) - np.log(x*(1-x))
        lp -= lp.max(); pr = np.exp(lp); pr /= np.trapezoid(pr,x)
        ax.plot(x,pr,'k--',lw=1.5,label='Prior',alpha=.8)
        ax.hist(flat[:,j],bins=60,density=True,alpha=.6,color=col,label='Posterior')
        ax.set_xlabel(list(PARAM_LABELS.values())[j],fontsize=9)
        ax.legend(fontsize=8,frameon=False)
    fig.suptitle(f'Prior vs. posterior — {SP}',fontsize=11)
    fig.tight_layout()
    p = outdir/'fig_cmr_prior_posterior.png'; fig.savefig(p,dpi=200); plt.close(fig)
    return p


def fig_N_time(samps, n_t, outdir):
    flat = transform(samps, n_t).reshape(-1, len(PARAM_NAMES))
    Ns   = flat[:,8:11]
    med  = np.median(Ns,0); lo=np.percentile(Ns,2.5,0); hi=np.percentile(Ns,97.5,0)
    fig,ax = plt.subplots(figsize=(7,4.5))
    x = np.arange(3)
    ax.fill_between(x,lo,hi,alpha=.25,color='#525252',label='95% CrI')
    ax.plot(x,med,'o-',color='#1b7837',lw=2,ms=9,label='Median')
    ax.plot(x,n_t,'s--',color='#d6604d',lw=1.5,ms=8,label='Observed n')
    ax.set_xticks(x); ax.set_xticklabels(LABELS_SHORT,fontsize=10)
    ax.set_ylabel(r'$\hat{N}$ estimated population',fontsize=11)
    ax.set_title(f'Population size — austral spring-summer\n{SP}',fontsize=10)
    ax.legend(fontsize=9,frameon=False)
    fig.tight_layout()
    p = outdir/'fig_cmr_N_time.png'; fig.savefig(p,dpi=200); plt.close(fig)
    return p


# ── Main ──────────────────────────────────────────────────────────────────────

def run_cmr(data_path, outdir, n_chains=4, n_warmup=3000, n_sample=5000, seed=42):
    outdir = Path(outdir); outdir.mkdir(parents=True, exist_ok=True)

    print(f"Loading: {data_path}")
    data = load_data(data_path)
    n_t  = data['n_t']

    print(f"\n=== DATA ===")
    for t in range(T):
        print(f"  {LABELS_SHORT[t]}: n={n_t[t]}, J={J[t]}, "
              f"K={data['K_t'][t]}, naive_p={data['K_t'][t]/(J[t]*n_t[t]):.3f}")
    print(f"  Total captured: {data['N_cap']}")
    print("  Histories:", {str(k):v for k,v in data['patterns'].items()})

    samps, log_posts, map_th = run_mcmc(data, n_chains, n_warmup, n_sample, seed)
    summary = summarize(samps, n_t)
    summary.to_csv(outdir/'cmr_posterior_summary.csv', index=False)

    print("\n=== CONVERGENCE ===")
    nat = transform(samps, n_t)
    for j,name in enumerate(PARAM_NAMES):
        r = rhat(nat[:,:,j])
        e = int(np.mean([ess(nat[c,:,j]) for c in range(n_chains)]))
        print(f"  {name:<12}: R-hat={r:.4f} {'✓' if r<1.05 else '⚠'}  ESS={e:5d}")

    print("\n=== vs rcapture R (MLE) ===")
    r_ref = dict(phi_01=(0.7927,0.2363),phi_12=(0.1653,0.0509),
                 p_oct=(0.1435,0.0653),p_nov=(0.2001,0.0442),p_feb=(0.4509,0.0666),
                 N_oct=(473.8,222.1),N_nov=(723.5,165.8),N_feb=(193.0,32.4))
    for _,row in summary.iterrows():
        nm = row['Parameter']
        if nm in r_ref:
            rv,rse = r_ref[nm]
            print(f"  {nm:<10}: Bayes={row['Median']:.3f} [{row['pct2_5']:.3f},{row['pct97_5']:.3f}]"
                  f"  R_MLE={rv:.3f} ±{rse:.3f}")

    print("\nFigures...")
    for f in [fig_main(samps,n_t,outdir), fig_trace(samps,n_t,outdir),
              fig_prior_post(samps,n_t,outdir), fig_N_time(samps,n_t,outdir)]:
        print(f"  {f.name}")

    return dict(data=data, samps=samps, summary=summary, map_th=map_th)


HERE = Path(__file__).resolve().parent
REPO_DIR = HERE.parent
DEFAULT_DATA = REPO_DIR / 'data' / 'capture_history.csv'
DEFAULT_OUTDIR = REPO_DIR / 'outputs'


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--data',    default=str(DEFAULT_DATA))
    ap.add_argument('--outdir',  default=str(DEFAULT_OUTDIR))
    ap.add_argument('--chains',  type=int, default=4)
    ap.add_argument('--warmup',  type=int, default=3000)
    ap.add_argument('--samples', type=int, default=5000)
    ap.add_argument('--seed',    type=int, default=42)
    args = ap.parse_args()
    run_cmr(args.data, args.outdir,
            args.chains, args.warmup, args.samples, args.seed)
