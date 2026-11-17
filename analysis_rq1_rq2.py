"""analysis_rq1_rq2.py - RQ1 (geography) and RQ2 (advertising organisations).

Inputs:  output/ba_validated.csv, output/ba_skills_long.csv   (from prep.py)
         data/bls_labour_force_jan2024.csv                    (BLS, January 2024)
         data/advertiser_types.csv                            (manual coding, top-50 advertisers)
Outputs: output/rq1_states.csv, output/rq2_all_advertisers.csv,
         output/rq2_top50_advertisers.csv, output/rq2_type_skill_compare.csv,
         output/results_rq1_rq2.json, and figures fig2_rq1.png, fig3_rq2.png, fig4_rq2.png

Run:  python analysis_rq1_rq2.py
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.multitest import multipletests

DATA, OUT = Path('data'), Path('output')
LATICRETE = 'LATICRETE International'
BLUE, ORANGE, GREEN, GOLD, GREY = '#2b7bd6', '#ea6a35', '#1db07a', '#eda100', '#e6e6e1'
res = {}

val = pd.read_csv(OUT / 'ba_validated.csv')
sk = pd.read_csv(OUT / 'ba_skills_long.csv')
lf = pd.read_csv(DATA / 'bls_labour_force_jan2024.csv')


# ============================== RQ1 ==============================
def state_table(df):
    g = (df[df.state_clean.notna()].groupby('state_clean').size()
         .rename('ads').reset_index().rename(columns={'state_clean': 'state'}))
    g = lf.merge(g, on='state', how='left').fillna({'ads': 0})   # keeps states with 0 ads
    g['ads'] = g.ads.astype(int)
    g['share_pct'] = g.ads / g.ads.sum() * 100
    g['ads_per_100k_lf'] = g.ads / g.labour_force * 1e5
    g['location_quotient'] = (g.ads / g.ads.sum()) / (g.labour_force / g.labour_force.sum())
    return g


g = state_table(val)
nat_rate = g.ads.sum() / g.labour_force.sum() * 1e5
res['rq1_national_rate_per_100k'] = round(nat_rate, 2)               # report: 2.22

rho, rho_p = stats.spearmanr(g.ads, g.labour_force)
expected = g.labour_force / g.labour_force.sum() * g.ads.sum()
chi = stats.chisquare(g.ads, expected)
res['rq1_spearman_rho'] = round(rho, 2)                               # report: 0.91
res['rq1_chi_square'] = {'chi2': round(chi.statistic, 1), 'df': len(g) - 1,
                         'p': float(chi.pvalue)}                      # report: 779.7, df 50


def poisson_two_sided(k, mu):
    return min(1.0, 2 * min(stats.poisson.cdf(k, mu), stats.poisson.sf(k - 1, mu)))


g['expected_ads'] = g.labour_force * nat_rate / 1e5
g['poisson_p'] = [poisson_two_sided(k, mu) for k, mu in zip(g.ads, g.expected_ads)]
g['p_holm'] = multipletests(g.poisson_p, method='holm')[1]
g['direction'] = np.where(g.p_holm >= 0.05, 'not significant',
                          np.where(g.location_quotient > 1, 'over', 'under'))

# sensitivity: without LATICRETE's multi-location template ads
g_ex = state_table(val[val.company != LATICRETE])
g = g.merge(g_ex[['state', 'location_quotient']].rename(
    columns={'location_quotient': 'lq_excl_laticrete'}), on='state')
res['rq1_lq_rank_rho_excl_laticrete'] = round(
    stats.spearmanr(g.location_quotient, g.lq_excl_laticrete)[0], 2)  # report: 0.99

# commuting check: DC pooled with Maryland and Virginia (report: 330 ads, 181.3 expected, LQ 1.82)
dmv = g[g.state.isin(['District of Columbia', 'Maryland', 'Virginia'])]
dmv_exp = dmv.labour_force.sum() * nat_rate / 1e5
res['rq1_pooled_dc_md_va'] = {
    'ads': int(dmv.ads.sum()), 'expected': round(dmv_exp, 1),
    'lq': round(dmv.ads.sum() / dmv_exp, 2),
    'poisson_p': float(poisson_two_sided(int(dmv.ads.sum()), dmv_exp))}

g['rank_raw'] = g.ads.rank(ascending=False, method='min').astype(int)
g['rank_rate'] = g.ads_per_100k_lf.rank(ascending=False, method='min').astype(int)
g = g.sort_values('ads', ascending=False)
g.round(4).to_csv(OUT / 'rq1_states.csv', index=False)
res['rq1_top5_share_pct'] = round(g.ads.head(5).sum() / g.ads.sum() * 100, 1)   # 38.3
res['rq1_top10_share_pct'] = round(g.ads.head(10).sum() / g.ads.sum() * 100, 1)  # 60.0
res['rq1_over'] = g[g.direction == 'over'].state.tolist()
res['rq1_under'] = g[g.direction == 'under'].state.tolist()

# Figure 2: top-12 raw counts and significant LQs
fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 5.4))
top = g.head(12).iloc[::-1]
a1.barh(top.state, top.ads, color=BLUE)
a1.bar_label(a1.containers[0], padding=3, fontsize=8)
a1.set_title('Raw volume (top 12 states)', loc='left', fontsize=11)
a1.set_xlabel('Explicit BA advertisements')
sig = g[g.direction != 'not significant'].sort_values('location_quotient')
sig = sig.iloc[::-1]
a2.barh(sig.state, sig.location_quotient - 1, left=1,
        color=[BLUE if d == 'over' else ORANGE for d in sig.direction])
for y, v in enumerate(sig.location_quotient):
    a2.text(max(v, 1) + 0.08, y, f'{v:.2f}', va='center', fontsize=8)
a2.axvline(1, color='black', lw=0.8)
a2.barh([], [], color=BLUE, label='Above labour-force share')
a2.barh([], [], color=ORANGE, label='Below labour-force share')
a2.legend(frameon=False, loc='lower right', fontsize=8)
a2.set_title('Over-/under-represented states (Holm p < .05)', loc='left', fontsize=11)
a2.set_xlabel('Location quotient (1 = in line with state labour-force share)')
for ax in (a1, a2):
    ax.spines[['top', 'right']].set_visible(False)
plt.tight_layout(); plt.savefig(OUT / 'fig2_rq1.png', dpi=200); plt.close()


# ============================== RQ2 ==============================
co = (val.groupby('company')
      .agg(ads=('job_link', 'size'), titles=('job_title', 'nunique'),
           locations=('job_location', 'nunique'), states=('state_clean', 'nunique'))
      .sort_values('ads', ascending=False).reset_index())
co['share_pct'] = co.ads / co.ads.sum() * 100
co['cum_pct'] = co.share_pct.cumsum()
co.to_csv(OUT / 'rq2_all_advertisers.csv', index=False)

p = co.ads / co.ads.sum()
hhi = float((p ** 2).sum() * 10000)
x = np.sort(co.ads.values); n = len(x)
gini = float((2 * np.arange(1, n + 1) - n - 1).dot(x) / (n * x.sum()))
uniq = val.drop_duplicates(['company', 'job_title']).company.value_counts()
hhi_uniq = float(((uniq / uniq.sum()) ** 2).sum() * 10000)
res['rq2'] = {
    'advertisers': int(n),                                             # 1,356
    'single_ad_advertisers': int((co.ads == 1).sum()),                 # 914
    'top_shares_pct': {k: round(co.share_pct.head(k).sum(), 1) for k in (5, 10, 20, 50)},
    'hhi': round(hhi), 'hhi_equivalent_n': round(10000 / hhi),         # 116, ~86
    'gini': round(gini, 2),                                            # 0.57
    'unique_company_title_pairs': int(uniq.sum()),                     # 2,672
    'hhi_unique_pairs': round(hhi_uniq),                               # 39
}

# advertiser types for the top 50 (manual coding file)
types = pd.read_csv(DATA / 'advertiser_types.csv')
top50 = co.head(50).merge(types, on='company', how='left')
missing = top50[top50.advertiser_type.isna()].company.tolist()
if missing:
    print('WARNING: add these top-50 advertisers to data/advertiser_types.csv:', missing)
top50.to_csv(OUT / 'rq2_top50_advertisers.csv', index=False)
by_type = top50.groupby('advertiser_type').agg(advertisers=('company', 'size'), ads=('ads', 'sum'))
by_type['share_pct'] = by_type.ads / by_type.ads.sum() * 100
res['rq2_top50_by_type'] = by_type.round(1).reset_index().to_dict('records')

# skill mix: intermediaries vs employers/consultancies (top 50, LATICRETE excluded)
INTER = {'Staffing / recruitment', 'Job board / aggregator'}
t50 = val.merge(top50[['company', 'advertiser_type']], on='company')
t50 = t50[(t50.company != LATICRETE) & t50.has_skills & t50.advertiser_type.notna()]
t50['group'] = np.where(t50.advertiser_type.isin(INTER), 'Intermediary', 'Employer / consultancy')
SKILLS = ['sql', 'data analysis', 'business analysis', 'requirements gathering', 'agile',
          'project management', 'communication', 'jira', 'user stories', 'tableau']
has = sk[sk.job_link.isin(t50.job_link)].drop_duplicates()
rows = []
for s in SKILLS:
    links = set(has.loc[has.skill_clean == s, 'job_link'])
    t50['has_s'] = t50.job_link.isin(links)
    tab = pd.crosstab(t50.group, t50.has_s).reindex(columns=[False, True], fill_value=0)
    chi2, pv, _, _ = stats.chi2_contingency(tab)
    pct = t50.groupby('group').has_s.mean() * 100
    rows.append({'skill': s, 'intermediary_pct': pct.get('Intermediary', np.nan),
                 'employer_pct': pct.get('Employer / consultancy', np.nan),
                 'chi2': chi2, 'p': pv})
cmp_ = pd.DataFrame(rows)
cmp_['p_holm'] = multipletests(cmp_.p, method='holm')[1]
cmp_.round(4).to_csv(OUT / 'rq2_type_skill_compare.csv', index=False)
res['rq2_group_n'] = t50.group.value_counts().to_dict()               # 884 / 250

# Figure 3: concentration curve + advertiser types
fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 5))
xs = np.arange(1, n + 1) / n * 100
a1.plot(xs, co.cum_pct, color=BLUE, lw=2, label='Observed')
a1.plot([0, 100], [0, 100], color='#444444', ls='--', lw=0.8, label='Perfect equality')
for k in (5, 20, 50):
    a1.scatter(k / n * 100, co.cum_pct.iloc[k - 1], color=BLUE, zorder=3)
    a1.annotate(f'Top {k}: {co.share_pct.head(k).sum():.1f}%', (k / n * 100, co.cum_pct.iloc[k - 1]),
                xytext=(8, -4), textcoords='offset points', fontsize=8)
a1.set(xlabel='Cumulative % of advertisers (largest first)', ylabel='Cumulative % of BA ads')
a1.set_title(f'Concentration curve (Gini {gini:.2f}, HHI {hhi:.0f})', loc='left', fontsize=11)
a1.legend(frameon=False, loc='lower right', fontsize=8)
bt = by_type.sort_values('share_pct')
type_col = {'Staffing / recruitment': BLUE, 'Direct employer': ORANGE,
            'Job board / aggregator': GREEN, 'IT services & consulting': GOLD}
labels = [f'{t}\n({int(a)} advertisers)' for t, a in zip(bt.index, bt.advertisers)]
a2.barh(labels, bt.share_pct, color=[type_col.get(t, GREY) for t in bt.index])
a2.bar_label(a2.containers[0], fmt='%.1f%%', padding=3, fontsize=8)
a2.set_xlabel(f"% of top-50 advertisers' ads (n = {int(bt.ads.sum()):,})")
a2.set_title('Who places the ads? (top 50 advertisers)', loc='left', fontsize=11)
for ax in (a1, a2):
    ax.spines[['top', 'right']].set_visible(False)
plt.tight_layout(); plt.savefig(OUT / 'fig3_rq2.png', dpi=200); plt.close()

# Figure 4: dumbbell of skill prevalence by advertiser group
c = cmp_.dropna(subset=['intermediary_pct', 'employer_pct']).copy()
c['skill'] = c.skill.str.title().str.replace('Sql', 'SQL')
c = c.sort_values('employer_pct')
gn = t50.group.value_counts()
if c.empty:
    print('WARNING: Figure 4 skipped - one advertiser group is empty '
          '(check data/advertiser_types.csv covers the top 50).')
fig, ax = plt.subplots(figsize=(8, 4.6))
ax.hlines(c.skill, c.intermediary_pct, c.employer_pct, color=GREY, lw=3)
ax.scatter(c.intermediary_pct, c.skill, color=BLUE, s=50, zorder=3,
           label=f"Intermediary (staffing / job board), n = {gn.get('Intermediary', 0):,}")
ax.scatter(c.employer_pct, c.skill, color=ORANGE, s=50, zorder=3,
           label=f"Employer / consultancy (excl. LATICRETE), n = {gn.get('Employer / consultancy', 0):,}")
for _, r in c.iterrows():
    if r.p_holm < 0.05:
        ax.text(max(r.intermediary_pct, r.employer_pct) + 1.5, r.skill, '*', va='center', fontsize=12)
ax.set_xlabel('% of ads listing skill (* = Holm-adjusted p < .05)')
ax.legend(frameon=False, loc='lower right', fontsize=8)
ax.spines[['top', 'right']].set_visible(False)
plt.tight_layout()
if not c.empty:
    plt.savefig(OUT / 'fig4_rq2.png', dpi=200)
plt.close()

json.dump(res, open(OUT / 'results_rq1_rq2.json', 'w'), indent=2, default=str)
print(json.dumps(res, indent=2, default=str))

# ===================== check against the report =====================
# Values printed in the Group 2 report (Sections 4.1-4.2, Appendix C.1-C.2).
REPORT = {
    'RQ1 national rate per 100k': (res['rq1_national_rate_per_100k'], 2.22),
    'RQ1 Spearman rho': (res['rq1_spearman_rho'], 0.91),
    'RQ1 chi-square': (res['rq1_chi_square']['chi2'], 779.7),
    'RQ1 top-5 share %': (res['rq1_top5_share_pct'], 38.3),
    'RQ1 top-10 share %': (res['rq1_top10_share_pct'], 60.0),
    'RQ1 LQ rank rho excl. LATICRETE': (res['rq1_lq_rank_rho_excl_laticrete'], 0.99),
    'RQ1 pooled DC+MD+VA ads': (res['rq1_pooled_dc_md_va']['ads'], 330),
    'RQ2 advertisers': (res['rq2']['advertisers'], 1356),
    'RQ2 single-ad advertisers': (res['rq2']['single_ad_advertisers'], 914),
    'RQ2 top-50 share %': (res['rq2']['top_shares_pct'][50], 40.0),
    'RQ2 HHI': (res['rq2']['hhi'], 116),
    'RQ2 Gini': (res['rq2']['gini'], 0.57),
    'RQ2 HHI unique pairs': (res['rq2']['hhi_unique_pairs'], 39),
    'RQ2 intermediary ads with skills': (res['rq2_group_n'].get('Intermediary', 0), 884),
    'RQ2 employer/consultancy ads with skills':
        (res['rq2_group_n'].get('Employer / consultancy', 0), 250),
}
print('\nCheck against report:')
bad = 0
for k, (got, want) in REPORT.items():
    ok = abs(float(got) - float(want)) < 1e-9 or round(float(got), 2) == round(float(want), 2)
    bad += not ok
    print(f"  {'OK ' if ok else 'DIFF'} {k}: script {got} | report {want}")
print('All report values reproduced.' if not bad else
      f'{bad} value(s) differ - see README "Checking the numbers".')

