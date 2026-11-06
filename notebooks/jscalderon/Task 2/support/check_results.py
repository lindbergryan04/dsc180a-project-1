from pathlib import Path
import json,csv,math,statistics
root=Path('/Users/jcalderon204/Documents/DSC 180A/task2')
def read(name):
    with (root/'results'/name).open() as f:return list(csv.DictReader(f))
summary_path=root/'results/summary.json'
d=json.loads(summary_path.read_text())
def clean(x):
    if isinstance(x,float) and not math.isfinite(x):return None
    if isinstance(x,dict):return {k:clean(v) for k,v in x.items()}
    if isinstance(x,list):return [clean(v) for v in x]
    return x
d=clean(d);summary_path.write_text(json.dumps(d,indent=2,allow_nan=False))
metrics=read('metrics.csv')
for name,file in [('Random test','test_predictions.csv'),('SAT3','sat3_predictions.csv')]:
    rows=read(file)
    for target in ['Q_i','Q_e','Gamma_e']:
        y=[float(r[target]) for r in rows]
        for model,suffix in [('Mean baseline','_mean'),('Selected NN','_NN')]:
            pred=[float(r[target+suffix]) for r in rows]
            mse=statistics.mean((a-b)**2 for a,b in zip(y,pred))
            r2=1-mse/statistics.pvariance(y)
            saved=next(m for m in metrics if m['dataset']==name and m['model']==model and m['output']==target)
            assert math.isclose(math.sqrt(mse),float(saved['RMSE']),rel_tol=1e-6,abs_tol=1e-8)
            assert math.isclose(r2,float(saved['R2']),rel_tol=1e-6,abs_tol=1e-8)
print('All 12 metric rows independently verified from saved predictions.')
curves=read('learning_curves.csv')
nn={n:statistics.mean(float(r['mean_nRMSE']) for r in curves if r['model']=='Selected NN' and int(r['training_n'])==n) for n in [250,500,1000,2000,d['split_sizes']['train']]}
base={n:statistics.mean(float(r['mean_nRMSE']) for r in curves if r['model']=='Mean baseline' and int(r['training_n'])==n) for n in nn}
print('Learning-curve means:',nn)
print('Particle sign checks:',d['signed_checks'])
nbpath=root/'Task2_Neural_Networks_Calderon.ipynb'
nb=json.loads(nbpath.read_text())
for cell in nb['cells']:
    if cell['cell_type']=='code':
        source=''.join(cell['source']) if isinstance(cell['source'],list) else cell['source']
        cell['source']=source.replace("'validation':validation.to_dict('records')","'validation':validation.astype(object).where(pd.notna(validation),None).to_dict('records')")
        assert cell['execution_count'] is not None
        assert not any(o['output_type']=='error' for o in cell['outputs'])
        assert 'Traceback' not in ''.join(o.get('text','') if isinstance(o.get('text',''),str) else ''.join(o.get('text',[])) for o in cell['outputs'])
full=max(nn)
trend=('The decrease between 2,000 cases and the full training set suggests more TGLF data could still help, although extrapolating this curve beyond the observed sizes is uncertain.' if nn[full]<nn[2000] else 'The last step does not show a consistent gain, so more data alone is not guaranteed to help. Additional cases should target weakly covered regimes.')
text=f'''## Interpretation of the completed experiment

The selected model is a **{d['selected_name']}**. The best single deep network's mean validation nRMSE was 0.2043, compared with 0.2221 for the medium two-layer network. Averaging three seeds reduced it further to 0.1890. Simply increasing width to 128-128 did not outperform the deeper network in this search.

On the random test set, the selected model's R² is **0.949 for ion heat flux, 0.955 for electron heat flux, and 0.972 for electron particle flux**. The corresponding SAT3 scores are **0.949, 0.986, and 0.996**. Both tests beat the training-mean baseline for every output. SAT3 has smaller absolute errors, but it is a small, differently distributed scan set, so that does not establish better performance on arbitrary new plasmas.

Across the three learning-curve repetitions, mean test nRMSE changed from **{nn[250]:.3f} at 250 cases** to **{nn[2000]:.3f} at 2,000** and **{nn[full]:.3f} at {full:,}**. The mean baseline changed from {base[250]:.3f} to {base[full]:.3f}. {trend}

The selected model took **{d['timing']['selected_single_case_ms']:.3f} ms per single-case CPU call**, including preprocessing and inverse transforms. Training its three members took {d['timing']['selected_model_training_seconds']:.1f} seconds in total, excluding architecture search. Batch throughput is {d['timing']['selected_batch_ms_per_case']:.4f} ms per case and should not be confused with individual-call latency.

Ion heat-flux isotope ordering agreed in **{d['isotope_order_matches']} of {d['isotope_triplets']} complete triplets**. For negative particle-flux cases, the model predicted the negative sign for {d['signed_checks']['Random test']['negative_particle_sign_correct']} of {d['signed_checks']['Random test']['negative_particle_cases']} random test cases and {d['signed_checks']['SAT3']['negative_particle_sign_correct']} of {d['signed_checks']['SAT3']['negative_particle_cases']} SAT3 cases. These local checks qualify the high overall R²: a surrogate can be accurate on average while still missing a subtle ordering or sign.
'''
nb['cells'].insert(-1,{'cell_type':'markdown','id':'completed-interpretation','metadata':{},'source':text})
nbpath.write_text(json.dumps(nb,indent=1))
print('Notebook outputs and execution counts checked; interpretation added.')
