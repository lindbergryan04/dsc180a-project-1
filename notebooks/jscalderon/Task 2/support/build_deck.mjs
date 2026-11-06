import fs from 'node:fs/promises';
import {Presentation,PresentationFile,FileBlob} from '@oai/artifact-tool';
import {applyPresentationChartFont,finalizePresentation} from '/Users/jcalderon204/.codex/plugins/cache/openai-primary-runtime/presentations/26.1025.11240/skills/presentations/container_tools/artifact_tool_utils.mjs';
process.env.RUNTIME_NODE_MODULES='/Users/jcalderon204/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules';
const root='/Users/jcalderon204/Documents/DSC 180A/task2';
const skill='/Users/jcalderon204/.codex/plugins/cache/openai-primary-runtime/presentations/26.1025.11240/skills/presentations';
const d=JSON.parse(await fs.readFile(`${root}/results/summary.json`,'utf8'));
const lc=(await fs.readFile(`${root}/results/learning_curves.csv`,'utf8')).trim().split('\n').slice(1).map(l=>{const a=l.split(',');return {repeat:+a[0],n:+a[1],model:a[2],value:+a[3]};});
const p=Presentation.create({slideSize:{width:1280,height:720}});
const font='Helvetica Neue',ink='#18364B',muted='#526E80',blue='#285F82',teal='#287F79',orange='#B57126';
function text(s,t,x,y,w,h,size=24,bold=false,color=ink){
 const a=s.shapes.add({geometry:'textbox',position:{left:x,top:y,width:w,height:h},fill:'none',line:{fill:'none',width:0}});
 a.text=t;a.text.style={fontFamily:font,typeface:font,fontSize:size,color,bold,verticalAlignment:'middle'};a.text.insets=0;return a;
}
function slide(title,subtitle){const s=p.slides.add();s.background.fill='#FFFFFF';text(s,title,70,48,1140,65,44,true);if(subtitle)text(s,subtitle,72,123,1135,55,23,false,muted);return s;}
function foot(s,t){text(s,t,72,660,1136,35,17,false,muted);}
function table(s,values,x,y,width,height,widths){
 const t=s.tables.add({rows:values.length,columns:values[0].length,left:x,top:y,width,height,columnWidths:widths,values});
 t.borders.assign({fill:'#DBE3E7',width:1});
 for(let r=0;r<values.length;r++) for(let c=0;c<values[0].length;c++){
  const a=t.getCell(r,c);a.fill=r===0?ink:(r%2?'#F1F5F6':'#FFFFFF');
  a.text.style={fontFamily:font,typeface:font,fontSize:r===0?20:21,color:r===0?'#FFFFFF':ink,bold:r===0,alignment:c===0?'left':'center',verticalAlignment:'middle'};
 }
 t.cells.block({row:0,column:0,rowCount:values.length,columnCount:values[0].length}).assign({margins:{left:14,right:14,top:6,bottom:6}});
 return t;
}
function chart(s,config){config.series=config.series.map(r=>({...r,xValues:r.xValues.map(v=>Number(v.toFixed(6))),values:r.values.map(v=>Number(v.toFixed(6)))}));const c=s.charts.add('scatter',{...config,chartFill:'#FFFFFF',chartLine:{fill:'none',width:0},plotAreaFill:'#FFFFFF',scatterOptions:{style:'lineWithMarkers'},hasLegend:true,legend:{position:'top',textStyle:{fontSize:18,fill:ink}}});applyPresentationChartFont(c,{fontFamily:font});return c;}
const axis=(title,min,max,step,fmt='0.0')=>({title:{text:title,textStyle:{fontSize:18,fill:ink}},textStyle:{fontSize:17,fill:ink},min,max,majorUnit:step,numberFormatCode:fmt,majorGridlines:{fill:'#E1E8EC',width:1},line:{fill:'#AABAC4',width:1}});
const metric=(set,target,model='Selected NN')=>d.metrics.find(x=>x.dataset===set&&x.output===target&&x.model===model);
const sources='Sources: Task2_TGLF_surrogate.docx; supplied scan_db.tar and SAT3_dataset.tar; executed Task2_Neural_Networks_Calderon.ipynb. Results are TGLF SAT2 predictions in gyro-Bohm units, not CGYRO validation. Case assignments, exclusions, full metrics and hardware details are in task2/results/. The reference decks motivated the experiment but do not provide directly comparable scores without their code and exact splits.';
{
 const s=slide('Neural network surrogates for TGLF','Calderon   DSC 180A Task 2');
 text(s,'Building on the group’s model comparison',72,215,1110,45,30,true);
 text(s,'Six network variants test size, depth and target scaling.\nA three-seed ensemble tests whether averaging helps.',72,280,1100,94,28);
 text(s,'5,000 random cases',72,420,440,40,28,true);
 text(s,`4 heat-flux exclusions\n${d.split_sizes.train.toLocaleString()} train / ${d.split_sizes.validation} validation / ${d.split_sizes.test} test`,72,470,530,84,25,false,muted);
 text(s,'11 input features',675,420,500,40,28,true);
 text(s,'Predict Qᵢ, Qₑ and signed Γₑ\nOutside test on 43 SAT3 cases',675,470,510,84,25,false,muted);
 foot(s,'Fixed split seed 42. Training-only scaling. Mean predictor retained as the required baseline.');
 s.speakerNotes.text=`The groupmates found neural networks strongest among the families they tested. My contribution compares neural-network configurations rather than redoing the tree/regression comparison. The candidates are 32-32, 64-64, 128-128 and 64-64-64 hidden layers, plus two 64-64 target-transform alternatives. Four cases with nonpositive ion heat flux are excluded after splitting. Particle flux stays signed. ${sources}`;
}
{
 const s=slide('Validation selects the network','Lower mean nRMSE is better. Selection uses validation data only.');
 const vals=[['Candidate','Parameters','Mean nRMSE','Train seconds'],...d.validation.map(r=>[r.model,Math.round(r.parameters).toLocaleString(),r.mean_validation_nRMSE.toFixed(4),r.training_seconds.toFixed(1)])];
 table(s,vals,72,205,1136,350,[476,210,220,230]);
 text(s,`Selected: ${d.selected_name}`,72,580,1136,48,27,true,teal);
 foot(s,'nRMSE = RMSE / training standard deviation, averaged over the three outputs in original units.');
 s.speakerNotes.text=`All networks use GELU, a linear output head, AdamW at 0.001 with weight decay 0.0001, batch size 256, at most 600 epochs, and validation early stopping with patience 70. Log candidates predict log10 heat flux and signed raw particle flux. Signed-asinh additionally transforms Gamma_e using a training-only scale. Raw targets use standardization alone. Every run has a fixed seed. Training seconds for the ensemble sum its member runs and exclude architecture search. Best checkpoint selection also uses mean validation nRMSE in original units. ${sources}`;
}
{
 const s=slide('Accuracy on unseen cases','Selected network predictions, evaluated after freezing the model choice');
 const labels={Q_i:'Qᵢ',Q_e:'Qₑ',Gamma_e:'Γₑ'};
 table(s,[['Set and output','NN RMSE','NN R²','Mean RMSE','Mean R²'],...['Random test','SAT3'].flatMap(set=>['Q_i','Q_e','Gamma_e'].map(target=>{
  const a=metric(set,target),b=metric(set,target,'Mean baseline');return [`${set}  ${labels[target]}`,a.RMSE.toFixed(3),a.R2.toFixed(3),b.RMSE.toFixed(3),b.R2.toFixed(3)];
 }))],72,205,1136,330,[356,195,195,195,195]);
 const testR=d.metrics.filter(x=>x.dataset==='Random test'&&x.model==='Selected NN').map(x=>x.R2);
 const satR=d.metrics.filter(x=>x.dataset==='SAT3'&&x.model==='Selected NN').map(x=>x.R2);
 text(s,`Random test R² ${Math.min(...testR).toFixed(3)}–${Math.max(...testR).toFixed(3)}\nSAT3 R² ${Math.min(...satR).toFixed(3)}–${Math.max(...satR).toFixed(3)}`,72,563,1100,75,27,true,teal);
 foot(s,`RMSE in gyro-Bohm units. ${d.split_sizes.test} random test cases and 43 SAT3 cases. Baseline = training mean.`);
 s.speakerNotes.text=`Each output has its own RMSE and R². R² denominators use the variance in the respective evaluation set, so the outside-test R² alone is not a like-for-like measure of difficulty. The notebook also reports nRMSE with the same training standard deviations for both sets and parity plots for both baseline and selected network. No test or SAT3 result was used for architecture selection. ${sources}`;
}
{
 const s=slide('Learning curve and CPU inference','Fixed selected architecture, with three repetitions at each training size');
 const sizes=[...new Set(lc.map(r=>r.n))].sort((a,b)=>a-b);
 const avg=(model,n)=>{const a=lc.filter(r=>r.model===model&&r.n===n);return a.reduce((v,r)=>v+r.value,0)/a.length;};
 chart(s,{position:{left:60,top:207,width:745,height:365},series:[['Selected NN',blue],['Mean baseline',orange]].map(([name,color])=>({name,xValues:sizes,values:sizes.map(n=>avg(name,n)),line:{fill:color,width:3},marker:{symbol:'circle',size:8,fill:color}})),xAxis:axis('Training cases',0,3500,1000,'0'),yAxis:axis('Mean test nRMSE',0,1.2,.3)});
 text(s,'Single-case latency',858,229,350,42,27,true);
 text(s,`${d.timing.selected_single_case_ms.toFixed(3)} ms`,858,286,350,58,40,false,teal);
 text(s,'Batch throughput',858,379,350,42,27,true);
 text(s,`${d.timing.selected_batch_ms_per_case.toFixed(4)} ms / case`,858,437,350,50,29,false,teal);
 text(s,`NN mean nRMSE: ${avg('Selected NN',250).toFixed(3)} at 250 cases, ${avg('Selected NN',sizes.at(-1)).toFixed(3)} with all training cases`,72,595,1136,38,24,true);
 foot(s,'One CPU thread. Median warmed-up timing includes scaling. TGLF timing was not measured here.');
 s.speakerNotes.text=`Learning curves average normalized errors over outputs and then over three repetitions. The notebook includes standard-deviation bands; this chart shows means for readability. Subsets are nested within each repetition. Scalers and the baseline mean are fitted on each subset. The validation split controls stopping; the displayed test curve does not trigger further tuning. The selected-model training time is ${d.timing.selected_model_training_seconds.toFixed(1)} seconds, and candidate plus extra-seed training totals ${d.timing.all_candidate_and_extra_seed_training_seconds.toFixed(1)} seconds, excluding learning-curve fits. One-case and full-batch timings include feature scaling, inference and inverse target transforms but exclude disk I/O. TGLF seconds/case is the assignment's qualitative reference, so a measured speedup factor is not available. ${sources}`;
}
{
 const s=slide('SAT3 temperature-gradient scan','Deuterium cases b, c, a and d, with both temperature gradients changing together');
 for(const [j,target,label,max,step] of [[0,'Q_i','Ion heat flux',80,20],[1,'Q_e','Electron heat flux',30,10]]){
  chart(s,{position:{left:55+j*625,top:200,width:590,height:340},title:label,titleTextStyle:{fontSize:25,fill:ink,bold:true},series:[['TGLF',d.temperature_truth[target],blue,'circle'],['NN',d.temperature_prediction[target],teal,'square']].map(([name,values,color,symbol])=>({name,xValues:d.temperature_x,values,line:{fill:color,width:3},marker:{symbol,size:8,fill:color}})),xAxis:axis('Both temperature gradients',1.5,3.5,.5),yAxis:axis('Heat flux (gyro-Bohm)',0,max,step,'0')});
 }
 text(s,`Ion heat-flux isotope ordering matches in ${d.isotope_order_matches}/${d.isotope_triplets} complete H/D/T triplets`,72,572,1136,45,28,true,teal);
 text(s,'Misses case c’s isotope reversal and 4 of 7 negative-particle signs in the random test.',72,620,1136,34,22);
 foot(s,'These 43 related scan cases support a limited transfer check. They do not establish broad extrapolation.');
 s.speakerNotes.text=`Both gradients change together, so these plots cannot separate their effects. Lines join the supplied cases. The notebook also checks particle flux and shows all eight complete H/D/T ion heat-flux triplets. The match count refers to the full H/D/T ordering, not only endpoints. Training-set input coverage and negative-particle-flux sign checks appear in the notebook. Further TGLF sampling should focus on local errors and weakly covered regimes rather than assume this small outside test proves generalization everywhere. ${sources}`;
}
// Keep final files separate from all private build and QA artifacts.
const candidate=`${root}/.build/neural_results_candidate.pptx`;
await (await PresentationFile.exportPptx(p)).save(candidate);
await fs.mkdir(`${root}/output`,{recursive:true});
const final=`${root}/output/Task2_Neural_Network_Results_Calderon.pptx`;
await finalizePresentation({workspaceDir:root,candidatePath:candidate,finalPath:final,
 pythonExecutable:'/Users/jcalderon204/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/bin/python3',
 integrityValidatorPath:`${skill}/container_tools/inspect_presentation_package_integrity.py`,layoutValidatorPath:`${skill}/container_tools/inspect_presentation_layout_geometry.py`,
 layoutArgs:['--expected-slide-size-emu','12192000,6858000','--validate-heading-fit','--require-native-table-slide','2','--require-native-table-slide','3'],
 requiredNativeTableOwnerSlides:[2,3],requiredNativeChartOwnerSlides:[4,5],materializeLiteralChartWorkbooks:true,
 fontPolicy:{basis:'design',families:[font]},verifyArtifactToolImport:true,receiptPath:`${root}/.build/deck_validation.json`});
const rendered=await PresentationFile.importPptx(await FileBlob.load(final));
for(let i=0;i<rendered.slides.items.length;i++){
 const b=await rendered.export({slide:rendered.slides.items[i],format:'png',scale:1});
 await fs.writeFile(`${root}/.build/final-slide-${i+1}.png`,new Uint8Array(await b.arrayBuffer()));
}
console.log(final);
