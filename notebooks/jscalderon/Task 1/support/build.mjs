import fs from 'node:fs/promises';
import path from 'node:path';
import {createHash} from 'node:crypto';
import {Presentation,PresentationFile} from '@oai/artifact-tool';
import {applyPresentationChartFont,finalizePresentation} from '/Users/jcalderon204/.codex/plugins/cache/openai-primary-runtime/presentations/26.1015.12040/skills/presentations/container_tools/artifact_tool_utils.mjs';
const root='/Users/jcalderon204/Documents/DSC 180A';
const skill='/Users/jcalderon204/.codex/plugins/cache/openai-primary-runtime/presentations/26.1015.12040/skills/presentations';
const data=JSON.parse(await fs.readFile('parsed.json','utf8'));
const p=Presentation.create({slideSize:{width:1280,height:720}});
const font='Helvetica Neue', ink='#111111',blue='#0077B6',orange='#D35A0A',green='#438566';
function text(s,t,x,y,w,h,size=20,bold=false,color=ink,align='left'){
 const a=s.shapes.add({geometry:'textbox',name:t.slice(0,50),position:{left:x,top:y,width:w,height:h},fill:'none',line:{fill:'none',width:0}});
 a.text=t; a.text.style={fontFamily:font,typeface:font,fontSize:size,color,bold,alignment:align,verticalAlignment:'middle'};a.text.insets=0;return a;
}
function base(title,sub){const s=p.slides.add();s.background.fill='#FFFFFF';text(s,title,40,14,1200,48,32,true,ink,'center');text(s,sub,55,73,1170,40,20,false,ink,'center');return s;}
const axisStyle={fontSize:18,fill:ink};
function scatter(s,{title,cases,key,xkey,xTitle,yMax,yUnit,color=blue,three=false,xMin,xMax,xMajor}){
 const series=(three?['H','D','T']:['D']).map((iso,i)=>({name:three?({H:'Hydrogen',D:'Deuterium',T:'Tritium'})[iso]:title,
 xValues:cases.map(l=>data[`${l}_${iso}`][xkey]),values:cases.map(l=>data[`${l}_${iso}`][key]),line:{fill:three?[blue,orange,green][i]:color,width:3},marker:{symbol:three?['circle','square','triangle'][i]:'circle',size:9,fill:three?[blue,orange,green][i]:color},
 dataLabels:{showValue:!three,position:'top',numberFormatCode:'0.0',textStyle:{fontSize:18,fill:ink}}
 }));
 const c=s.charts.add('scatter',{position:{left:key==='Q_i'?35:650,top:132,width:590,height:410},title,titleTextStyle:{fontSize:23,bold:true,fill:ink},scatterOptions:{style:'lineWithMarkers'},series,hasLegend:three,legend:{position:'top',textStyle:{fontSize:16,fill:ink}},chartFill:'#FFFFFF',chartLine:{fill:'none',width:0},plotAreaFill:'#FFFFFF',
 xAxis:{title:{text:xTitle,textStyle:{fontSize:18,fill:ink}},textStyle:axisStyle,min:xMin,max:xMax,majorUnit:xMajor,numberFormatCode:xkey==='Q_PRIME_LOC'?'0':'0.00',majorGridlines:null,line:{fill:'#AFB5BA',width:1}},
 yAxis:{title:{text:'Heat flux (gyro-Bohm units)',textStyle:{fontSize:18,fill:ink}},textStyle:axisStyle,min:0,max:yMax,majorUnit:yUnit,numberFormatCode:'0',majorGridlines:{fill:'#DFE6EB',width:1},line:{fill:'none',width:0}}
 });applyPresentationChartFont(c,{fontFamily:font});for(let i=0;i<series.length;i++)c.series.getItemAt(i).dataLabels.textStyle.typeface=font;return c;
}
const source='Source: SAT3_dataset.tar, data/<case>/tglf_sat2/input.tglf.gen and out.tglf.gbflux. Task1_SAT3database_revised.docx defines the file layout. Species 1 is the electron (ZS_1=-1), species 2 is the singly charged main ion (ZS_2=1). All heat fluxes are TGLF predictions with SAT_RULE=2 and UNITS=GYRO. These are normalized model outputs, not measured heat losses or CGYRO validation. Lines connect simulated cases and do not establish behavior between sampled points.';
{
const s=base('Steeper temperature gradients raise heat transport','Matched H, D and T scans with both electron and ion temperature gradients changing together');
scatter(s,{title:'Ion heat flux',cases:['b','c','a','d'],key:'Q_i',xkey:'RLTS_2',xTitle:'Temperature gradients (RLTS_1 = RLTS_2)',yMax:80,yUnit:20,three:true,xMin:1.5,xMax:3.5,xMajor:.5});
scatter(s,{title:'Electron heat flux',cases:['b','c','a','d'],key:'Q_e',xkey:'RLTS_1',xTitle:'Temperature gradients (RLTS_1 = RLTS_2)',yMax:30,yUnit:10,three:true,xMin:1.5,xMax:3.5,xMajor:.5});
text(s,'Deuterium: ion heat flux rises 12.6×, from 4.8 to 60.6, as the gradients rise from 1.5 to 3.5',55,563,1170,34,23,true,orange,'center');
text(s,'Why it matters: the curved response makes transport sensitive to changes in temperature gradients.',55,609,1170,52,21,false,ink,'center');
text(s,'Cases b, c, a, d for H/D/T (12 runs). Fixed density gradients = 1 and XNUE = 0.1. TGLF SAT2 predictions.',55,682,1170,24,15,false,'#555555','center');
s.speakerNotes.text=`What to say: Both panels show the same four temperature-gradient settings: 1.5 (b), 2.25 (c), 3 (a) and 3.5 (d). Each colored line keeps the ion species fixed. All generated input parameters remain constant within each isotope scan except RLTS_1 and RLTS_2, which change together. For D, ion heat flux is 4.7995, 20.862, 43.101 and 60.586. Electron heat flux is 2.2156, 7.7842, 15.258 and 20.765. The gradient increases by 2.33 times, while Q_i rises 12.6234 times and Q_e rises 9.3722 times. H and T show the same increasing trend. This is an empirical curved response over four settings. The scan cannot separate the effects of the ion gradient and electron gradient, and it does not locate an instability threshold.\n\nWhy it matters: A surrogate model should reproduce the strong response to the coupled temperature-gradient inputs. A small change in gradient can correspond to a large difference in predicted flux in this sampled regime.\n\n${source}`;
}
{
const s=base('Isotope effects depend on the plasma setting','Eight matched H/D/T triplets isolate ion mass while keeping every other input fixed');
const letters=['a','b','c','d','e','f','j','x'];
const series=['D','T'].map((iso,i)=>({name:`${iso==='D'?'Deuterium':'Tritium'} relative to hydrogen`,values:letters.map(l=>Number((100*(data[`${l}_${iso}`].Q_i/data[`${l}_H`].Q_i-1)).toFixed(6))),fill:i?orange:blue,dataLabels:{showValue:true,position:'outEnd',numberFormatCode:'0.0',textStyle:{fontSize:16,fill:ink}}}));
const c=s.charts.add('bar',{position:{left:40,top:145,width:815,height:415},categories:letters,series,barOptions:{direction:'column',grouping:'clustered',gapWidth:65},hasLegend:true,legend:{position:'top',textStyle:{fontSize:17,fill:ink}},chartFill:'#FFFFFF',chartLine:{fill:'none',width:0},
xAxis:{title:{text:'Matched case letter',textStyle:{fontSize:19,fill:ink}},textStyle:{fontSize:19,fill:ink},line:{fill:'#AFB5BA',width:1},majorGridlines:null},
yAxis:{title:{text:'Ion heat flux increase from H (%)',textStyle:{fontSize:19,fill:ink}},min:0,max:75,majorUnit:15,numberFormatCode:'0',textStyle:axisStyle,majorGridlines:{fill:'#DFE6EB',width:1},line:{fill:'none',width:0}}});applyPresentationChartFont(c,{fontFamily:font});for(let i=0;i<series.length;i++)c.series.getItemAt(i).dataLabels.textStyle.typeface=font;
text(s,'T exceeds H in all 8 triplets',900,169,330,64,25,true,orange);
text(s,'The increase ranges from 7% in c to 63% in j.',900,246,330,82,22);
text(s,'One ordering reverses',900,363,330,36,23,true);
text(s,'In c, T has 12% less ion heat flux than D (18.4 vs 20.9).',900,411,330,91,22);
text(s,'Why it matters: one fixed isotope multiplier would miss the variation across plasma settings.',55,597,1170,60,22,false,ink,'center');
text(s,'24 runs. Mass inputs: H = 0.5, D = 1.0, T = 1.5. Each bar uses its own letter’s H flux as the baseline. TGLF SAT2.',55,681,1170,27,15,false,'#555555','center');
s.speakerNotes.text=`What to say: The eight complete isotope triplets are a, b, c, d, e, f, j and x. Comparing all generated inputs verifies that only MASS_2 changes within each triplet. The bars show 100*(Q_i,isotope / Q_i,H - 1), so each letter has its own H baseline. T/H increases are 40.5503%, 11.7717%, 6.7223%, 42.3125%, 25.3540%, 19.9235%, 63.1741% and 56.4535%. The D/H increases are 24.3071%, 9.5776%, 21.0023%, 25.3382%, 18.1571%, 13.7913%, 37.3454% and 33.0708%. All eight triplets have T above H. Seven have H < D < T. Case c instead has H=17.241, D=20.862 and T=18.4, making T 11.8014% below D. Letter c uses both temperature gradients 2.25. Letters a/b/d use gradients 3/1.5/3.5, e/f use density gradients 2/3, j uses XNUE=0.01, and x combines density gradients 3 with XNUE=0.01.\n\nWhy it matters: The mass response changes with the other inputs. A model should allow mass to interact with plasma conditions rather than apply one constant scaling. These eight controlled comparisons support a pattern in this dataset, not a universal isotope law. Letter y has H/T only and is excluded because the chart compares complete H/D/T triplets.\n\n${source}`;
}
{
const s=base('A larger safety-factor gradient raises both heat fluxes','Matched deuterium scan with Q_LOC = 2 and RMIN_LOC = 0.5 fixed');
scatter(s,{title:'Ion heat flux',cases:['g','h','a','i'],key:'Q_i',xkey:'Q_PRIME_LOC',xTitle:'Safety-factor gradient (Q_PRIME_LOC)',yMax:60,yUnit:20,color:orange,xMin:0,xMax:28,xMajor:4});
scatter(s,{title:'Electron heat flux',cases:['g','h','a','i'],key:'Q_e',xkey:'Q_PRIME_LOC',xTitle:'Safety-factor gradient (Q_PRIME_LOC)',yMax:24,yUnit:8,color:blue,xMin:0,xMax:28,xMajor:4});
text(s,'From Q_PRIME_LOC = 4 to 24: ion heat flux rises 2.8× and electron heat flux rises 2.5×',55,564,1170,34,23,true,orange,'center');
text(s,'Why it matters: magnetic geometry adds predictive information even when the driving gradients stay fixed.',55,608,1170,58,21,false,ink,'center');
text(s,'Cases g_D, h_D, a_D, i_D (4 runs). Only Q_PRIME_LOC varies. Q_i = 18.5 to 51.6, Q_e = 7.0 to 17.4. TGLF SAT2.',55,681,1170,27,15,false,'#555555','center');
s.speakerNotes.text=`What to say: The horizontal axis is Q_PRIME_LOC, the safety-factor gradient parameter that the assignment associates with magnetic shear. The four cases are g_D (4), h_D (8), a_D (16) and i_D (24). All generated input parameters except Q_PRIME_LOC match exactly. In particular, Q_LOC=2, RMIN_LOC=0.5, both temperature gradients=3, both density gradients=1, XNUE=0.1 and ion mass=1 stay fixed. Q_i is 18.466, 30.01, 43.101 and 51.562. Q_e is 7.0371, 10.972, 15.258 and 17.405. The endpoint factors are 2.7923 and 2.4733. Both responses rise, with smaller increments in flux per unit of Q_PRIME_LOC at higher values.\n\nWhy it matters: Input variation in magnetic geometry changes predicted heat transport even at fixed density and temperature gradients. This makes the safety-factor gradient a useful candidate feature for the regression dataset. The result applies to this four-point scan and does not show that all geometric changes or all plasma regimes behave similarly.\n\n${source}`;
}
for(let i=0;i<p.slides.items.length;i++){
 const blob=await p.export({slide:p.slides.items[i],format:'png',scale:1.5});await fs.writeFile(`slide-${i+1}.png`,new Uint8Array(await blob.arrayBuffer()));
}
await fs.writeFile('speaker_notes.json',JSON.stringify(p.slides.items.map(s=>s.speakerNotes.text),null,2));
const candidate=path.join(root,'.sat3_build','candidate.pptx');await (await PresentationFile.exportPptx(p)).save(candidate);
const referencePath=path.join(root,'density_gradients_and_collisions.pptx');
const result=await finalizePresentation({workspaceDir:root,candidatePath:candidate,finalPath:path.join(root,'output','SAT3_dataset_patterns.pptx'),pythonExecutable:'/Users/jcalderon204/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/bin/python3',integrityValidatorPath:path.join(skill,'container_tools/inspect_presentation_package_integrity.py'),layoutValidatorPath:path.join(skill,'container_tools/inspect_presentation_layout_geometry.py'),layoutArgs:['--expected-slide-size-emu','12192000,6858000','--validate-bullet-geometry','--validate-heading-fit'],explicitTotalSlideCount:3,requiredNativeChartOwnerSlides:[1,2,3],materializeLiteralChartWorkbooks:true,fontPolicy:{basis:'reference',families:[font],referencePath,referenceSha256:createHash('sha256').update(await fs.readFile(referencePath)).digest('hex')},verifyArtifactToolImport:true,receiptPath:path.join(root,'.sat3_build','validation.json')});
console.log(JSON.stringify(result));
