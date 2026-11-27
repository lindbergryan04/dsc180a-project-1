import fs from 'node:fs/promises';
import path from 'node:path';
import {execFileSync} from 'node:child_process';
import {createHash} from 'node:crypto';
import {PresentationFile,FileBlob} from '@oai/artifact-tool';
import {finalizePresentation} from '/Users/jcalderon204/.codex/plugins/cache/openai-primary-runtime/presentations/26.1015.12040/skills/presentations/container_tools/artifact_tool_utils.mjs';
const root='/Users/jcalderon204/Documents/DSC 180A';
const skill='/Users/jcalderon204/.codex/plugins/cache/openai-primary-runtime/presentations/26.1015.12040/skills/presentations';
const source=path.join(root,'output/SAT3_dataset_patterns.pptx');
const p=await PresentationFile.importPptx(await FileBlob.load(source));
const edits=[
 ['sh/ts7md4r2','Temperature gradients: Increasing both temperature gradients from 1.5 to 3.5 raises deuterium ion heat flux from 4.8 to 60.6, a 12.6× increase. This demonstrates that transport responds strongly and nonlinearly to the gradients, making temp gradients a great lever for further study.',609,98],
 ['sh/obq90bml','Leaving all other inputs fixed but increasing the safety factor gradient induces great change in ion and electron heatflux, making it a strong predictive feature for a TGLF model down the road.',608,98]
];
for(const [id,content,top,height] of edits){const a=p.resolve(id);a.text=content;a.position={left:55,top,width:1170,height};a.text.style={fontSize:21,typeface:'Helvetica Neue',color:'#111111',alignment:'center',verticalAlignment:'middle'};a.text.insets=0;}
p.delete(['sh/sryl4zqx','sh/pcjqtg36']);
const candidate=path.join(root,'.sat3_build/revised-candidate.pptx');
await(await PresentationFile.exportPptx(p)).save(candidate);
const preserved=path.join(root,'.sat3_build/revised-preserved.pptx');
execFileSync('/Users/jcalderon204/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/bin/python3',['preserve_chart_package.py',source,candidate,preserved]);
const finalPath=path.join(root,'output/SAT3_dataset_patterns_revised.pptx');
await finalizePresentation({workspaceDir:root,candidatePath:preserved,finalPath,pythonExecutable:'/Users/jcalderon204/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/bin/python3',integrityValidatorPath:path.join(skill,'container_tools/inspect_presentation_package_integrity.py'),layoutValidatorPath:path.join(skill,'container_tools/inspect_presentation_layout_geometry.py'),layoutArgs:['--expected-slide-size-emu','12192000,6858000','--validate-bullet-geometry','--validate-heading-fit'],explicitTotalSlideCount:3,requiredNativeChartOwnerSlides:[1,2,3],materializeLiteralChartWorkbooks:false,fontPolicy:{basis:'reference',families:['Helvetica Neue'],referencePath:source,referenceSha256:createHash('sha256').update(await fs.readFile(source)).digest('hex')},verifyArtifactToolImport:true,receiptPath:path.join(root,'.sat3_build/revised-validation.json')});
const final=await PresentationFile.importPptx(await FileBlob.load(finalPath));
for(let i=0;i<final.slides.items.length;i++){
const b=await final.export({slide:final.slides.items[i],format:'png',scale:1.5});await fs.writeFile(`revised-${i+1}.png`,new Uint8Array(await b.arrayBuffer()));}
console.log(finalPath);
