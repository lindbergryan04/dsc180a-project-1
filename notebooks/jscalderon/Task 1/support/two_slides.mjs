import fs from 'node:fs/promises';
import path from 'node:path';
import {execFileSync} from 'node:child_process';
import {createHash} from 'node:crypto';
import {PresentationFile,FileBlob} from '@oai/artifact-tool';
import {finalizePresentation} from '/Users/jcalderon204/.codex/plugins/cache/openai-primary-runtime/presentations/26.1015.12040/skills/presentations/container_tools/artifact_tool_utils.mjs';
const root='/Users/jcalderon204/Documents/DSC 180A';
const skill='/Users/jcalderon204/.codex/plugins/cache/openai-primary-runtime/presentations/26.1015.12040/skills/presentations';
const source=path.join(root,'output/SAT3_dataset_patterns_revised.pptx');
const p=await PresentationFile.importPptx(await FileBlob.load(source));
p.slides.keep([0,2]);
await(await PresentationFile.exportPptx(p)).save('two-slide-edited.pptx');
const candidate=path.join(root,'.sat3_build/two-slide-preserved.pptx');
execFileSync('/Users/jcalderon204/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/bin/python3',['remove_slide2.py',source,candidate]);
const finalPath=path.join(root,'output/SAT3_dataset_patterns_two_slides.pptx');
await finalizePresentation({workspaceDir:root,candidatePath:candidate,finalPath,pythonExecutable:'/Users/jcalderon204/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/bin/python3',integrityValidatorPath:path.join(skill,'container_tools/inspect_presentation_package_integrity.py'),layoutValidatorPath:path.join(skill,'container_tools/inspect_presentation_layout_geometry.py'),layoutArgs:['--expected-slide-size-emu','12192000,6858000','--validate-bullet-geometry','--validate-heading-fit'],explicitTotalSlideCount:2,requiredNativeChartOwnerSlides:[1,2],fontPolicy:{basis:'reference',families:['Helvetica Neue'],referencePath:source,referenceSha256:createHash('sha256').update(await fs.readFile(source)).digest('hex')},verifyArtifactToolImport:true,receiptPath:path.join(root,'.sat3_build/two-slide-validation.json')});
const final=await PresentationFile.importPptx(await FileBlob.load(finalPath));
for(let i=0;i<final.slides.items.length;i++){const b=await final.export({slide:final.slides.items[i],format:'png',scale:1});await fs.writeFile(`two-slide-${i+1}.png`,new Uint8Array(await b.arrayBuffer()));}
console.log(finalPath);
