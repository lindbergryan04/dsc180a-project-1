import fs from 'node:fs/promises';
import {FileBlob,PresentationFile} from '@oai/artifact-tool';
const p=await PresentationFile.importPptx(await FileBlob.load('../output/SAT3_dataset_patterns.pptx'));
for(let i=0;i<p.slides.items.length;i++){
 const b=await p.export({slide:p.slides.items[i],format:'png',scale:1});await fs.writeFile(`final-${i+1}.png`,new Uint8Array(await b.arrayBuffer()));
}
console.log((await p.inspect({kind:'slide,chart',maxChars:3000})).ndjson);
