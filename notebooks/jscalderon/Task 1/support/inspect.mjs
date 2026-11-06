import fs from 'node:fs/promises';
import {FileBlob,PresentationFile} from '@oai/artifact-tool';
const p=await PresentationFile.importPptx(await FileBlob.load('../density_gradients_and_collisions.pptx'));
console.log((await p.inspect({kind:'slide,textbox,layout',maxChars:12000})).ndjson);
for(let i=0;i<p.slides.items.length;i++){
 const b=await p.export({slide:p.slides.items[i],format:'png',scale:1});await fs.writeFile(`reference-${i+1}.png`,new Uint8Array(await b.arrayBuffer()));
}
