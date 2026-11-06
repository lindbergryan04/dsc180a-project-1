import fs from 'node:fs/promises';
import {FileBlob,PresentationFile} from '@oai/artifact-tool';
for (const [j,file] of ['TGLF_surrogate.pptx','Task 2_ Predicting TGLF Results w_ Modeling.pptx'].entries()) {
 const p=await PresentationFile.importPptx(await FileBlob.load(`task2/${file}`));
 for (let i=0;i<p.slides.items.length;i++) {
  const b=await p.export({slide:p.slides.items[i],format:'png',scale:1});
  await fs.writeFile(`task2/.build/reference/deck${j}-slide${i+1}.png`,new Uint8Array(await b.arrayBuffer()));
 }
}
