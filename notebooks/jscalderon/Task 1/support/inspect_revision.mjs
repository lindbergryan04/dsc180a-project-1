import fs from 'node:fs/promises';
import {PresentationFile,FileBlob} from '@oai/artifact-tool';
const p=await PresentationFile.importPptx(await FileBlob.load('../output/SAT3_dataset_patterns.pptx'));
console.log((await p.inspect({kind:'textbox',maxChars:18000})).ndjson);
