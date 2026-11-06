from pathlib import Path
import json,csv
from reportlab.platypus import SimpleDocTemplate,Paragraph,Spacer,KeepTogether
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import letter
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
pdfmetrics.registerFont(TTFont('TimesEmbedded','/System/Library/Fonts/Supplemental/Times New Roman.ttf'))
pdfmetrics.registerFont(TTFont('TimesBoldEmbedded','/System/Library/Fonts/Supplemental/Times New Roman Bold.ttf'))

root=Path('/Users/jcalderon204/Documents/DSC 180A')
with (root/'task2/results/metrics.csv').open() as f:
    metrics=list(csv.DictReader(f))
for row in metrics: row['R2']=float(row['R2'])
d={'metrics':metrics}
test=[r for r in d['metrics'] if r['dataset']=='Random test' and r['model']=='Selected NN']
sat=[r for r in d['metrics'] if r['dataset']=='SAT3' and r['model']=='Selected NN']
q1='1. Describe one result you found interesting doing your task.'
q2='2. Describe one thing you struggled with doing your task.'
q3='3. Identify one choice that you made in your investigation. Why did you make this choice?'
p1=(f'One result I found interesting was that the selected neural network could predict the TGLF fluxes well on both the random test set and the separate SAT3 cases. '
    f'Its R-squared values ranged from {min(r["R2"] for r in test):.3f} to {max(r["R2"] for r in test):.3f} on the 750 random test cases and from '
    f'{min(r["R2"] for r in sat):.3f} to {max(r["R2"] for r in sat):.3f} on the 43 SAT3 cases. '
    'The SAT3 check was especially useful because those cases came from controlled scans rather than the random sampling used for training. '
    'However, the model missed the isotope-order reversal in case c even though it matched the other seven complete triplets. '
    'This showed me why checking physical trends matters in addition to reporting a high average score.')
p2=('One difficulty in this task was handling the different scales and signs of the outputs without making the comparison misleading. '
    'The heat fluxes span a wide range, so predicting their logarithms was a reasonable option, but particle flux can be negative and cannot use the same transformation. '
    'I also found four cases with nonpositive ion heat flux and excluded them from the modeling data. '
    'To compare the network variants fairly, I evaluated their predictions after converting them back to the original units. '
    'Keeping the training, validation, and test sets separate was also important because choosing settings based on test results would make the reported accuracy too optimistic.')
p3=('One choice I made was to focus on neural networks because my groupmates had already found that they performed best among the model families they compared. '
    'I built on that work by comparing network size, depth, and output transformations, then checking whether averaging three networks improved validation error. '
    'I kept a training-mean predictor as the simple baseline required by the assignment. '
    'Repeating the full comparison with random forests and other models would also have been reasonable, but focusing on neural networks gave my contribution a more specific purpose. '
    'I used the validation set to choose the model and reserved the random test set and SAT3 cases for evaluating the final choice.')
output=root/'Participation Assignments/PA 4 DSC 180A | Calderon.pdf'
doc=SimpleDocTemplate(str(output),pagesize=letter,rightMargin=72,leftMargin=72,topMargin=72,bottomMargin=65,
                     title='Participation Assignment 4 DSC 180A',author='Calderon')
question=ParagraphStyle('Question',fontName='TimesBoldEmbedded',fontSize=12,leading=17,spaceAfter=12,keepWithNext=True)
body=ParagraphStyle('Body',fontName='TimesEmbedded',fontSize=12,leading=18,firstLineIndent=36,spaceAfter=18,alignment=TA_LEFT)
story=[]
for q,p in [(q1,p1),(q2,p2),(q3,p3)]:story.extend([Paragraph(q,question),Paragraph(p,body)])
doc.build(story)
(root/'task2/.build/participation_text.json').write_text(json.dumps([[q1,p1],[q2,p2],[q3,p3]],indent=2))
print(output)
