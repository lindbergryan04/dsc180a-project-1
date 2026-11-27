import sys,zipfile
from lxml import etree as E
source,out=sys.argv[1:]
ns={'p':'http://schemas.openxmlformats.org/presentationml/2006/main','r':'http://schemas.openxmlformats.org/officeDocument/2006/relationships'}
drop={'ppt/slides/slide2.xml','ppt/slides/_rels/slide2.xml.rels','ppt/notesSlides/notesSlide2.xml','ppt/notesSlides/_rels/notesSlide2.xml.rels','ppt/slides/charts/chart3.xml','ppt/slides/charts/_rels/chart3.xml.rels','ppt/embeddings/chart-data-snapshot-003.xlsx'}
with zipfile.ZipFile(source) as src,zipfile.ZipFile(out,'w',zipfile.ZIP_DEFLATED) as dst:
 root=E.fromstring(src.read('ppt/presentation.xml'));lst=root.find('p:sldIdLst',ns);assert len(lst)==3;rid=lst[1].get('{'+ns['r']+'}id');lst.remove(lst[1])
 for item in src.infolist():
  if item.filename in drop:continue
  data=src.read(item.filename)
  if item.filename=='ppt/presentation.xml':data=E.tostring(root,xml_declaration=True,encoding='UTF-8',standalone=True)
  elif item.filename=='ppt/_rels/presentation.xml.rels':
   rels=E.fromstring(data)
   for n in list(rels):
    if n.get('Id')==rid:rels.remove(n)
   data=E.tostring(rels,xml_declaration=True,encoding='UTF-8',standalone=True)
  elif item.filename=='[Content_Types].xml':
   c=E.fromstring(data)
   for n in list(c):
    if n.get('PartName','').lstrip('/') in drop:c.remove(n)
   data=E.tostring(c,xml_declaration=True,encoding='UTF-8',standalone=True)
  elif item.filename=='docProps/app.xml':
   c=E.fromstring(data)
   for n in c:
    if E.QName(n).localname in ['Slides','Notes']:n.text='2'
   data=E.tostring(c,xml_declaration=True,encoding='UTF-8',standalone=True)
  dst.writestr(item,data)
