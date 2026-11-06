import sys,zipfile,copy
from lxml import etree as E
src,edited,out=sys.argv[1:]
ns={'p':'http://schemas.openxmlformats.org/presentationml/2006/main','a':'http://schemas.openxmlformats.org/drawingml/2006/main'}
changes={
'ppt/slides/slide1.xml':('Why it matters: the curved response makes transport sensitive to changes in temperature gradients.','Temperature gradients: Increasing both temperature gradients from 1.5 to 3.5 raises deuterium ion heat flux from 4.8 to 60.6, a 12.6× increase. This demonstrates that transport responds strongly and nonlinearly to the gradients, making temp gradients a great lever for further study.','Cases b, c, a, d for H/D/T (12 runs). Fixed density gradients = 1 and XNUE = 0.1. TGLF SAT2 predictions.'),
'ppt/slides/slide3.xml':('Why it matters: magnetic geometry adds predictive information even when the driving gradients stay fixed.','Leaving all other inputs fixed but increasing the safety factor gradient induces great change in ion and electron heatflux, making it a strong predictive feature for a TGLF model down the road.','Cases g_D, h_D, a_D, i_D (4 runs). Only Q_PRIME_LOC varies. Q_i = 18.5 to 51.6, Q_e = 7.0 to 17.4. TGLF SAT2.')}
def txt(node):return ''.join(node.xpath('.//a:t/text()',namespaces=ns))
with zipfile.ZipFile(src) as a,zipfile.ZipFile(edited) as b,zipfile.ZipFile(out,'w',zipfile.ZIP_DEFLATED) as z:
 for item in a.infolist():
  payload=a.read(item.filename)
  if item.filename in changes:
   old,new,foot=changes[item.filename];r=E.fromstring(payload);t=E.fromstring(b.read(item.filename))
   target=[s for s in r.xpath('.//p:sp',namespaces=ns) if txt(s)==old];replacement=[s for s in t.xpath('.//p:sp',namespaces=ns) if txt(s)==new];footer=[s for s in r.xpath('.//p:sp',namespaces=ns) if txt(s)==foot]
   assert len(target)==len(replacement)==len(footer)==1
   updated=copy.deepcopy(replacement[0]);updated.find('p:nvSpPr/p:cNvPr',ns).attrib.update(target[0].find('p:nvSpPr/p:cNvPr',ns).attrib)
   target[0].getparent().replace(target[0],updated);footer[0].getparent().remove(footer[0]);payload=E.tostring(r,xml_declaration=True,encoding='UTF-8',standalone=True)
  z.writestr(item,payload)
