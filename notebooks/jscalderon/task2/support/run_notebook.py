import os,sys,json,io,base64,traceback,time
from pathlib import Path
root=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(root/'.packages'))
os.environ['MPLBACKEND']='Agg'
os.environ['MPLCONFIGDIR']=str(root/'.build/mpl')
os.chdir(root)
import nbformat
from IPython.core.interactiveshell import InteractiveShell
from IPython.utils.capture import capture_output
import matplotlib.pyplot as plt
from IPython.display import display,Image
nbpath=root/'Task2_Neural_Networks_Calderon.ipynb'
nb=nbformat.read(nbpath,as_version=4)
shell=InteractiveShell.instance()
def show(*args,**kwargs):
    for num in plt.get_fignums():
        fig=plt.figure(num);buffer=io.BytesIO();fig.savefig(buffer,format='png',bbox_inches='tight')
        display(Image(data=buffer.getvalue()))
    plt.close('all')
plt.show=show
count=0
for cell in nb.cells:
    if cell.cell_type!='code':continue
    count+=1;print(f'Starting code cell {count}',flush=True);start=time.time()
    with capture_output() as cap:
        result=shell.run_cell(cell.source,store_history=True)
    cell.execution_count=count;cell.outputs=[]
    if cap.stdout:cell.outputs.append(nbformat.v4.new_output('stream',name='stdout',text=cap.stdout))
    if cap.stderr:cell.outputs.append(nbformat.v4.new_output('stream',name='stderr',text=cap.stderr))
    for output in cap.outputs:
        cell.outputs.append(nbformat.v4.new_output('display_data',data=output.data,metadata=output.metadata))
    nbformat.write(nb,nbpath)
    print(cap.stdout,flush=True);print(f'Finished cell {count} in {time.time()-start:.1f}s',flush=True)
    if result.error_before_exec or result.error_in_exec:
        print(cap.stderr,flush=True);raise SystemExit(1)
nbformat.validate(nb)
print('Notebook executed successfully',flush=True)
