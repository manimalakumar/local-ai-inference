"""Plot only the records explicitly selected in this notebook session."""
from pathlib import Path
import uuid


def figure(study):
    results=study.current_results()
    rows=[]
    for name,result in results.items():
        if result.get('status')!='completed':continue
        phases=(result.get('cache_study') or {}).get('phases') or {}
        if phases:
            for phase in ('cold','warm'):
                stats=phases.get(phase) or {}
                if stats.get('ttft_p50_seconds') is not None:
                    rows.append((name+' / '+phase,stats['ttft_p50_seconds'],stats.get('e2e_p50_seconds')))
        elif (result.get('serving') or {}).get('e2e_p50_seconds') is not None:
            stats=result['serving'];rows.append((name,stats.get('ttft_p50_seconds'),stats['e2e_p50_seconds']))
    if not rows:
        return None
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(12,max(3,len(rows)*.45)))
    for ax,index,title in ((axes[0],1,'TTFT'),(axes[1],2,'Request completion')):
        selected=[row for row in rows if row[index] is not None]
        ax.barh([r[0] for r in selected],[r[index] for r in selected])
        ax.set_xlabel('Median seconds');ax.set_title(title);ax.invert_yaxis()
    fig.suptitle(study.mode+' · '+study.session_id)
    fig.tight_layout()
    if study.live:
        path=study.root/'experiment-tracker/mvp4/workflow'/f'{study.tag}-figure-{uuid.uuid4().hex[:8]}.png'
        path.parent.mkdir(parents=True,exist_ok=True);fig.savefig(path,dpi=150)
        print('Session figure:',path)
    return fig
