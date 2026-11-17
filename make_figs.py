import json, numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams.update({"font.family":"serif","font.size":8,"axes.linewidth":0.8,"savefig.dpi":300})
BLUE,VERM,GREEN,GRAY,BLACK="#0072B2","#D55E00","#009E73","#7a7a7a","#222222"
M=json.load(open("res/main.json")); res=M["results"]
names=["Confidence Cascade","Decision Tree (single)","HGB (single)","Random Forest (single)","Specialist Ensemble","AdaBoost (SAMME)","Bagging (DT x100)","Hierarchical IDS","Mixture of Experts","Hard Voting","Soft Voting","Blending (LR meta)","Stacking (LR meta)","Feature-View Ensemble","Balanced Bagging","DCS","DES"]
num={n:i+1 for i,n in enumerate(names)}
nd=set(M["pareto"]["f1_vs_median_latency"]); freq=M["pareto"]["bootstrap_membership_freq"]

def clusters(pts, tx, ty, fx=0.045, fy=0.055):
    """group points whose normalised positions are close; pts = {name:(x,y)} in axis-transformed units"""
    keys=list(pts); used=set(); out=[]
    for k in keys:
        if k in used: continue
        grp=[k]; used.add(k)
        for j in keys:
            if j in used: continue
            if abs(pts[k][0]-pts[j][0])/tx<fx and abs(pts[k][1]-pts[j][1])/ty<fy: grp.append(j); used.add(j)
        out.append(grp)
    return out

def legend_text(ax, x, y, mark_nd=False, title=None):
    lines=[]
    for n in names:
        star=" *" if (mark_nd and n in nd) else ""
        lines.append(f"{num[n]:>2}  {n}{star}")
    ax.text(x,y,"\n".join(lines),transform=ax.transAxes,va="top",ha="left",family="monospace",fontsize=6.6,linespacing=1.35)

# ---------- Pareto: macro-F1 vs median latency
fig,ax=plt.subplots(figsize=(7.2,3.5)); 
ax.set_xscale("log")
pts={}
for n in names:
    r=res[n]; l=r["latency"]; x=l["median_ms"]; y=r["macro_f1"]
    if n=="Balanced Bagging": continue
    ax.plot([x,l["p95_ms"]],[y,y],color=GRAY,lw=0.7,alpha=0.7,zorder=1)
    isnd=n in nd
    ax.scatter([x],[y],s=46 if isnd else 30,color=VERM if isnd else "#b8b8b8",edgecolor=BLACK,linewidth=0.6,marker="o" if isnd else "s",zorder=3)
    pts[n]=(np.log10(x),y)
ylo,yhi=0.99925,0.99980
ax.set_ylim(ylo,yhi); ax.set_xlim(0.1,200)
for g in clusters(pts,np.log10(200/0.1),yhi-ylo):
    xs=np.mean([10**pts[k][0] for k in g]); ys=max(pts[k][1] for k in g)
    ax.annotate(", ".join(str(num[k]) for k in sorted(g,key=lambda k:num[k])),(xs,ys),xytext=(0,8),textcoords="offset points",ha="center",fontsize=6.8)
# front
fr=sorted([(res[n]["latency"]["median_ms"],res[n]["macro_f1"]) for n in nd])
fx,fy=[],[]
for i,(x,y) in enumerate(fr):
    if i>0: fx.append(x); fy.append(fr[i-1][1])
    fx.append(x); fy.append(y)
ax.step(fx,fy,where="post",color=VERM,lw=1.0,ls="--",zorder=2)
ax.annotate("15: Balanced Bagging, macro-F1 0.99808\n(below the plotted range)",(0.12,ylo),xytext=(0.12,ylo+0.00004),fontsize=6.5,color=BLACK)
ax.set_xlabel("Median single-row latency (ms, log scale). Grey bars end at the 95th percentile.")
ax.set_ylabel("Macro-F1")
ax.grid(True,which="major",color="#dddddd",lw=0.6); ax.set_axisbelow(True)
for s in ("top","right"): ax.spines[s].set_visible(False)
ax.text(1.02,1.0,"Detector",transform=ax.transAxes,fontsize=7,weight="bold",va="top")
legend_text(ax,1.02,0.95,mark_nd=True)
ax.text(1.02,0.15,"* non-dominated on macro-F1\n  versus median latency.\nCircles: non-dominated.\nSquares: dominated.",transform=ax.transAxes,fontsize=6.5,va="top")
fig.subplots_adjust(left=0.09,right=0.70,bottom=0.15,top=0.97)
fig.savefig("figs/fig_pareto.png"); fig.savefig("figs/fig_pareto.pdf"); plt.close(fig)

# ---------- Accuracy vs Brier
fig,ax=plt.subplots(figsize=(7.2,3.5)); ax.set_yscale("log")
xlo,xhi=0.99935,0.99995; ax.set_xlim(xlo,xhi); ax.set_ylim(1e-4,2)
zoom=dict(x=(0.99983,0.99993),y=(1.45e-4,4.4e-4))
inz=lambda r: zoom["x"][0]<=r["accuracy"]<=zoom["x"][1] and zoom["y"][0]<=r["brier"]<=zoom["y"][1]
for n in names:
    r=res[n]
    if "brier" not in r: continue
    ax.scatter([r["accuracy"]],[r["brier"]],s=34,color=BLUE,edgecolor=BLACK,linewidth=0.6,zorder=3)
    if not inz(r):
        right=r["accuracy"]>0.99989
        ax.annotate(str(num[n]),(r["accuracy"],r["brier"]),xytext=(-6,3) if right else (6,3),textcoords="offset points",ha="right" if right else "left",fontsize=6.8)
axi=ax.inset_axes([0.30,0.10,0.34,0.62]); axi.set_yscale("log")
axi.set_xlim(*zoom["x"]); axi.set_ylim(*zoom["y"])
pi={}
for n in names:
    r=res[n]
    if "brier" in r and inz(r):
        axi.scatter([r["accuracy"]],[r["brier"]],s=28,color=BLUE,edgecolor=BLACK,linewidth=0.6,zorder=3); pi[n]=(r["accuracy"],np.log10(r["brier"]))
for g in clusters(pi,zoom["x"][1]-zoom["x"][0],np.log10(zoom["y"][1]/zoom["y"][0]),0.03,0.045):
    xs=np.mean([pi[k][0] for k in g]); ys=10**np.mean([pi[k][1] for k in g])
    axi.annotate(", ".join(str(num[k]) for k in sorted(g,key=lambda k:num[k])),(xs,ys),xytext=(5,0),textcoords="offset points",ha="left",va="center",fontsize=6.8)
axi.set_xticks([0.99985,0.99990]); axi.set_xticklabels(["0.99985","0.99990"],fontsize=6)
axi.set_yticks([2e-4,3e-4]); axi.set_yticklabels(["2e-4","3e-4"],fontsize=6); axi.minorticks_off()
axi.grid(True,color="#e2e2e2",lw=0.5); axi.set_title("zoom on the dense group",fontsize=6.5)
ax.indicate_inset_zoom(axi,edgecolor=GRAY,lw=0.6)
ax.set_xlabel("Accuracy (grouped test split, 60,229 flows)"); ax.set_ylabel("Brier score (log scale)")
ax.grid(True,which="major",color="#dddddd",lw=0.6); ax.set_axisbelow(True)
for sp in ("top","right"): ax.spines[sp].set_visible(False)
ax.text(1.02,1.0,"Detector",transform=ax.transAxes,fontsize=7,weight="bold",va="top")
legend_text(ax,1.02,0.95)
ax.text(1.02,0.10,"Hard Voting (10) has no probability\noutput, so it has no Brier score.",transform=ax.transAxes,fontsize=6.5,va="top")
fig.subplots_adjust(left=0.10,right=0.70,bottom=0.15,top=0.97)
fig.savefig("figs/fig_acc_vs_brier.png"); fig.savefig("figs/fig_acc_vs_brier.pdf"); plt.close(fig)

# ---------- reliability helpers
z=np.load("/mnt/user-data/uploads/ab7615f7-main_preds.npz",allow_pickle=True); y=z["y"].astype(int)
def bins(P,y,B=15):
    conf=P.max(1); ok=(P.argmax(1)==y); idx=np.minimum((conf*B).astype(int),B-1)
    out=[]
    for b in range(B):
        m=idx==b; n=int(m.sum())
        if n: out.append((conf[m].mean(),ok[m].mean(),n))
    return out
def draw(ax,series,title):
    ax.plot([0,1],[0,1],ls="--",color=BLACK,lw=0.7)
    for lab,col,mk,P in series:
        bb=bins(P,y)
        ax.scatter([b[0] for b in bb],[b[1] for b in bb],s=[6+9*np.log10(b[2]+1)**1.6 for b in bb],color=col,marker=mk,edgecolor=BLACK,linewidth=0.4,alpha=0.85,label=lab,zorder=3)
    ax.set_title(title,fontsize=7); ax.set_xlim(-0.03,1.03); ax.set_ylim(-0.05,1.08)
    ax.grid(True,color="#e2e2e2",lw=0.5)
    for s in ("top","right"): ax.spines[s].set_visible(False)
# raw reliability, six detectors
six=["Decision Tree (single)","Random Forest (single)","Confidence Cascade","Soft Voting","AdaBoost (SAMME)","Stacking (LR meta)"]
fig,axs=plt.subplots(2,3,figsize=(7.2,4.6),sharex=True,sharey=True)
for ax,n in zip(axs.ravel(),six):
    P=z["proba__"+n].astype(float); conf=P.max(1); acc=(P.argmax(1)==y).mean()
    draw(ax,[("raw",BLUE,"o",P)],f"{n}\nECE {res[n]['ece']:.4f}, mean conf. {conf.mean():.3f}")
    bb=bins(P,y); big=max(bb,key=lambda b:b[2]); ax.text(0.97,0.14,f"largest bin: {big[2]:,} flows",transform=ax.transAxes,ha="right",fontsize=6)
for ax in axs[1]: ax.set_xlabel("Mean confidence in bin")
for ax in axs[:,0]: ax.set_ylabel("Accuracy in bin")
fig.text(0.5,0.005,"Marker area grows with the number of test flows in the bin (15 equal-width bins). Dashed line: perfect calibration.",ha="center",fontsize=6.5)
fig.subplots_adjust(left=0.07,right=0.99,bottom=0.12,top=0.90,wspace=0.08,hspace=0.40)
fig.savefig("figs/fig_reliability.png"); fig.savefig("figs/fig_reliability.pdf"); plt.close(fig)

# ---------- calibration after
c=np.load("/mnt/user-data/uploads/155d4842-calib_probas.npz",allow_pickle=True); yc=c["y"].astype(int); assert (yc==y).all()
Cj=json.load(open("res/calib.json"))["calibration"]
fig,axs=plt.subplots(1,3,figsize=(7.2,2.9),sharey=True)
for ax,n in zip(axs,["AdaBoost (SAMME)","Soft Voting","Feature-View Ensemble"]):
    ser=[("raw",VERM,"o",c[n+"__raw"].astype(float)),("temperature",BLUE,"s",c[n+"__temperature"].astype(float)),("isotonic",GREEN,"^",c[n+"__isotonic"].astype(float))]
    draw(ax,ser,n+"\nECE raw "+f"{Cj[n]['raw']['ece']:.4f}"+", temp. "+f"{Cj[n]['temperature']['ece']:.4f}"+", iso. "+f"{Cj[n]['isotonic']['ece']:.4f}")
    ax.set_xlabel("Mean confidence in bin")
axs[0].set_ylabel("Accuracy in bin"); axs[0].legend(loc="center left",fontsize=6.5,frameon=False,markerscale=0.8)
fig.subplots_adjust(left=0.07,right=0.99,bottom=0.19,top=0.84,wspace=0.08)
fig.savefig("figs/fig_calibration_after.png"); fig.savefig("figs/fig_calibration_after.pdf"); plt.close(fig)
print("ok")
