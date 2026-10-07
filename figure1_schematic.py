"""Figure 1 (fig1_schematic.pdf): model schematic.

Pure vector drawing: no numerical content, so nothing here needs to agree with a
table.  Kept in the package so the figure can be regenerated from source.
"""
import numpy as np
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch, Rectangle
import matplotlib.pyplot as plt
from _common import banner, OUT
from organoid_null import style
from organoid_null.style import (SURFACE, INK, SUBTLE, MUTED, GRID, BLUE, ORANGE,
                                 AQUA, RAMP)

TP, TS, MUT, SURF = INK, SUBTLE, MUTED, SURFACE
pan = lambda a, L, t, sub=None: style.panel(a, L, t, sub, x=-0.02, y=1.20,
                                            dx=0.075, dy=0.125, size=8.6)


def main():
    fig=plt.figure(figsize=(9.6,6.4))
    gs=fig.add_gridspec(2,2,hspace=0.55,wspace=0.30)
    # A chamber
    a=fig.add_subplot(gs[0,0]); a.set_aspect("equal"); a.axis("off")
    a.add_patch(Rectangle((0,0),1,1,fc="#f4f7fb",ec=TS,lw=1.6,zorder=1))
    for c in [(0.30,0.62),(0.62,0.68),(0.48,0.30),(0.78,0.34)]:
        a.add_patch(Circle(c,0.055,fc=BLUE,ec=SURF,lw=1.2,zorder=3))
    a.plot(0.12,0.86,marker="*",ms=15,color=ORANGE,zorder=4)
    a.text(0.155,0.885,"dose source $\\mathbf{x}_0$",fontsize=7.4,color=ORANGE,fontweight="bold")
    for c in [(0.30,0.62),(0.62,0.68)]:
        a.add_patch(FancyArrowPatch((0.14,0.84),(c[0]-0.03,c[1]+0.03),arrowstyle="-|>",
            mutation_scale=9,color=MUT,lw=1.0,ls=(0,(3,2)),zorder=2))
    a.text(0.30,0.545,"$\\mathcal{U}_j$",ha="center",fontsize=7.6,color=BLUE,fontweight="bold")
    a.text(0.5,-0.075,"reflecting boundary $\\partial\\Omega$",ha="center",fontsize=7.4,color=TS)
    a.set_xlim(-0.05,1.05); a.set_ylim(-0.13,1.05)
    pan(a,"A","Chamber and dose source","$N$ small well-separated organoids, $\\varepsilon\\ell_j$ radius")
    # B boundary
    b=fig.add_subplot(gs[0,1]); b.set_aspect("equal"); b.axis("off")
    b.add_patch(Circle((0.30,0.5),0.26,fc="#dce9f9",ec=BLUE,lw=2,zorder=1))
    b.text(0.30,0.5,"organoid\ninterior",ha="center",va="center",fontsize=7.4,color=BLUE,linespacing=1.5)
    ar=lambda p,q,c,st: b.add_patch(FancyArrowPatch(p,q,arrowstyle="-|>",mutation_scale=10,color=c,lw=1.6,
        connectionstyle=st,zorder=4))
    ar((1.02,0.42),(0.56,0.42),TS,"arc3,rad=0.0"); b.text(1.04,0.455,"$\\kappa_{jm}$  adsorption",fontsize=7.4,color=TS,ha="right")
    ar((0.58,0.40),(0.90,0.24),ORANGE,"arc3,rad=0.15"); b.text(0.92,0.20,"$\\gamma^d_{jm}$  desorption",fontsize=7.4,color=ORANGE,ha="right",fontweight="bold")
    ar((0.62,0.86),(0.40,0.66),AQUA,"arc3,rad=-0.2"); b.text(0.64,0.90,"$\\bar\\gamma_{jm}$  internalisation",fontsize=7.4,color=AQUA,fontweight="bold",ha="left")
    b.text(0.5,-0.06,"filter  $g_{jm}(s)=(s+\\bar\\gamma_{jm})/(s+\\gamma^d_{jm}+\\bar\\gamma_{jm})$",
           ha="center",fontsize=7.4,color=TP)
    b.set_xlim(-0.02,1.30); b.set_ylim(-0.12,1.0)
    pan(b,"B","The partially accessible boundary","adsorption, desorption and internalisation are separate")
    # C cascade
    c=fig.add_subplot(gs[1,0]); c.axis("off")
    xsn=[0.10,0.38,0.66]; lbl=["$\\upsilon_{\\mathrm{pr}}$","$\\upsilon_{\\mathrm{df}}$","$\\upsilon_{\\mathrm{mt}}$"]
    for x,l,col in zip(xsn,lbl,RAMP[1:4]):
        c.add_patch(FancyBboxPatch((x,0.52),0.18,0.20,boxstyle="round,pad=0.012",fc=col,ec="none",zorder=3))
        c.text(x+0.09,0.62,l,ha="center",va="center",fontsize=10,color="white",fontweight="bold",zorder=4)
    for x in xsn[:2]:
        c.add_patch(FancyArrowPatch((x+0.185,0.62),(x+0.275,0.62),arrowstyle="-|>",mutation_scale=11,color=TS,lw=1.4))
    c.add_patch(FancyBboxPatch((0.10,0.10),0.30,0.18,boxstyle="round,pad=0.012",fc="#f0f0ec",ec=MUT,zorder=3))
    c.text(0.25,0.19,"$\\upsilon_{\\mathrm{ap}}$  apoptosis",ha="center",va="center",fontsize=8,color=TP,zorder=4)
    c.add_patch(FancyBboxPatch((0.50,0.10),0.34,0.18,boxstyle="round,pad=0.012",fc="#fde8de",ec=ORANGE,zorder=3))
    c.text(0.67,0.19,"$\\upsilon_{\\mathrm{ds}}$  disease",ha="center",va="center",fontsize=8,color=ORANGE,fontweight="bold",zorder=4)
    c.add_patch(FancyArrowPatch((0.42,0.90),(0.42,0.75),arrowstyle="-|>",mutation_scale=11,color=AQUA,lw=1.8))
    c.text(0.44,0.90,"productive flux  $\\mathcal{J}_{jm}(t)$",fontsize=8,color=AQUA,fontweight="bold",va="top")
    c.add_patch(FancyArrowPatch((0.60,0.86),(0.70,0.30),arrowstyle="-|>",mutation_scale=11,color=AQUA,lw=1.4,
        connectionstyle="arc3,rad=0.25"))
    c.text(0.86,0.55,"rescue",fontsize=7.4,color=AQUA,fontweight="bold")
    c.set_xlim(0,1.02); c.set_ylim(0,1.0)
    pan(c,"C","Productive uptake drives the state","not extracellular exposure")
    # D variability
    d=fig.add_subplot(gs[1,1]); d.axis("off")
    lv=[("line","#b7d3f6",0.80),("batch","#6da7ec",0.55),("organoid","#2a78d6",0.30)]
    for i,(nm,col,y) in enumerate(lv):
        d.add_patch(FancyBboxPatch((0.06,y),0.30,0.15,boxstyle="round,pad=0.012",fc=col,ec="none",zorder=3))
        d.text(0.21,y+0.075,nm,ha="center",va="center",fontsize=7.8,color="white" if i else TP,fontweight="bold",zorder=4)
        if i<2: d.add_patch(FancyArrowPatch((0.21,y-0.005),(0.21,y-0.10),arrowstyle="-|>",mutation_scale=9,color=MUT,lw=1.1))
    d.add_patch(FancyArrowPatch((0.375,0.375),(0.52,0.375),arrowstyle="-|>",mutation_scale=11,color=TS,lw=1.4))
    d.text(0.55,0.53,"$\\theta_{bj}$",fontsize=9.5,color=TP,fontweight="bold")
    d.text(0.55,0.40,"transport $\\to$ phenotype",fontsize=7.4,color=TS)
    d.text(0.55,0.29,"$\\to\\ \\mathcal{R}_{\\mathrm{GMP}}$",fontsize=8.4,color=ORANGE,fontweight="bold")
    d.set_xlim(0,1.02); d.set_ylim(0.16,1.02)
    pan(d,"D","Nested variability propagates to release","line / batch / process random effects")
    fig.savefig(OUT / "fig1_schematic.pdf", bbox_inches="tight"); plt.close(fig)
    print("  -> outputs/fig1_schematic.pdf")

if __name__ == "__main__":
    banner("Figure 1 -- model schematic")
    style.use()
    main()
