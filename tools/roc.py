import os
import sys
import math
import torch
#import gnuplot

def isSameFile(f1, f2, sp="/"):
    d1,fn1 = os.path.split(f1) 
    d2,fn2 = os.path.split(f2) 
    if d1=="": d1, fn1 = f1.split(sp, 1)
    if d2=="": d2, fn2 = f2.split(sp, 1)
    return d1==d2


class ROC():
    def __init__(self, sp=None, bad_count=15, displayInterval=100000):
        self.frr={}
        self.far={}
        self.cache_pos, self.cache_neg = [], []
        self.minv=10000000000
        self.maxv=-10000000000
        self.max_far=-1000000000
        self.count = 0
        self.displayInterval = displayInterval
        self.bad_count = bad_count
        self.badFR=[]
        self.badFA=[]
        self.min_count=80000
        self.warning_far = 328
        self.offset = None
        self.scale = None
        self.eer_score = None
        self.sp = sp or "/"
        self.count = 0
    def statROC(self, frr_, far_, minv, maxv, max_far, show=0):
        show=show or 0
        frr, far={}, {}
        for k, v in frr_.items(): frr[k]=v
        for k, v in far_.items(): far[k]=v
        frr[maxv+1], far[maxv+1], frr[minv-1], far[minv-1] = 0,0,0,0
        if show>1: print(minv, maxv, max_far)
        for i in range(minv, maxv+1):
            frr[i]=frr[i-1]+(frr[i] if i in frr else 0)
    
        for i in range(maxv, minv-1, -1):
            far[i]=(far[i] if i in far else 0)+far[i+1]
    
        if show>1: print("SCORE","FRR","FAR")
        err, nofar, far3, far4, far5, far6, lastfrr=None, None, None, None, None, None, 100
        err_score, nofar_score, far3_score, far4_score, far5_score, far6_score=max_far,max_far,max_far,max_far,max_far,max_far
        print(minv, maxv)
        for i in range(minv, maxv+1):
#            if i in frr: print('frr',i,'is null')
#            if i in far: print('far',i,'is null')
            frrp, farp=frr[i]*100/frr[maxv], far[i]*100/far[minv]
            if show>1: print("%d\t%0.4f\t%0.8f\t%d"%(i, frrp, farp, far[i]))
            if i>minv+3: 
                if far[i-3]==0: break
            if frrp>farp:
                if not err:
                    err=(frrp+lastfrr+lastfar+farp)/4
                    err_score=i
                elif far[i]*1e3<=far[minv]:
                    if far3 is None: far3=frrp; far3_score=i
                    if far[i]*1e4<=far[minv]:
                        if far4 is None: far4=frrp; far4_score=i
                        if far[i]*1e5<=far[minv]:
                            if far5 is None: far5=frrp; far5_score=i
                            if far[i]*1e6<=far[minv]:
                                if far6 is None: far6=frrp; far6_score=i
                                if far[i]==0:
                                    if nofar is None: nofar=frrp; nofar_score=i
            lastfrr=frrp
            lastfar=farp
        self.eer, self.far3, self.far4, self.far5, self.far6, \
            self.nofar, self.eer_score, self.nofar_score, \
            self.far3_score, self.far4_score, self.far5_score, self.far6_score = \
                err, far3, far4, far5, far6, nofar, err_score, nofar_score, far3_score, far4_score, far5_score, far6_score

    def stat(self, show=0):
        if (self.offset==None and self.scale==None):
            if not self.calcInput(): return
    
        self.statROC(self.frr, self.far, self.minv, self.maxv, self.max_far, show)
        if show>0: self.show()

    def show(self):
        if not self.eer_score: self.stat(0)
        self.offset=self.offset or 0
        self.scale=self.scale or 1
        self.eer_score, self.far3_score, self.far4_score, self.far5_score, self.far6_score, self.nofar_score = \
            (self.eer_score-200)/self.scale+self.offset, (self.far3_score-200)/self.scale+self.offset, (self.far4_score-200)/self.scale+self.offset, \
            (self.far5_score-200)/self.scale+self.offset, (self.far6_score-200)/self.scale+self.offset, (self.nofar_score-200)/self.scale+self.offset
        self.nofar=self.nofar or 100
        self.far6=self.far6 or self.nofar
        self.far5=self.far5 or self.far6
        self.far4=self.far4 or self.far5
        self.far3=self.far3 or self.far4
        print("        EER      FAR=1e-3  FAR=1e-4  FAR=1e-5  FAR=1e-6  FAR=0")
        print("----------------------------------------------------------------")
        print("FRR    %7.4f%%  %7.4f%%  %7.4f%%  %7.4f%%  %7.4f%%  %7.4f%%"%(
                    self.eer or 100, self.far3, self.far4, self.far5, self.far6, self.nofar))
        print("SCORE   %-8g %-8g  %-8g  %-8g  %-8g  %-8g"%(
                    self.eer_score, self.far3_score, self.far4_score, self.far5_score, self.far6_score, self.nofar_score))
        print("Verify Count=%d. Score in [ %d - %d ]"%(self.count,
            (self.minv-200)/self.scale+self.offset, (self.maxv-200)/self.scale+self.offset))

        if len(self.badFR)>0:
            print('Top-'+str(len(self.badFR))+' False Rejected:')
            for i in range(len(self.badFR),0,-1): 
                print('\t',self.badFR[i-1][0], self.badFR[i-1][1],self.badFR[i-1][2])
    
        if len(self.badFA)>0:
            print('Top-'+str(len(self.badFA))+' False Accepted:')
            for i in range(len(self.badFA),0,-1): 
                print('\t',self.badFA[i-1][0], self.badFA[i-1][1],self.badFA[i-1][2])

    def input(self, fn=None):
        self.cache_pos, self.cache_neg = [], []
        def proc(line):
            if line[0]=='\t':
                items = line.split('\t')
                if items[0]=="" and len(items)==4:
                    p, f1, f2, score = items                
                    score=float(score)
                    max_far_line, warning = self.add_name(f1, f2, score)
    #                 if max_far_line: print(line)
    #                 if warning: print(line)            
                    if self.count%self.displayInterval==0: self.stat(1)
            elif line[0]=='#':
                print(line)
        if fn is not None:
            f=open(fn,'r')
            for line in f.readlines(): proc(line)
        else:
            for line in sys.stdin:
                if line is None: break
                proc(line)
        self.stat(1)
        #self.draw('score.png')
        #self.roc()


    def add_(self, iss, score):
        warning=False; max_far_line=False
        score=math.ceil(score)
        if score<self.minv:  self.minv=score
        if score>self.maxv:  self.maxv=score
        self.count = self.count+1
        if iss:
            self.frr[score]=(self.frr[score] if score in self.frr else 0) +1
        else:
            self.far[score]=(self.far[score] if score in self.far else 0) +1
            if score>=self.warning_far:
                warning=True
            elif (score>=self.max_far-5 and self.count>20000) or (score>=self.max_far and self.count>4000):
                warning=True
        
            if score>self.max_far:
                self.max_far=score
                max_far_line=True
    
        return max_far_line, warning


    def calcInput(self):
        cache_pos=self.cache_pos or []
        cache_neg=self.cache_neg or []
        mean_pos, mean_neg = 0, 0
        for i in range(len(cache_pos)): mean_pos=mean_pos+cache_pos[i]
        for i in range(len(cache_neg)): mean_neg=mean_neg+cache_neg[i]
        if len(cache_pos)==0 or len(cache_neg)==0:
            print((len(cache_pos)==0) and "No False Reject! " or "No False Accept! ", "Cannot calculate.")
            self.offset=0
            self.scale=1
            return False
    
        mean_pos=mean_pos/len(cache_pos); mean_neg=mean_neg/len(cache_neg)
        print(f'score: {mean_neg}[negative] - {mean_pos}[positive]')
        self.offset=mean_neg
        self.scale=800/(mean_pos-mean_neg+1)
        self.count=0
        for i in range(len(cache_pos)): self.add_(True, math.floor((cache_pos[i]-self.offset)*self.scale+200))
        for i in range(len(cache_neg)): self.add_(False, math.floor((cache_neg[i]-self.offset)*self.scale)+200)
        return True


    def add(self, iss, score):
        if self.count <= self.min_count:
            cache_pos=self.cache_pos or []
            cache_neg=self.cache_neg or []
            if iss:
                cache_pos.append(score)
            else:
                cache_neg.append(score)
        
            self.cache_neg=cache_neg; self.cache_pos=cache_pos
            if self.count >= self.min_count:
                self.calcInput()
        
            self.count=self.count+1
            return False, False
        else:
            return self.add_(iss, math.floor((score-self.offset)*self.scale+200))

    def add_name(self, f1, f2, score, iss=None):
        if iss is None: iss=isSameFile(f1, f2, self.sp)
        if iss:
            if len(self.badFR)<self.bad_count:
                self.badFR.append([f1, f2, score])
            elif score<self.badFR[0][2]:
                self.badFR[0]=[f1, f2, score]
            self.badFR.sort(key=lambda x: -x[2])
        else:
            if len(self.badFA)<self.bad_count:
                self.badFA.append([f1, f2, score])
            elif score>self.badFA[0][2]:
                self.badFA[0]=[f1, f2, score]
            self.badFA.sort(key=lambda x: x[2])
        return self.add(iss, score)


    def roc(self, fns=None):
        import numpy as np
        import matplotlib.pyplot as plot
        frr, far={}, {}
        for k, v in self.frr.items(): frr[k]=v or 0
        for k, v in self.far.items(): far[k]=v or 0
        minv, maxv=self.minv, self.maxv
        frr[maxv+1], far[maxv+1], frr[minv-1], far[minv-1]=1e-10,1e-10,1e-10,1e-10
        for i in range(minv, maxv+1):
            frr[i]=frr[i-1]+(frr[i] if i in frr else 0)
    
        for i in range(maxv, minv-1, -1):
            far[i]=(far[i] if i in far else 0)+far[i+1]
    
        for i in range(minv, maxv+1):
            frr[i]=frr[i]/frr[maxv]
    
        for i in range(maxv, minv-1, -1):
            far[i]=far[i]/far[minv]
    
        self.frr_={}
        self.far_={}
        for i in range(maxv+1-(minv-1)+1):
            self.frr_[i]=frr[i+minv-1]
            self.far_[i]=far[i+minv-1]
    
        frr=self.frr_
        far=self.far_
        self.offset=minv-1
        frr=list(frr.values())[1:-1]
        far=list(far.values())[1:-1]
    #     print('frr', frr)
    #     print('far', far)
        y1=(torch.log(torch.Tensor(frr))/2.302585092994).numpy()
        y2=(torch.log(torch.Tensor(far))/2.302585092994).numpy()

        x=np.linspace(0,len(frr),len(frr))     
        plot.figure()   
        plot.plot(x, y1, 'red', 'FRR')
        plot.plot(x, y2, 'blue', 'FAR')
        plot.ylim((-6, 0))
        plot.ylim((0, 800))
        plot.yticks(np.linspace(-8, 0, 8))
        plot.title('ROC')
        plot.savefig("roc.png")
    
        fns=fns or {"det":'det.png', "frrfar":'frr-far.png', "roc":'roc.png'}
        if type(fns)=='string': fns={"roc":fns}
        if "det" in fns: 
            gnuplot.pngfigure(fns.det)
            gnuplot.axis({-6,0,-6,0})
            gnuplot.plot({y2, y1, '-'},{torch.linspace(-6,0), torch.linspace(-6,0), '-'})
            gnuplot.grid(true)
            gnuplot.xlabel('Log of FAR')
            gnuplot.ylabel('Log of FRR')
            gnuplot.title(string.format('DET Curve [ EER=%g%% ]',self.eer))
            gnuplot.plotflush()
            gnuplot.close()
            print('save det image to:', fns.det)
    
        if "frrfar" in fns:
            gnuplot.pngfigure(fns.frrfar)
            gnuplot.axis({10, maxv-10, -6, 0})
            gnuplot.plot(
                {'Log of FRR', y1, '-'},
                {'Log of FAR', y2, '-'})
            gnuplot.grid(true)
            offset='- '+str(self.offset)
            if self.offset<0: offset='+ '+str(0-self.offset)
            gnuplot.xlabel('Threshold: (score '+str(offset)+') x '+str(self.scale)+' + 200')
            gnuplot.title('FRR/FAR Curve')
            gnuplot.plotflush()
            gnuplot.close()
            print('save far/frr image to:', fns.frrfar)
    
        if fns.roc:
            y1=-torch.Tensor(frr)+1
            y2=torch.log(torch.Tensor(far))/2.302585092994
            gnuplot.pngfigure(fns.roc)
            gnuplot.axis({-7,0,0,0.9999})
            gnuplot.plot({y2, y1, '-'})
            gnuplot.xlabel('Log of False Accept Rate')
            gnuplot.ylabel('True Accept Rate')
            gnuplot.grid(true)
            gnuplot.title('Receiver Operating Characterictic Curve')
            gnuplot.plotflush()
            gnuplot.close()
            print('save roc image to:', fns.roc)

    def draw(self, fn):
        height, width=600, self.maxv-self.minv+21
        I=torch.Tensor(3, height+40, width)
        I.fill_(0)
        p, h=1, 0
        maxc=0
        return
        for i in range(self.minv, self.maxv+1):
    #         if (self.far[i] or 0)>maxc: maxc=self.far[i]
            if (self.frr[i] if i in self.frr else 0)>maxc: maxc=self.frr[i]
    
        for i in range(self.minv-10, self.maxv+11):
            h=height-math.floor(height*(self.far[i] or 0)/maxc)+20
            if h<1: h=1
            if h<height+20: I[{{1,1}, {h, height+19},{p, p}}]=1
            h=height-floor(height*(self.frr[i] or 0)/maxc)+20
            if h<1: h=1
            if h<height+20: I[{{2,2}, {h, height+19},{p, p}}]=1
            p=p+1
    
        fn=fn or "scores.jpg"
        image.save(fn, I)
        print('save score image to:', fn)
        if itorch: itorch.image(I)
        image.display(I)

if __name__ == "__main__":
    roc = ROC(displayInterval=1e20, sp='_')
    if len(sys.argv)>1:
        roc.input(sys.argv[1])
    else:
        roc.input()
    print(roc.eer, roc.far3 or 100, roc.far4 or 100, roc.far5 or 100, roc.far6 or 100, roc.nofar or 100)
