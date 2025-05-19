import os
import torch
import torch.nn.functional as F
import time
import torchvision.utils as vutils
import numpy as np

def test_acc(model, test_loader, device="cpu", log_dir="logs", iter=0, max_verify_count=500000):
    if type(model) != list:
        models = [model]
    if device:
        for model in models:  model.to(device)
    for model in models:  model.eval()
    count = len(test_loader.dataset)
    if count > max_verify_count: count = max_verify_count
    classes = test_loader.dataset.classes
    class_cross = torch.zeros((len(classes), len(classes)))
    start_time = time.time()
    print("[test_acc] Start testing accuracy ...")
    c = 0
    for i, (data, target, path, d0) in enumerate(test_loader):
        if device:
            data = data.to(device)
            target = target.to(device)
        probs = None
        for model in models:
            y = model(data, target=target)
            if probs is None:
                probs = y
                if len(data.shape)==4 and c==0 and save_dir: 
                    vutils.save_image(data[:40].data.cpu(), os.path.join(log_dir, f'val_{iter}.png'), nrow=10)
            else:
                probs += y
        if len(models) > 1: probs /= len(models)
        preds = torch.argmax(probs, dim=1)
        for label, pred in zip(target, preds):
            if c==0: print(f'{path[0]}, label: {label.item()}, pred: {pred.item()}, probs: {probs[0].cpu().numpy()}')
            class_cross[label][pred] += 1
            c += 1
        if c >= count: break
    class_cross = class_cross.int().numpy()
    print("confusion matrix:") 
    print(class_cross)
    print("    numbers per class:", np.sum(class_cross, axis=1))
    print("predictions per class:", np.sum(class_cross, axis=0))
    # recall
    recall = np.diag(class_cross) / np.sum(class_cross, axis=1)
    recall = dict(((c, v) for c,v in zip(classes, recall)))
    print("   recall:", " ".join(["%s=%0.4f"%(k,r) for k,r in recall.items()]))
    # precision
    precision = np.diag(class_cross) / np.sum(class_cross, axis=0)
    precision = dict(((c, v) for c,v in zip(classes, precision)))
    print("precision:", " ".join(["%s=%0.4f"%(k,r) for k,r in precision.items()]))
    # accuracy
    accuracy = np.sum(np.diag(class_cross)) / np.sum(class_cross)
    print(" accuracy:", "%0.4f"%accuracy)
    Time=time.time()-start_time
    print("    count:", count, "time: %0.6f seconds."%Time)
    # count: 20000, accuracy: 99.375124975005%, time: 137.18355464935303 second.
    ret = dict(Accuracy=accuracy)
    for k,v in recall.items(): ret[f'Recall_{k}'] = v
    for k,v in precision.items(): ret[f'Precision_{k}'] = v
    return ret

def test_1classifier(model, test_loader, class_index=1, device=None, save_dir='logs', max_verify_count=500000):
    if type(model) != list:
        models = [model]
    if device:
        for model in models:
            model.to(device)
    for model in models:
        model.eval()
    count = len(test_loader.dataset)
    classes = test_loader.dataset.classes
    if count>max_verify_count: count=max_verify_count
    all_features = [None]*len(models)
    all_classes = torch.zeros((count,)).long()
    all_files = []
    c = 0
    start_time = time.time()
    print("[test_1classifier] Start extract features ...")
    for datas in test_loader:
        data, target, path = datas[0], datas[1], datas[2]
        if device:
            data = data.to(device)
        mi = 0
        c1 = c
        c2 = c+target.size(0)
        if c2>count: 
            c2=count
            target = target[:c2-c]
        all_classes[c:c2] = target.view_as(all_classes[c:c2])
        all_files += path
        for model in models:
            features = model(data).cpu()[:,-1:]
            if all_features[mi] is None:
                feature_size = features.size(1)
                all_features[mi] = torch.zeros(count, feature_size)
                if len(data.shape)==4 and c==0 and save_dir: vutils.save_image(data[:40].data.cpu(), os.path.join(save_dir, 'test.png'))
            c1 = c
            for i in range(data.size(0)):
                f = features[i]
                all_features[mi][c1] = f*100
                c1 += 1
                if c1 == 1:
                    print(f"file: {path[0]}, score: {f.item()}, class: {all_classes[0].item()}-{classes[all_classes[0]]}")
                if c1 >= count:  break
            mi += 1
        c = c1
        if c >= count:
            break
    print("{:.5f} ms/sample.".format((time.time()-start_time)*1000/count))
    from .roc import ROC
    r = ROC(displayInterval=1e20)
    start_time = time.time()
    print("Verify test for " + str(c) + " samples ...")
    c2 = classes[class_index]
    mc = len(models)
    for i in range(c):
        c1 = all_files[i]
        ff = sum([f[i] for f in all_features])/mc # average
        c = all_classes[i].item()
        for j in range(ff.size(0)):
            score = ff[j].item()
#            print(c1, c, c2, score)
            r.add_name(c1, c2, score, c==1)
    print("Verify finished. It take {:.2f} second.".format((time.time()-start_time)))
    r.stat(1)
    return dict(EER=r.eer or 100.0, FAR3=r.far3, FAR4=r.far4, FAR5=r.far5, FAR6=r.far6, NOFAR=r.nofar)

def test_verify(model, test_loader, device=None, save_dir='logs', max_verify_count=20000):
    if type(model) == list:
        models = model
    else:
        models = [model]
    if device:
        for model in models:
            model.to(device)
    for model in models:
        model.eval()
    count = len(test_loader.dataset)
    if count>max_verify_count: count=max_verify_count
    all_features = [None]*len(models)
    all_classes = []
    all_files = []
    c = 0
    start_time = time.time()
    print("[test_verify] Start extract features ...")
    for datas in test_loader:
        data, path = datas[0], datas[2]
        if device:
            data = data.to(device)
        mi = 0
        c1 = c
        all_classes += [p.split('/')[-2] + '/' + p.split('/')[-1]
                        for p in path]
        all_files += path
        for model in models:
            features = model(data)
            if len(features) == 2 and type(features) == tuple:
                features = features[0]
            features = F.normalize(features[:].cpu(), dim=1)
            if all_features[mi] is None:
                feature_size = features.size(1)
                all_features[mi] = torch.zeros(count, feature_size)
                if len(data.shape)==4 and c==0 and save_dir: vutils.save_image(data[:40].data.cpu(), os.path.join(save_dir, 'test.png'))
            c1 = c
            for i in range(data.size(0)):
                f = features[i]
                all_features[mi][c1] = f
                c1 += 1
                if c1 == 1:
                    print(all_classes[0], f, f.shape)
                if c1 >= count: break
            mi += 1
        c = c1
        if c >= count:
            break
    print("{:.5f} ms/sample.".format((time.time()-start_time)*1000/count))
    from .roc import ROC
    r = ROC(displayInterval=1e20)
    start_time = time.time()
    print("Verify test for " + str(c) + " samples ...")
    for i in range(c-1):
        c1 = all_classes[i]
        ff = sum([(f[i].unsqueeze(0)*f[1+i:]).sum(dim=1)    # cosine similarity
            for f in all_features])
        for j in range(ff.size(0)):
            score = ff[j].item()
            c2 = all_classes[i+j+1]
            if not c2: break
            #print(c1, c2, score)
            r.add_name(c1, c2, score)
    print("Verify finished. It take {:.2f} second.".format((time.time()-start_time)))
    r.stat(1)
    return dict(EER=r.eer or 100.0, FAR3=r.far3, FAR4=r.far4, FAR5=r.far5, FAR6=r.far6, NOFAR=r.nofar)

def test_classifier(model, test_loader, device=None, save_dir='logs', max_verify_count=500000):
    count = len(test_loader.dataset)
    classes = test_loader.dataset.classes
    if len(classes)>count/2:
        return test_acc(model, test_loader, device=device, log_dir=save_dir, iter=0, max_verify_count=max_verify_count)
    if type(model) != list:
        models = [model]
    if device:
        for model in models:
            model.to(device)
    for model in models:
        model.eval()
    if count>max_verify_count: count=max_verify_count
    all_features = [None]*len(models)
    all_classes = torch.zeros((count,)).long()
    all_files = []
    c = 0
    start_time = time.time()
    print("[test_classifier] Start extract features ...")
    for datas in test_loader:
        data, target, path = datas[0], datas[1], datas[2]
        if device:
            data = data.to(device)
            target = target.to(device)
        mi = 0
        c1 = c
        c2 = c+target.size(0)
        if c2>count: 
            c2=count
            target = target[:c2-c]
        all_classes[c:c2] = target
        all_files += path
        for model in models:
            features = model(data, target=target)
            if len(features) == 2 and type(features) == tuple:
                features = features[0]
#            features = F.softmax(features[:].cpu(), dim=1)
            if all_features[mi] is None:
                feature_size = features.size(1) if len(features.size())>1 else 1
                all_features[mi] = torch.zeros(count, feature_size)
                if len(data.shape)==4 and c==0 and save_dir: vutils.save_image(data[:40].data.cpu(), os.path.join(save_dir, 'test.png'))
            c1 = c
            for i in range(data.size(0)):
                f = features[i]
                all_features[mi][c1] = f
                c1 += 1
                if c1 == 1:
                    print(path[0], all_classes[0], f)
                if c1 >= count:
                    break
            mi += 1
        c = c1
        if c >= count:
            break
    print("{:.5f} ms/sample.".format((time.time()-start_time)*1000/count))
    from .roc import ROC
    r = ROC(displayInterval=1e20)
    start_time = time.time()
    print("Verify test for " + str(c) + " samples ...")
    mc = len(models)
    for i in range(c):
        c1 = all_files[i]
        ff = sum([f[i] for f in all_features])/mc # average
        for j in range(ff.size(0)):
            score = ff[j].item()
            c2 = classes[j]
            #print(c1, c2, score)
            r.add_name(c1, c2, score, all_classes[i]==j)
    print("Verify finished. It take {:.2f} second.".format((time.time()-start_time)))
    r.stat(1)
    return dict(EER=r.eer or 100.0, FAR3=r.far3, FAR4=r.far4, FAR5=r.far5, FAR6=r.far6, NOFAR=r.nofar)

def verify_file(roc, features, fn, ftrs, weights=[1,]*200):
    W = sum(weights[:len(ftrs)])
    for fn_, ftrs2 in features:
        scores = []
        for f1, f2, weight in zip(ftrs, ftrs2, weights):
            scores.append((f1*f2).sum().item()*weight)
        score = sum(scores)/W
        roc.add_name(fn, fn_, score)

def verify_dir(paths, weights=[], roc=None, features=[], root_lens=[]):
    from .roc import ROC
    stat = False
    if roc==None:
        roc = ROC(displayInterval=1e20, bad_count=10)
        weights=[1.0]*len(paths)
        new_paths = []
        for i, p in enumerate(paths):
            if ':' in p:
                p, w = p.split(':')
                weights[i] = float(w)
            new_paths.append(os.path.join(p,""))
        root_lens = [len(p) for p in paths]
        paths = new_paths
        stat = True
        print(f"verify dirs: {paths}, weights={weights}")
    for f in os.listdir(paths[0]):
        fn = os.path.join(paths[0], f)
        if os.path.isdir(fn):
            verify_dir([os.path.join(path, f) for path in paths], weights, roc, features, root_lens)
        elif f.endswith('.bin'):
            ftrs=[]
            for path in paths:
                fn = os.path.join(path, f)
                ftr = torch.from_numpy(np.fromfile(fn, dtype=np.float32))
                ftr = ftr/ftr.norm(2)
                ftrs.append(ftr)
            name = fn[root_lens[0]:]
            verify_file(roc, features, name, ftrs, weights)
            features.append((name, ftrs))
    if stat: 
        roc.stat(1)
        return roc.eer or 100, roc.far3, roc.far4, roc.far5, roc.far6, roc.nofar

if __name__=="__main__":
    import sys
    if len(sys.argv)<2:
        print("1:1 Verification Evaluation. Support merging scores for multiply feature paths.\nUsage: verify.py <dir1> [<dir2>] [<dir3>] ...\n")
        sys.exit(0)
    verify_dir(sys.argv[1:])
