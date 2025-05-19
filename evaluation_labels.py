import torch
import torch.nn.functional as F
import os
import json
from tools.roc import ROC

def evaluate_db(db_path, embedding_path):
    if not os.path.exists(db_path): 
        print(f"{db_path} does not exist.")
        return
    with open(db_path, 'r') as f:
        all_data = json.load(f)[:5000]
    i = 0
    roc_caption = ROC()
    roc_prompt_caption = ROC()
    roc_attr = ROC()
    print("start evaluate")
    datas = []
    for item in all_data:
        file_path = item['file_path']
        attributes = item['attributes']
        captions = item['captions']
        prompt_caption = item['prompt_caption']
        person_id = item['id']
        id_name, fn_name = os.path.split(file_path)[-2:]
        fn_name = f"{person_id}/{id_name}_{fn_name}"
        out_fn = os.path.join(embedding_path, file_path + ".bin")
        caption = captions[0]
        if os.path.exists(out_fn):
            data = torch.load(out_fn)
            image_embedding = data['image_embedding'].float().cpu()
            text_embedding_attr = data['text_embedding_attr'].float().cpu()
            text_embedding_caption = data['text_embedding_captions'][:1].float().cpu()
            text_embedding_prompt_caption = data['text_embedding_prompt_caption'].float().cpu()
            score = (image_embedding @ text_embedding_caption.T).item()
            roc_caption.add_name(fn_name, fn_name+"_caption", score)
            score = (image_embedding @ text_embedding_prompt_caption.T).item()
            roc_prompt_caption.add_name(fn_name, fn_name+"_prompt_caption", score)
            score = (image_embedding @ text_embedding_attr.T).item()
            roc_attr.add_name(fn_name, fn_name+"_attr", score)
            for person_id2, f2, data2 in datas:
                if person_id2 == person_id:
                    continue
                score = (image_embedding @ data2['text_embedding_captions'].T).item()
                roc_caption.add_name(fn_name, f2+"_caption", score)
                score = (image_embedding @ data2['text_embedding_prompt_caption'].T).item()
                roc_prompt_caption.add_name(fn_name, f2+"_prompt_caption", score)
                score = (image_embedding @ data2['text_embedding_attr'].T).item()
                roc_attr.add_name(fn_name, f2+"_attr", score)
            datas.append((person_id, fn_name, {
                "text_embedding_captions": text_embedding_caption,
                "text_embedding_prompt_caption": text_embedding_prompt_caption,
                "text_embedding_attr": text_embedding_attr
            }))
            i += 1
            print(f"{i}: {file_path} -> {out_fn}")
    roc_caption.stat(1)
    roc_prompt_caption.stat(1)
    roc_attr.stat(1)
    
        
if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--db_path", type=str, default="SYNTH-PEDES/synthpedes-dataset.json")
    parser.add_argument("--embedding_path", type=str, default="embeddings")
    args = parser.parse_args()
    evaluate_db(args.db_path, args.embedding_path)

"""
roc_caption.stat(1)
        EER      FAR=1e-3  FAR=1e-4  FAR=1e-5  FAR=1e-6  FAR=0
----------------------------------------------------------------
FRR    12.2132%  90.5400%  98.5000%  99.9200%  100.0000%  100.0000%
SCORE   0.0641772 0.153719  0.180852  0.201203  0.213413  0.217483
Verify Count=12466498. Score in [ 0 - 0 ]
Top-15 False Rejected:
         25/Part1/25_3.jpg 25/Part1/25_3.jpg_caption -0.070696622133255
         39/Part1/39_3.jpg 39/Part1/39_3.jpg_caption -0.04543854668736458
         246/Part1/246_4.jpg 246/Part1/246_4.jpg_caption -0.0404655858874321
         24/Part1/24_0.jpg 24/Part1/24_0.jpg_caption -0.030871551483869553
         358/Part1/358_17.jpg 358/Part1/358_17.jpg_caption -0.029601838439702988
         149/Part1/149_12.jpg 149/Part1/149_12.jpg_caption -0.027269314974546432
         39/Part1/39_1.jpg 39/Part1/39_1.jpg_caption -0.02673865668475628
         174/Part1/174_7.jpg 174/Part1/174_7.jpg_caption -0.024261921644210815
         19/Part1/19_0.jpg 19/Part1/19_0.jpg_caption -0.02111504040658474
         19/Part1/19_2.jpg 19/Part1/19_2.jpg_caption -0.02033238485455513
         39/Part1/39_2.jpg 39/Part1/39_2.jpg_caption -0.017875097692012787
         25/Part1/25_7.jpg 25/Part1/25_7.jpg_caption -0.017572619020938873
         39/Part1/39_5.jpg 39/Part1/39_5.jpg_caption -0.01609090343117714
         149/Part1/149_8.jpg 149/Part1/149_8.jpg_caption -0.014132577925920486
         347/Part1/347_4.jpg 347/Part1/347_4.jpg_caption -0.013866718858480453
Top-15 False Accepted:
         354/Part1/354_5.jpg 339/Part1/339_7.jpg_caption 0.21786625683307648
         354/Part1/354_5.jpg 107/Part1/107_3.jpg_caption 0.21781933307647705
         354/Part1/354_5.jpg 342/Part1/342_2.jpg_caption 0.21781933307647705
         354/Part1/354_5.jpg 342/Part1/342_4.jpg_caption 0.21781933307647705
         354/Part1/354_5.jpg 342/Part1/342_5.jpg_caption 0.21781933307647705
         350/Part1/350_15.jpg 345/Part1/345_5.jpg_caption 0.21439039707183838
         350/Part1/350_15.jpg 345/Part1/345_3.jpg_caption 0.21404986083507538
         350/Part1/350_15.jpg 342/Part1/342_6.jpg_caption 0.2136557549238205
         350/Part1/350_15.jpg 339/Part1/339_7.jpg_caption 0.21331331133842468
         350/Part1/350_15.jpg 107/Part1/107_3.jpg_caption 0.2129790037870407
         350/Part1/350_15.jpg 342/Part1/342_2.jpg_caption 0.2129790037870407
         350/Part1/350_15.jpg 342/Part1/342_4.jpg_caption 0.2129790037870407
         350/Part1/350_15.jpg 342/Part1/342_5.jpg_caption 0.2129790037870407
         350/Part1/350_4.jpg 345/Part1/345_5.jpg_caption 0.21262650191783905
         350/Part1/350_15.jpg 345/Part1/345_9.jpg_caption 0.2125965803861618

roc_prompt_caption.stat(1)
        EER      FAR=1e-3  FAR=1e-4  FAR=1e-5  FAR=1e-6  FAR=0
----------------------------------------------------------------
FRR    13.4707%  89.6000%  97.9600%  99.7400%  99.9200%  100.0000%
SCORE   0.0646558 0.152181  0.176419  0.196617  0.211429  0.219508
Verify Count=12466498. Score in [ 0 - 0 ]
Top-15 False Rejected:
         352/Part1/352_2.jpg 352/Part1/352_2.jpg_prompt_caption -0.06046738103032112
         25/Part1/25_3.jpg 25/Part1/25_3.jpg_prompt_caption -0.05291982740163803
         19/Part1/19_9.jpg 19/Part1/19_9.jpg_prompt_caption -0.04820236936211586
         69/Part1/69_1.jpg 69/Part1/69_1.jpg_prompt_caption -0.02881758101284504
         334/Part1/334_9.jpg 334/Part1/334_9.jpg_prompt_caption -0.024562593549489975
         355/Part1/355_16.jpg 355/Part1/355_16.jpg_prompt_caption -0.02302488684654236
         25/Part1/25_1.jpg 25/Part1/25_1.jpg_prompt_caption -0.021676283329725266
         149/Part1/149_7.jpg 149/Part1/149_7.jpg_prompt_caption -0.019634250551462173
         244/Part1/244_5.jpg 244/Part1/244_5.jpg_prompt_caption -0.016294926404953003
         39/Part1/39_3.jpg 39/Part1/39_3.jpg_prompt_caption -0.01612282544374466
         358/Part1/358_17.jpg 358/Part1/358_17.jpg_prompt_caption -0.0156002938747406
         172/Part1/172_0.jpg 172/Part1/172_0.jpg_prompt_caption -0.01370699517428875
         246/Part1/246_5.jpg 246/Part1/246_5.jpg_prompt_caption -0.013437265530228615
         130/Part1/130_15.jpg 130/Part1/130_15.jpg_prompt_caption -0.0133988531306386
         244/Part1/244_4.jpg 244/Part1/244_4.jpg_prompt_caption -0.011119551956653595
Top-15 False Accepted:
         350/Part1/350_15.jpg 286/Part1/286_7.jpg_prompt_caption 0.21985231339931488
         354/Part1/354_5.jpg 339/Part1/339_7.jpg_prompt_caption 0.2187100350856781
         350/Part1/350_4.jpg 286/Part1/286_7.jpg_prompt_caption 0.21727418899536133
         342/Part1/342_16.jpg 286/Part1/286_7.jpg_prompt_caption 0.2172427475452423
         112/Part1/112_12.jpg 34/Part1/34_6.jpg_prompt_caption 0.21456408500671387
         350/Part1/350_9.jpg 342/Part1/342_7.jpg_prompt_caption 0.2127760499715805
         342/Part1/342_2.jpg 286/Part1/286_7.jpg_prompt_caption 0.2127239853143692
         350/Part1/350_12.jpg 286/Part1/286_7.jpg_prompt_caption 0.2126401662826538
         350/Part1/350_0.jpg 286/Part1/286_7.jpg_prompt_caption 0.21257425844669342
         350/Part1/350_9.jpg 286/Part1/286_7.jpg_prompt_caption 0.212398499250412
         350/Part1/350_15.jpg 342/Part1/342_5.jpg_prompt_caption 0.21218252182006836
         350/Part1/350_10.jpg 286/Part1/286_7.jpg_prompt_caption 0.21110492944717407
         133/Part1/133_8.jpg 34/Part1/34_7.jpg_prompt_caption 0.21093860268592834
         354/Part1/354_18.jpg 286/Part1/286_7.jpg_prompt_caption 0.2108091562986374
         350/Part1/350_2.jpg 286/Part1/286_7.jpg_prompt_caption 0.20941327512264252

roc_attr.stat(1)
        EER      FAR=1e-3  FAR=1e-4  FAR=1e-5  FAR=1e-6  FAR=0
----------------------------------------------------------------
FRR    14.2411%  90.2200%  98.2600%  99.7000%  99.9400%  99.9600%
SCORE   0.0750233 0.162455  0.186667  0.201463  0.210878  0.220294
Verify Count=12466498. Score in [ 0 - 0 ]
Top-15 False Rejected:
         352/Part1/352_2.jpg 352/Part1/352_2.jpg_attr -0.05064548924565315
         25/Part1/25_3.jpg 25/Part1/25_3.jpg_attr -0.04942462220788002
         362/Part1/362_0.jpg 362/Part1/362_0.jpg_attr -0.04157068952918053
         19/Part1/19_9.jpg 19/Part1/19_9.jpg_attr -0.04149693250656128
         234/Part1/234_4.jpg 234/Part1/234_4.jpg_attr -0.026700317859649658
         249/Part1/249_6.jpg 249/Part1/249_6.jpg_attr -0.018901512026786804
         244/Part1/244_6.jpg 244/Part1/244_6.jpg_attr -0.016156725585460663
         20/Part1/20_15.jpg 20/Part1/20_15.jpg_attr -0.015324441716074944
         244/Part1/244_5.jpg 244/Part1/244_5.jpg_attr -0.012067310512065887
         362/Part1/362_12.jpg 362/Part1/362_12.jpg_attr -0.011860577389597893
         25/Part1/25_1.jpg 25/Part1/25_1.jpg_attr -0.011673148721456528
         5/Part1/5_4.jpg 5/Part1/5_4.jpg_attr -0.00994056649506092
         174/Part1/174_7.jpg 174/Part1/174_7.jpg_attr -0.0092778280377388
         347/Part1/347_1.jpg 347/Part1/347_1.jpg_attr -0.008618065156042576
         244/Part1/244_4.jpg 244/Part1/244_4.jpg_attr -0.007541447877883911
Top-15 False Accepted:
         258/Part1/258_10.jpg 120/Part1/120_2.jpg_attr 0.2198067158460617
         258/Part1/258_11.jpg 120/Part1/120_2.jpg_attr 0.21729062497615814
         350/Part1/350_15.jpg 342/Part1/342_2.jpg_attr 0.21602441370487213
         107/Part1/107_8.jpg 102/Part1/102_6.jpg_attr 0.2125379741191864
         350/Part1/350_17.jpg 86/Part1/86_4.jpg_attr 0.21218717098236084
         350/Part1/350_15.jpg 339/Part1/339_8.jpg_attr 0.21106484532356262
         350/Part1/350_15.jpg 339/Part1/339_11.jpg_attr 0.21106484532356262
         350/Part1/350_15.jpg 339/Part1/339_14.jpg_attr 0.21106484532356262
         350/Part1/350_15.jpg 342/Part1/342_5.jpg_attr 0.21106484532356262
         350/Part1/350_4.jpg 342/Part1/342_2.jpg_attr 0.2108759582042694
         98/Part1/98_0.jpg 93/Part1/93_7.jpg_attr 0.21073894202709198
         350/Part1/350_12.jpg 345/Part1/345_11.jpg_attr 0.2102069854736328
         350/Part1/350_9.jpg 342/Part1/342_0.jpg_attr 0.20987683534622192
         350/Part1/350_9.jpg 342/Part1/342_9.jpg_attr 0.20987683534622192
         350/Part1/350_9.jpg 86/Part1/86_4.jpg_attr 0.20946341753005981
"""                 