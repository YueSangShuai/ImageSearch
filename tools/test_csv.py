import csv

<<<<<<< HEAD
IN_CSV  = r"/data/yuesang/LLM/contrastors/src/ckpts/person/pa-100k/emov2-2m/attr_metrics.csv"      # 你的文件
=======
IN_CSV  = r"/data/yuesang/LLM/contrastors/src/ckpts/person/emov2/attr_metrics.csv"      # 你的文件
>>>>>>> parent of f0c4ccb (增加教师蒸馏部分)
OUT_CSV = r"attr_metrics.csv"     # 想保存的结果
K = 10                         # 取最低 K 个属性

with open(IN_CSV, newline='') as f_in, open(OUT_CSV, 'w', newline='') as f_out:
    reader = csv.reader(f_in)
    writer = csv.writer(f_out)

    header = next(reader)                 # 第 1 行 = 字段名
    attr_names = header[1:]               # 去掉 "step"
    writer.writerow(['step', 'avg_score',
                     *[f'lowest{i+1}_attr' for i in range(K)],
                     *[f'lowest{i+1}_score' for i in range(K)]])
    # ↑ 输出列：step、平均分、最低K个属性名、最低K个分值

    for row in reader:
        step = row[0]
        scores = list(map(float, row[1:]))

        avg_score = sum(scores) / len(scores)

        # 取最低 K
        lowest = sorted(zip(scores, attr_names))[:K]   # (score, name)

        # 拆成两列
        low_names  = [name   for score, name in lowest]
        low_scores = [f'{score:.6f}' for score, name in lowest]

        writer.writerow([step, f'{avg_score:.3f}', *low_names, *low_scores])

        # 也可直接打印
        print(f'step {step} | avg={avg_score:.3f} | lowest: {", ".join(f"{n}:{s:.3f}" for s,n in lowest)}')
