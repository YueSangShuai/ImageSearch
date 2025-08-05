# 属性列表（57个）
labels = [
    'Female',
    'AgeOver60', 'Age18-60', 'AgeLess18',
    'Front', 'Side', 'Back',
    'Hat', 'Glasses',
    'HandBag', 'ShoulderBag', 'Backpack', 'HoldObjectsInFront',
    'ShortSleeve', 'LongSleeve', 'UpperStride',
    'UpperLogo', 'UpperPlaid', 'UpperSplice', 
    'LowerStripe', 'LowerPattern',
    'LongCoat', 'Trousers', 'Shorts', 'Skirt&Dress',
    'boots',
    'ublack', 'ugray', 'ublue', 'ugreen','uwhite', 'upurple', 'ured', 'ubrown', 'uyellow', 'upink', 'uorange', 'ubeige', 'ustriped_color', 'umulticolor',
    'lwhite', 'lpink', 'lred', 'lgreen','lyellow', 'lpurple', 'lbrown', 'lblack', 'lorange', 'lblue', 'lgray', 'lbeige', 'lstriped_color', 'lmulticolor'
]

def parse_and_save(input_path, output_path):
    with open(input_path, 'r', encoding='utf-8') as f_in, open(output_path, 'w', encoding='utf-8') as f_out:
        for line in f_in:
            parts = line.strip().split()
            if not parts or len(parts) != 1 + len(labels):
                continue  # 跳过无效行
            filename = parts[0]
            values = list(map(int, parts[1:]))
            active_labels = [label for label, v in zip(labels, values) if v == 1]
            line_out = filename + ' ' + ' '.join(active_labels)
            f_out.write(line_out + '\n')

# 示例使用
parse_and_save('/data/yuesang/LLM/contrastors/data/pa100k_label/label/check/checked_test_lb.txt', '/data/yuesang/LLM/contrastors/data/pa100k_label/label/yinshe/checked_test_lb.txt')
parse_and_save('/data/yuesang/LLM/contrastors/data/pa100k_label/label/check/checked_train_lb.txt', '/data/yuesang/LLM/contrastors/data/pa100k_label/label/yinshe/checked_train_lb.txt')
parse_and_save('/data/yuesang/LLM/contrastors/data/pa100k_label/label/check/checked_val_lb.txt', '/data/yuesang/LLM/contrastors/data/pa100k_label/label/yinshe/checked_val_lb.txt')
