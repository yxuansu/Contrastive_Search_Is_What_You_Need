from pyrouge import Rouge155
def get_one_line_result(line):
    return float(line.strip().split()[3]) * 100
    
def get_rouge_scores(summary_dir, model_dir):
    r = Rouge155()
    r.system_dir = summary_dir
    r.model_dir = model_dir
    r.system_filename_pattern = r'(\d+)_decoded.txt'
    r.model_filename_pattern = r'#ID#_reference.txt'
    output = r.convert_and_evaluate()
    output_list = output.split('\n')
    rogue_1_score = get_one_line_result(output_list[3])
    rogue_2_score = get_one_line_result(output_list[7])
    rogue_l_score = get_one_line_result(output_list[19])
    return rogue_1_score, rogue_2_score, rogue_l_score

def write_results(dir, text_list, mode):
    if mode == 'decoded':
        pass
    elif mode == 'reference':
        pass
    else:
        raise Exception('Wrong Result Mode!!!')

    data_num = len(text_list)
    for i in range(data_num):
        one_text = text_list[i]
        one_out_f = dir + '/' + str(i) + '_' + mode + '.txt'
        with open(one_out_f, 'w', encoding = 'utf8') as o:
            o.writelines(one_text)

def remove_tmp_files():
    import os
    path = r'/tmp/'
    folder_name_list = os.listdir(path)
    for name in folder_name_list:
        if name.startswith('tmp') and name != 'tmpaddon':
            pass
        else:
            continue

        one_full_path = path + '/' + name
        try:
            #os.system(r'rm -r ' + one_full_path)
            os.system(r'rm -r --interactive=never ' + one_full_path)
        except:
            continue

def compute_rogue_scores(prediction_list, reference_list):
    path_prefix = r'./tmp/'
    summary_dir, model_dir = path_prefix + '/prediction/', path_prefix + '/reference/'
    import os
    if os.path.exists(summary_dir):
        pass
    else: # recursively construct directory
        os.makedirs(summary_dir, exist_ok=True)

    if os.path.exists(model_dir):
        pass
    else: # recursively construct directory
        os.makedirs(model_dir, exist_ok=True)

    write_results(summary_dir, prediction_list, mode = r'decoded')
    write_results(model_dir, reference_list, mode = r'reference')
    rogue_1_score, rogue_2_score, rogue_l_score = get_rouge_scores(summary_dir, model_dir)
    # remove the local tmp directory
    os.system(r'rm -r ./tmp/')
    # remove the global tmp directory
    remove_tmp_files()
    return rogue_1_score, rogue_2_score, rogue_l_score
