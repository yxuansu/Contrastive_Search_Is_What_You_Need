import argparse
def parse_config():
    parser = argparse.ArgumentParser()
    parser.add_argument("--test_path", type=str)
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_config()
    evaluation_save_path = args.test_path[:-5] + '_diversity_mauve_gen_length_result.json'
    print ('evaluation save name is {}'.format(evaluation_save_path))
    
    print ('Measuring diversity...')
    from utlis.compute_diversity import measure_diversity
    diversity_dict = measure_diversity(args.test_path)
    print (diversity_dict)
    print ('Diversity measurement completed!')

    print ('Measuring generation length...')
    from utlis.compute_gen_length import measure_gen_length
    gen_length_dict = measure_gen_length(args.test_path)
    print (gen_length_dict)
    print ('Generation length measurement completed!')

    print ('Measuring MAUVE...')
    from utlis.compute_mauve import measure_mauve
    mauve_dict = measure_mauve(args.test_path)
    print (mauve_dict)
    print ('MAUVE measurement completed!')

    result_dict = {
        'diversity_dict': diversity_dict,
        'gen_length_dict': gen_length_dict,
        'mauve_dict': mauve_dict
    }
    
    import json
    with open(evaluation_save_path, 'w') as outfile:
        json.dump([result_dict], outfile, indent=4)



