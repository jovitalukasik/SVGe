import os
import pdb
import numpy as np



gpu_id = 0

arcs_scores = '''
3 2 0 4 0 1 1 1 0 0 3 1 0 1 0 5 0 0 0 1 0 0.7516
5 0 0 5 0 0 3 1 0 1 5 0 0 0 0 4 1 0 0 0 0 0.7502
0 4 0 3 0 0 0 1 0 1 2 0 0 0 0 5 1 0 0 0 0 0.7502
3 5 0 0 0 1 5 1 0 0 3 1 0 1 0 1 1 1 0 1 0 0.75
0 5 0 3 0 0 5 1 0 1 0 0 1 0 0 0 1 0 0 1 0 0.75
0 5 0 5 0 0 0 1 0 1 3 0 1 0 0 0 1 0 0 1 0 0.75
0 5 0 3 0 1 0 1 0 0 0 1 0 1 0 5 1 1 0 1 0 0.75
0 3 0 2 0 0 4 0 0 1 1 0 0 0 0 0 1 0 0 1 0 0.7498
2 3 0 4 0 0 5 0 0 1 0 1 0 0 0 5 0 0 0 1 0 0.7498
2 3 0 4 0 0 1 0 0 1 2 0 0 0 0 5 1 0 0 1 0 0.7498
0 3 0 4 0 0 0 0 0 1 5 1 1 0 0 2 1 0 0 0 0 0.7496
0 4 0 3 0 0 2 0 0 1 0 1 1 0 0 1 1 0 0 0 0 0.7496
0 3 0 4 0 0 2 0 0 1 1 1 1 0 0 0 1 0 0 0 0 0.7496
0 3 0 4 0 0 0 0 0 1 5 1 1 0 0 0 1 0 0 0 0 0.7496
0 3 0 4 0 0 2 0 0 1 1 1 0 0 0 3 1 0 0 1 0 0.7496
'''



enas_pos = '../software/enas/'

arcs = [x.split('.')[0][:-2] for x in arcs_scores.strip().split('\n')]
print(arcs)

scores = []

for arc in arcs:
    print('Fully training ENAS architecture ' + arc)
    save_appendix = ''.join(arc.split())
    if not os.path.exists(enas_pos + 'outputs_' + save_appendix):
        pwd = os.getcwd()
        os.chdir(enas_pos)
        os.system('CUDA_VISABLE_DEVICES={} ./scripts/custom_cifar10_macro_final_6.sh'.format(gpu_id) + ' "' + arc + '" ' + save_appendix)
        os.chdir(pwd)
    with open(enas_pos + 'outputs_' + save_appendix + '/stdout', 'r') as f:
        last_line = f.readlines()[-1]
        scores.append(last_line)


new_arcs_scores = [x+' '+y for x, y in zip(arcs_scores.strip().split('\n'), scores)]
new_arcs_scores = ''.join(new_arcs_scores)
print()
print('Fully trained architecture, WS acc, and test acc:')
print(new_arcs_scores)
print('Average score is {}'.format(np.mean([float(x.split()[1]) for x in scores])))

# pdb.set_trace()

