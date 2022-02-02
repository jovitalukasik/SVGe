import os
import pdb
import numpy as np



gpu_id = 0

arcs_scores = '''
1 3 0 3 0 0 3 0 0 0 3 0 0 0 0 3 0 1 0 1 0 3 0 0 1 0 0 0 1 0 0 1 1 0 1 0 3 0 0 0 0 1 0 1 0 1 0 0 0 1 1 1 0 1 0 1 0 1 0 1 1 0 0 1 0 1 1 0 0 1 0 1 0 0 1 0 0 1 0.6038
'''

enas_pos = '../software/enas/'

arcs = [x.split('.')[0][:-2] for x in arcs_scores.strip().split('\n')]
print(arcs)

scores = []

for arc in arcs:
    print('Fully training ENAS architecture ' + arc)
    save_appendix = ''.join(arc.split()) + '_256filters'
    if not os.path.exists(enas_pos + 'outputs_' + save_appendix):
        pwd = os.getcwd()
        os.chdir(enas_pos)
        os.system('CUDA_VISABLE_DEVICES={} ./scripts/custom_cifar10_macro_final_12.sh'.format(gpu_id) + ' "' + arc + '" ' + save_appendix + " 256")
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
