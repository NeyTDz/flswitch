import numpy as np

def switch_propose(switchs,switch_propose,raw_switch=None):
    switchs = np.array(switchs)
    if switch_propose == 'all':
        if raw_switch == None: assert 0 
        if (switchs == '1').all(): 
            return '1'
        elif (switchs == '2').all(): 
            return '2'
        else: 
            return raw_switch
    elif switch_propose == 'most':
        if np.sum((switchs == '1')) >= np.sum((switchs == '2')):
            return '1'
        else:
            return '2'
    else:
        print('unknown propose choice!')
        assert 0
