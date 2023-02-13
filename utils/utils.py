import numpy as np

def print_CF(cf, classes):
    
    classes2 = []
    
    for c in classes:
        classes2.append(c[:5])#(c[:int(len(c)*0.7)])
    classes = classes2
    
    lens = []
    for c in classes:
        lens.append(len(c))
    lens = np.array(lens)
    
    l = lens.max()
        
    # Print top row and rules
    row = ' ' * (l+2)
    row2 = ' ' * (l+2)
    for i, c in enumerate(classes):
        row = row + c + ' |'
        for _ in range(len(c)):
            row2 = row2 + '='
        row2 = row2 +'='*2

    print(row)
    print(row2)
    
    # Print rows and columns
    row = ''
    for i, c in enumerate(classes):
        row = row + c + (' ' * (l-len(c))) +'||'
        for j in range(len(classes)):
            cs = '%.2f'%cf[i,j]
            row = row + cs + ' '*(1+(lens[j]-len(cs))) + '|'
        print(row)
        row = ''
