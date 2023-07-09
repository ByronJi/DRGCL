import numpy as np
with open('res_collect_80_nocca_0.001','r') as f:
    lines=f.readlines()
    bbbp,tox21,toxcast,sider,clintox,muv,hiv,bace=[],[],[],[],[],[],[],[]
    for line in lines:
        list = line.strip().split(' ')
        if list[0]=='bbbp':
            bbbp.append(float(list[2]))
        elif list[0]=='tox21':
            tox21.append(float(list[2]))
        elif list[0]=='toxcast':
            toxcast.append(float(list[2]))
        elif list[0]=='sider':
            sider.append(float(list[2]))
        elif list[0]=='clintox':
            clintox.append(float(list[2]))
        elif list[0]=='muv':
            muv.append(float(list[2]))
        elif list[0]=='hiv':
            hiv.append(float(list[2]))
        elif list[0]=='bace':
            bace.append(float(list[2]))
    # print(bbbp)
    np_bbbp = np.array(bbbp)
    print("bbbp",np.mean(np_bbbp),np.std(np_bbbp))
    
    # print(tox21)
    np_tox21 =np.array(tox21)
    print("tox21",np.mean(np_tox21),np.std(np_tox21))
    
    # print(toxcast)
    np_toxcast = np.array(toxcast)
    print("toxcast",np.mean(np_toxcast),np.std(np_tox21))
    
    # print(sider)
    np_sider = np.array(sider)
    print("sider",np.mean(np_sider),np.std(np_sider))
    
    # print(clintox)
    np_clintox = np.array(clintox)
    print("clintox",np.mean(np_clintox),np.std(np_clintox))
    
    # print(muv)
    np_muv = np.array(muv)
    print("muv",np.mean(np_muv),np.std(np_muv))
    
    # print(hiv)
    np_hiv = np.array(hiv)
    print("hix",np.mean(np_hiv),np.std(np_hiv))
    
    # print(bace)
    np_bace = np.array(bace)
    print("bace",np.mean(np_bace),np.std(np_bace))
    
