
import os
import re
import torch
import shutil
import configargparse
parser = configargparse.ArgumentParser()

parser.add_argument("--basepath", type=str, default='./Checkpoint_all/triplane_omni', 
                    help='path of fitted triplane ')
parser.add_argument("--mode", type=str, default='omni', 
                    help='the name of dataset')
parser.add_argument("--newpath", type=str, default='./Checkpoint_all/triplane_omni_fordiffusion', 
                    help='the newpath of fitted triplane')
args = parser.parse_args()


if args.mode=='omni':
    basepath=args.basepath
    sharedecoderpath=args.newpath
    newpath=os.path.join(sharedecoderpath,'triplane')
    os.makedirs(os.path.join(newpath), exist_ok=True)
    allname=[di for di in os.listdir(basepath) if os.path.isdir(os.path.join(basepath,di))]
    allname.sort(key=lambda l: int(re.findall('\d+', l)[0]))
    names=os.listdir(newpath)
    i=0
    max_id=[]
    for name in allname:
        try:
            if name+'.tar' in names:
                print(name)
                continue
            if i==0:
                shutil.copyfile(os.path.join(basepath,name,'003000.tar'), os.path.join(sharedecoderpath,'003000.tar'))
            alldata=torch.Tensor(torch.load(os.path.join(basepath,name,'003000.tar'),map_location='cpu')['network_fn_state_dict']['tri_planes'])
            i=i+1
            os.makedirs(os.path.join(newpath, name[:-4]), exist_ok=True)
            torch.save(alldata,os.path.join(newpath, name[:-4],name+'.tar'))

        except:
            print('unfinished',name)
        
        print(i)
    print(max_id)
else:

    basepath=args.basepath
    sharedecoderpath=args.newpath
    newpath=os.path.join(sharedecoderpath,'triplane')
    allname=[di for di in os.listdir(basepath) if os.path.isdir(os.path.join(basepath,di))]
    allname.sort(key=lambda l: int(re.findall('\d+', l)[0]))
    names=os.listdir(newpath)
    i=0

    max_id=[]
    for name in allname:
        try:
            if name+'.tar' in names:
                print(name)
                continue
            if i==0:
                shutil.copyfile(os.path.join(basepath,name,'003000.tar'), os.path.join(sharedecoderpath,'003000.tar'))
            alldata=torch.Tensor(torch.load(os.path.join(basepath,name,'003000.tar'),map_location='cpu')['network_fn_state_dict']['tri_planes'])
            i=i+1
            torch.save(alldata,os.path.join(newpath,name+'.tar'))


        except:
            print('unfinished',name)
        
        print(i)
    print(max_id)
