#!/usr/bin/env python3
import pickle

src = 'data/my_formatted_data/nuscenes_infos_train.pkl'
with open(src, 'rb') as f:
    infos = pickle.load(f)

subset = [i for i in infos if int(i['sample_token'].split('_')[-1]) >= 700000]

dst = 'data/my_formatted_data/nuscenes_infos_train_700k.pkl'
with open(dst, 'wb') as f:
    pickle.dump(subset, f)

print(f'总帧 {len(infos)} -> 700000批次 {len(subset)} 帧')
for i in subset[:3]:
    print(f'  示例: {i["sample_token"]}  scene={i["scene_name"]}')
