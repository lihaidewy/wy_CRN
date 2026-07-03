import mmcv
infos = mmcv.load("./data/my_formatted_data/nuscenes_infos_val.pkl")
print(f"前五帧 Token: {[i['sample_token'] for i in infos[:5]]}")