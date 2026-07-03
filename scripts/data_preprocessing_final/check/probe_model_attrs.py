import os
import torch
from exps.det.CRN_r50_256x704_128x128_4key import CRNLightningModel

CKPT_PATH = "./outputs/det/CRN_r50_256x704_128x128_4key/lightning_logs/version_68/checkpoints/epoch=23-step=4320.ckpt"

def probe_model():
    print("\n" + "="*60)
    print("🕵️‍♂️ 启动 Lightning 模型解剖探针...")
    print("="*60 + "\n")

    # 1. 初始化并加载权重
    model = CRNLightningModel()
    checkpoint = torch.load(CKPT_PATH, map_location='cuda')
    model.load_state_dict(checkpoint['state_dict'])
    
    # 2. 探查基础配置属性 (看看哪些存在，长什么样)
    attrs_to_check = [
        'classes', 'data_root', 'info_paths', 
        'ida_aug_conf', 'bda_aug_conf', 'rda_aug_conf',
        'return_depth', 'return_radar_pv'
    ]
    
    print("🔍 [基础参数检查结果]：")
    for attr in attrs_to_check:
        if hasattr(model, attr):
            val = getattr(model, attr)
            # 缩短字典类型的打印，避免刷屏
            if isinstance(val, dict):
                print(f" -> ✅ model.{attr:15s} : 存在，包含的 Key 键为: {list(val.keys())}")
            else:
                print(f" -> ✅ model.{attr:15s} : 存在，其真实数值为: {val}")
        else:
            print(f" -> ❌ model.{attr:15s} : ！！！不存在这个属性！！！")

    # 3. 探查原作者在底层实例化 Dataset 时偷偷用了哪些骚操作
    print("\n🔍 [内部数据流线索追溯]：")
    try:
        # 尝试调一下原作者创建数据加载器的方法，看看他在里面读了什么
        train_loader = model.train_dataloader()
        internal_ds = train_loader.dataset
        print(f" -> 🎯 成功抓到训练集绑定的类名: {internal_ds.__class__.__name__}")
        
        # 顺藤摸瓜，直接看看被模型包在里面的 Dataset 实例里有什么
        print(f" -> 🎯 训练集 dataset 内部绑定的 info_paths 真实形态: {getattr(internal_ds, 'info_paths', '未定义')}")
        if hasattr(internal_ds, 'is_train'):
            print(f" -> 🎯 训练集 dataset 内部绑定的 is_train 开关状态: {internal_ds.is_train}")
    except Exception as e:
        print(f" -> ⚠️ 尝试追溯 train_dataloader 时遇到小阻碍: {e}")

    print("\n" + "="*60)
    print("✅ 模型内部透视完毕！请把上面的打印结果发给我！")

if __name__ == "__main__":
    probe_model()