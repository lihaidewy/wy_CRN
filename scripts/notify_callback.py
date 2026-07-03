"""
训练通知回调 — 支持多种推送方式
================================
用法：在 base_cli.py 的 callbacks 列表中加入 NotifyCallback()

支持方式（按优先级）:
  1. Bark (iOS):     填 BARK_URL
  2. Server酱(微信): 填 SENDKEY
  3. 企业微信:       填 WECOM_WEBHOOK
  4. 钉钉:           填 DINGTALK_WEBHOOK
  5. 日志文件:       默认，不需要配置
"""
import os
import time
import requests
from pytorch_lightning.callbacks import Callback


class NotifyCallback(Callback):
    """每个 epoch 结束时推送训练进度"""

    def __init__(self):
        self.start_time = None
        # 读取环境变量配置
        self.bark_url = os.environ.get('BARK_URL', '')
        self.serverchan_key = os.environ.get('SERVERCHAN_KEY', '')
        self.wecom_webhook = os.environ.get('WECOM_WEBHOOK', '')
        self.dingtalk_webhook = os.environ.get('DINGTALK_WEBHOOK', '')
        self.log_file = os.environ.get('NOTIFY_LOG', '/tmp/train_notify.log')

    def on_train_start(self, trainer, pl_module):
        self.start_time = time.time()
        msg = f"🚀 训练开始: {trainer.max_epochs} epochs"
        self._send(msg)

    def on_validation_epoch_end(self, trainer, pl_module):
        epoch = int(trainer.current_epoch)
        max_epochs = int(trainer.max_epochs)

        # 获取最新的 val loss（转为 float）
        logs = trainer.callback_metrics
        val_det = float(logs['val/detection']) if 'val/detection' in logs else None
        val_bbox = float(logs['val/bbox']) if 'val/bbox' in logs else None
        val_depth = float(logs['val/depth']) if 'val/depth' in logs else None

        elapsed = time.time() - self.start_time
        eta = elapsed / max(epoch, 1) * (max_epochs - epoch)

        val_det_str = f"{val_det:.3f}" if val_det is not None else 'N/A'
        val_bbox_str = f"{val_bbox:.3f}" if val_bbox is not None else 'N/A'
        val_depth_str = f"{val_depth:.3f}" if val_depth is not None else 'N/A'

        msg = (
            f"Epoch {epoch}/{max_epochs}\n"
            f"val/det: {val_det_str}\n"
            f"val/bbox: {val_bbox_str}\n"
            f"val/depth: {val_depth_str}\n"
            f"已用: {self._fmt_time(elapsed)} | 预计剩余: {self._fmt_time(eta)}"
        )

        # 关键节点才推送（开始、每10轮、结束），避免消息轰炸
        if epoch == 0 or epoch % 10 == 0 or epoch == max_epochs - 1:
            self._send(msg)
        else:
            # 其他轮次只写日志
            self._log(msg)

    def on_train_end(self, trainer, pl_module):
        elapsed = time.time() - self.start_time
        msg = f"✅ 训练完成！总耗时: {self._fmt_time(elapsed)}"
        self._send(msg)

    def _send(self, msg):
        """优先尝试推送，失败则写日志"""
        sent = False
        if self.bark_url:
            sent = self._send_bark(msg)
        elif self.serverchan_key:
            sent = self._send_serverchan(msg)
        elif self.wecom_webhook:
            sent = self._send_wecom(msg)
        elif self.dingtalk_webhook:
            sent = self._send_dingtalk(msg)

        if not sent:
            self._log(msg)

    def _log(self, msg):
        with open(self.log_file, 'a', encoding='utf-8') as f:
            f.write(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}\n")

    def _send_bark(self, msg):
        try:
            requests.get(f"{self.bark_url}/{msg}", timeout=5)
            return True
        except Exception:
            return False

    def _send_serverchan(self, msg):
        try:
            requests.post(
                f"https://sctapi.ftqq.com/{self.serverchan_key}.send",
                data={'title': 'CRN训练进度', 'desp': msg},
                timeout=5
            )
            return True
        except Exception:
            return False

    def _send_wecom(self, msg):
        try:
            requests.post(self.wecom_webhook, json={'msgtype': 'text', 'text': {'content': msg}}, timeout=5)
            return True
        except Exception:
            return False

    def _send_dingtalk(self, msg):
        try:
            requests.post(self.dingtalk_webhook, json={'msgtype': 'text', 'text': {'content': msg}}, timeout=5)
            return True
        except Exception:
            return False

    @staticmethod
    def _fmt_time(seconds):
        h = int(seconds // 3600)
        m = int((seconds % 3600) // 60)
        return f"{h}h{m}m"
