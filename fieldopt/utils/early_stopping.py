import torch
import numpy as np

def check_convergence(values, window_size=10, threshold=1e-9):
    """
    检查数值序列是否收敛
    """
    if len(values) < window_size:
        return False
    
    recent_values = values[-window_size:]
    # 计算最近窗口内的方差
    variance = torch.var(torch.tensor(recent_values)).item()
    return variance < threshold

def check_trend_improvement(values, window_size=5, min_delta=1e-6):
    """
    检查趋势是否有改进
    """
    if len(values) < window_size * 2:
        return True
    
    recent_avg = np.mean(values[-window_size:])
    previous_avg = np.mean(values[-window_size*2:-window_size])
    
    improvement = previous_avg - recent_avg
    return improvement > min_delta

class EarlyStoppingMonitor:
    """
    多策略早停监控器
    """
    def __init__(self, config):
        self.config = config
        self.best_loss = float('inf')
        self.best_model_state = None
        self.best_each_loss = None
        
        # 各种patience计数器
        self.loss_patience_counter = 0
        
        # 历史记录
        self.loss_history = []
        
    def update(self, monitor_loss, current_lr, model, each_loss=None):
        """
        更新早停监控状态
        
        Returns:
            (should_stop, reason, best_model_state)
        """
        self.loss_history.append(monitor_loss)
        
        # 策略1: 基于监控损失的改进
        if monitor_loss < self.best_loss - self.config['min_delta']:
            self.best_loss = monitor_loss
            self.best_model_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            self.best_each_loss = dict(each_loss) if each_loss is not None else None
            self.loss_patience_counter = 0
            # print(f"Best loss updated to {monitor_loss:.6f}")
        else:
            self.loss_patience_counter += 1
            # print(f"Loss patience counter increased to {self.loss_patience_counter}")
        
        # 策略2: 学习率过小
        lr_too_small = current_lr < self.config['lr_threshold']
        
        # 策略3: 损失收敛
        loss_converged = check_convergence(
            self.loss_history, 
            window_size=self.config['window_size'],
            threshold=self.config['convergence_threshold']
        )
        
        # 综合判断
        reasons = []
        should_stop = False
        
        # 损失早停
        if self.loss_patience_counter >= self.config['loss_patience']:
            should_stop = True
            reasons.append(f"loss patience ({self.config['loss_patience']})")
        
        # 学习率早停
        # if lr_too_small:
        #     should_stop = True
        #     reasons.append(f"learning rate too small ({current_lr:.2e})")
        
        # 收敛早停
        if loss_converged and len(self.loss_history) > 50:
            should_stop = True
            reasons.append("loss converged")
        
        reason = "Early stopping: " + ", ".join(reasons) if reasons else ""
        
        return should_stop, reason, self.best_model_state
    
    def get_status(self):
        """获取当前状态信息"""
        return {
            'best_loss': self.best_loss,
            'loss_patience': self.loss_patience_counter,
        }
    
    def get_best_state(self):
        """获取最佳模型状态"""
        return self.best_model_state

    def get_best_each_loss(self):
        """获取最佳模型对应的各项损失"""
        return self.best_each_loss
