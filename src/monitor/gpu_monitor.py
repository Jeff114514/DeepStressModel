"""
GPU监控模块，提供远程GPU监控功能
"""
import time
import platform
import subprocess
from typing import Dict, Optional, List
import paramiko
import psutil
from src.utils.logger import setup_logger
from src.utils.config import config
from src.data.db_manager import db_manager

logger = setup_logger("gpu_monitor")

class GPUStats:
    """GPU统计数据类"""
    def __init__(
        self,
        gpus: List[Dict] = None,  # 多GPU数据列表
        cpu_util: float = 0.0,    # CPU使用率
        memory_util: float = 0.0,  # 系统内存使用率
        disk_util: float = 0.0,    # 磁盘使用率
        disk_io_latency: float = 0.0,  # 磁盘IO延时（毫秒）
        network_io: Dict[str, float] = None,  # 网络IO统计
        timestamp: float = None,
        cpu_info: str = "",       # CPU型号信息
        gpu_count: int = 0,       # GPU数量
        total_memory: int = 0     # 系统总内存(GB)
    ):
        self.gpus = gpus or []    # 多GPU数据
        self.cpu_util = cpu_util 
        self._memory_util = memory_util     # %
        self.disk_util = disk_util         # %
        self.disk_io_latency = disk_io_latency  # ms
        self.network_io = network_io or {}  # 包含上传下载速度
        self.timestamp = timestamp or time.time()
        self.cpu_info = cpu_info           # CPU型号
        self.gpu_count = gpu_count         # GPU数量
        self.total_memory = total_memory   # 系统总内存
    
    # 为了兼容现有代码，提供属性访问方法
    @property
    def gpu_util(self) -> float:
        """第一个GPU的利用率"""
        return self.gpus[0]['util'] if self.gpus else 0.0
    
    @property
    def memory_used(self) -> float:
        """第一个GPU的已用显存"""
        return self.gpus[0]['memory_used'] if self.gpus else 0.0
    
    @property
    def memory_total(self) -> float:
        """第一个GPU的总显存"""
        return self.gpus[0]['memory_total'] if self.gpus else 0.0
    
    @property
    def temperature(self) -> float:
        """第一个GPU的温度"""
        return self.gpus[0]['temperature'] if self.gpus else 0.0
    
    @property
    def power_usage(self) -> float:
        """第一个GPU的功率使用"""
        return self.gpus[0]['power_usage'] if self.gpus else 0.0
    
    @property
    def power_limit(self) -> float:
        """第一个GPU的功率限制"""
        return self.gpus[0]['power_limit'] if self.gpus else 0.0
    
    @property
    def gpu_info(self) -> str:
        """第一个GPU的型号信息"""
        return self.gpus[0]['info'] if self.gpus else ""
    
    @property
    def gpu_memory_util(self) -> float:
        """第一个GPU的显存使用率"""
        if not self.gpus:
            return 0.0
        gpu = self.gpus[0]
        return (gpu['memory_used'] / gpu['memory_total']) * 100 if gpu['memory_total'] > 0 else 0
    
    @property
    def memory_util(self) -> float:
        """系统内存使用率"""
        return self._memory_util
    
    @memory_util.setter
    def memory_util(self, value: float):
        """设置系统内存使用率"""
        self._memory_util = value
    
    def get_gpu(self, index: int) -> Dict:
        """获取指定索引的GPU数据"""
        if 0 <= index < len(self.gpus):
            return self.gpus[index]
        return None
    
    def get_gpu_memory_util(self, index: int) -> float:
        """获取指定GPU的显存使用率"""
        gpu = self.get_gpu(index)
        if not gpu:
            return 0.0
        return (gpu['memory_used'] / gpu['memory_total']) * 100 if gpu['memory_total'] > 0 else 0

class GPUMonitor:
    """远程GPU监控类"""
    def __init__(self, host: str, username: str, password: str, port: int = 22):
        self.host = host
        self.username = username
        self.password = password
        self.port = port
        self.client = None
        self.max_retries = 3  # 最大重试次数
        self.retry_interval = 2  # 重试间隔（秒）
        # 添加网络IO历史数据
        self._last_net_stats = None
        self._last_net_time = None
        self._connect()

    def _is_connected(self) -> bool:
        """检查 SSH 连接是否可用"""
        try:
            transport = self.client.get_transport() if self.client else None
            return bool(transport and transport.is_active())
        except Exception:
            return False
    
    def _connect(self) -> bool:
        """连接到远程服务器"""
        for attempt in range(self.max_retries):
            try:
                if self.client:
                    try:
                        self.client.close()
                    except:
                        pass
                
                self.client = paramiko.SSHClient()
                self.client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
                self.client.connect(
                    self.host,
                    port=self.port,
                    username=self.username,
                    password=self.password,
                    timeout=5
                )
                logger.info(f"成功连接到远程服务器: {self.host}:{self.port}")
                return True
            except Exception as e:
                logger.error(f"连接远程服务器失败 (尝试 {attempt + 1}/{self.max_retries}): {e}")
                if attempt < self.max_retries - 1:
                    time.sleep(self.retry_interval)
                    continue
                return False
        return False

    def _execute_command(self, command: str) -> Optional[str]:
        """执行远程命令"""
        for attempt in range(self.max_retries):
            try:
                if not self._is_connected():
                    if not self._connect():
                        return None
                
                stdin, stdout, stderr = self.client.exec_command(command, timeout=5)
                return stdout.read().decode().strip()
            except Exception as e:
                logger.error(f"执行远程命令失败 (尝试 {attempt + 1}/{self.max_retries}): {e}")
                if attempt < self.max_retries - 1:
                    # 重置连接
                    try:
                        self.client.close()
                    except:
                        pass
                    self.client = None
                    time.sleep(self.retry_interval)
                    continue
                return None
        return None

    def _get_network_speed(self) -> Dict[str, float]:
        """获取网络速度"""
        try:
            # 获取当前网络数据
            net_command = "cat /proc/net/dev | grep -E 'eth0|ens|enp' | head -n1 | awk '{print $2,$10}'"
            net_output = self._execute_command(net_command)
            if not net_output:
                return {'receive': 0, 'transmit': 0, 'receive_rate': 0.1, 'send_rate': 0.1}
            
            current_time = time.time()
            net_parts = net_output.split()
            if len(net_parts) < 2:
                return {'receive': 0, 'transmit': 0, 'receive_rate': 0.1, 'send_rate': 0.1}
                
            # 获取字节数
            net_recv = float(net_parts[0])
            net_send = float(net_parts[1])
            
            # 如果是第一次获取数据
            if self._last_net_stats is None:
                self._last_net_stats = (net_recv, net_send)
                self._last_net_time = current_time
                return {'receive': net_recv, 'transmit': net_send, 'receive_rate': 0.1, 'send_rate': 0.1}
            
            # 计算时间差
            time_diff = current_time - self._last_net_time
            if time_diff <= 0:
                return {'receive': net_recv, 'transmit': net_send, 'receive_rate': 0.1, 'send_rate': 0.1}
            
            # 计算速率（bytes/s）
            recv_speed = (net_recv - self._last_net_stats[0]) / time_diff
            send_speed = (net_send - self._last_net_stats[1]) / time_diff
            
            # 更新历史数据
            self._last_net_stats = (net_recv, net_send)
            self._last_net_time = current_time
            
            # 确保速率不小于0.1KB/s以保证显示
            recv_rate = max(0.1, recv_speed / 1024)  # KB/s
            send_rate = max(0.1, send_speed / 1024)
            
            return {
                'receive': net_recv, 
                'transmit': net_send,
                'receive_rate': recv_rate,
                'send_rate': send_rate
            }
            
        except Exception as e:
            logger.error(f"获取网络速度失败: {e}")
            return {'receive': 0, 'transmit': 0, 'receive_rate': 0.1, 'send_rate': 0.1}

    def get_stats(self, db_manager=None) -> Optional[GPUStats]:
        """获取GPU和系统状态

        Returns:
            GPUStats: GPU统计信息
        """
        try:
            if not self._is_connected():
                logger.warning("无法获取GPU信息：SSH连接未建立或已断开，尝试重连")
                if not self._connect():
                    return None

            # 获取GPU使用信息
            all_gpus = []
            stdin, stdout, stderr = self.client.exec_command(
                "nvidia-smi --query-gpu=index,utilization.gpu,memory.used,memory.total,temperature.gpu,power.draw,power.limit,name "
                "--format=csv,noheader,nounits"
            )
            gpu_outputs = stdout.read().decode().strip().split('\n')
            gpu_count = len(gpu_outputs)
            
            for line in gpu_outputs:
                if not line.strip():
                    continue
                parts = [p.strip() for p in line.split(',')]
                if len(parts) >= 8:
                    try:
                        index = int(parts[0])
                        gpu_util = float(parts[1])
                        memory_used = float(parts[2])
                        memory_total = float(parts[3])
                        temperature = float(parts[4])
                        power_usage = float(parts[5]) if parts[5] else 0.0
                        power_limit = float(parts[6]) if parts[6] else 0.0
                        gpu_info = parts[7]
                        
                        gpu_data = {
                            'index': index,
                            'util': gpu_util,
                            'memory_used': memory_used,
                            'memory_total': memory_total,
                            'temperature': temperature,
                            'power_usage': power_usage,
                            'power_limit': power_limit,
                            'info': gpu_info
                        }
                        all_gpus.append(gpu_data)
                    except (ValueError, IndexError) as e:
                        logger.error(f"解析GPU数据错误: {e}, 数据行: {line}")
            
            if not all_gpus:
                return None
            
            # 获取CPU使用率
            stdin, stdout, stderr = self.client.exec_command(
                "top -bn1 | grep 'Cpu(s)' | sed 's/.*, *\\([0-9.]*\\)%* id.*/\\1/' | awk '{print 100 - $1}'"
            )
            cpu_util = float(stdout.read().decode().strip())
            
            # 获取内存使用情况
            stdin, stdout, stderr = self.client.exec_command(
                "free -m | grep Mem | awk '{print $3,$2}'"
            )
            mem_output = stdout.read().decode().strip().split()
            if len(mem_output) >= 2:
                mem_used = float(mem_output[0])
                mem_total = float(mem_output[1])
                memory_util = (mem_used / mem_total) * 100
                total_memory = int(mem_total / 1024)  # 转换为GB
            else:
                memory_util = 0
                total_memory = 0
            
            # 获取磁盘使用情况
            stdin, stdout, stderr = self.client.exec_command(
                "df -h / | grep / | awk '{print $5}' | sed 's/%//'"
            )
            disk_util = float(stdout.read().decode().strip())
            
            # 获取磁盘IO延迟 - 使用iowait百分比作为指标
            # iowait表示CPU等待I/O完成的时间百分比，更准确地反映了系统I/O性能
            iowait_command = "top -bn1 | grep '%Cpu' | awk '{print $10}'"
            iowait_output = self._execute_command(iowait_command)
            
            try:
                # 将iowait百分比转换为毫秒单位的延迟值 (0-100% -> 0-100ms)
                iowait_percent = float(iowait_output) if iowait_output else 1.0
                disk_io_latency = iowait_percent  # 直接使用iowait百分比值作为毫秒数
            except (ValueError, TypeError):
                disk_io_latency = 1.0  # 默认值
            
            # 获取网络IO信息
            stdin, stdout, stderr = self.client.exec_command(
                "cat /proc/net/dev | grep -e eth0 -e ens -e eno -e enp | awk '{print $2,$10}'"
            )
            net_output = stdout.read().decode().strip().split()
            if len(net_output) >= 2:
                net_receive = float(net_output[0]) / 1024 / 1024  # 转换为MB
                net_send = float(net_output[1]) / 1024 / 1024    # 转换为MB
                
                # 计算网络IO速率
                curr_time = time.time()
                if self._last_net_stats:
                    time_diff = curr_time - self._last_net_time
                    if time_diff > 0:
                        if isinstance(self._last_net_stats, dict) and 'receive' in self._last_net_stats and 'transmit' in self._last_net_stats:
                            last_receive = self._last_net_stats['receive']
                            last_send = self._last_net_stats['transmit']
                            
                            receive_rate = max(0.1, (net_receive - last_receive) / time_diff * 1000)  # KB/s
                            send_rate = max(0.1, (net_send - last_send) / time_diff * 1000)          # KB/s
                        else:
                            receive_rate = 0.1
                            send_rate = 0.1
                    else:
                        receive_rate = 0.1
                        send_rate = 0.1
                else:
                    receive_rate = 0.1
                    send_rate = 0.1
                
                network_io = {
                    'receive': net_receive,
                    'transmit': net_send,
                    'receive_rate': receive_rate,
                    'send_rate': send_rate
                }
            else:
                network_io = {'receive': 0, 'transmit': 0, 'receive_rate': 0.1, 'send_rate': 0.1}
            
            # 获取CPU信息
            stdin, stdout, stderr = self.client.exec_command(
                "lscpu | grep 'Model name' | sed 's/Model name: *//'"
            )
            cpu_info = stdout.read().decode().strip()
            
            # 创建GPU统计对象
            stats = GPUStats(
                gpus=all_gpus,
                cpu_util=cpu_util,
                memory_util=memory_util,
                disk_util=disk_util,
                disk_io_latency=disk_io_latency,
                network_io=network_io,
                cpu_info=cpu_info.strip(),
                gpu_count=gpu_count,
                total_memory=total_memory
            )
            
            self._last_net_stats = network_io
            self._last_net_time = curr_time
            
            # 保存到数据库
            if db_manager:
                db_manager.add_gpu_stats(
                    self.host,
                    stats.gpu_util,
                    stats.gpu_memory_util,
                    stats.temperature,
                    stats.power_usage,
                    stats.cpu_util,
                    stats.memory_util
                )
            
            return stats
            
        except Exception as e:
            logger.error(f"获取GPU状态错误: {e}")
            self.handle_connection_error()
            return None
    
    def handle_connection_error(self):
        """处理连接错误"""
        logger.info(f"处理连接错误: {self.host}")
        if self.client:
            try:
                self.client.close()
            except:
                pass
            self.client = None
        # 尝试重新连接
        self._connect()
    
    def __del__(self):
        """析构函数，关闭SSH连接"""
        if self.client:
            try:
                self.client.close()
            except:
                pass

class GPUMonitorManager:
    """GPU监控管理器"""
    def __init__(self):
        self.monitor = None
        self.active_server = None
        self._local_monitor = None
    
    def init_monitor(self):
        """初始化GPU监控器"""
        try:
            use_remote = config.get("gpu_monitor.remote.enabled", False)
            if use_remote:
                # 获取活动的GPU服务器配置
                server = db_manager.get_active_gpu_server()
                if server:
                    if not server.get("host") or not server.get("username") or not server.get("password"):
                        logger.warning("GPU服务器配置不完整，跳过初始化")
                        self.monitor = None
                        return
                    self.setup_monitor(
                        server["host"],
                        server["username"],
                        server["password"],
                        server.get("port", 22)  # 使用默认端口22
                    )
                    logger.info(f"GPU监控器初始化成功: {server['host']}:{server.get('port', 22)}")
                    return
                logger.warning("未找到活动的GPU服务器配置，将尝试使用本地监控")
            # 本地模式
            self._init_local_monitor()
        except Exception as e:
            logger.error(f"初始化GPU监控器失败: {e}")
            self.monitor = None
    
    def setup_monitor(self, host: str, username: str, password: str, port: int = 22):
        """设置GPU监控器
        
        Args:
            host: 主机地址
            username: 用户名
            password: 密码
            port: SSH端口
        """
        try:
            self.monitor = GPUMonitor(host, username, password, port)
            logger.info(f"GPU监控器设置成功: {host}:{port}")
        except Exception as e:
            logger.error(f"设置GPU监控器失败: {e}")
            self.monitor = None

    def _init_local_monitor(self):
        """初始化本地监控器"""
        try:
            self._local_monitor = LocalGPUMonitor()
            self.monitor = self._local_monitor
            logger.info("已启用本地GPU监控（使用 nvidia-smi）")
        except Exception as exc:  # noqa: BLE001
            logger.warning(f"本地GPU监控初始化失败: {exc}")
            self.monitor = None
    
    def get_stats(self) -> Optional[GPUStats]:
        """获取GPU统计数据"""
        try:
            if not self.monitor:
                # 若未初始化且允许本地，则尝试本地
                self._init_local_monitor()
            if not self.monitor:
                return None
            return self.monitor.get_stats()
        except Exception as e:
            # 只在第一次错误时记录日志，避免重复报错
            if not hasattr(self, '_get_stats_error_logged'):
                logger.warning("获取GPU统计数据失败: %s (后续错误将静默处理)", e)
                self._get_stats_error_logged = True
            return None

# 创建全局GPU监控管理器实例，但不自动初始化
gpu_monitor = GPUMonitorManager()


class LocalGPUMonitor:
    """
    本地 GPU 监控：直接调用 nvidia-smi，无需 SSH。
    适用于单机测试环境。
    """

    def __init__(self):
        self._last_net = None
        self._last_time = None
        self._get_stats_error_logged = False  # 标记是否已经记录过错误
        # 预检 nvidia-smi
        if not self._has_nvidia_smi():
            raise RuntimeError("未找到 nvidia-smi，无法启用本地GPU监控")

    @staticmethod
    def _has_nvidia_smi() -> bool:
        try:
            subprocess.run(["nvidia-smi", "-h"], capture_output=True, check=False)
            return True
        except Exception:
            return False

    def _run_cmd(self, cmd: List[str]) -> str:
        res = subprocess.run(cmd, capture_output=True, text=True, check=False)
        return res.stdout.strip()

    def _network_speed(self) -> Dict[str, float]:
        now = time.time()
        counters = psutil.net_io_counters()
        if not counters:
            return {"receive_rate": 0.1, "send_rate": 0.1, "receive": 0, "transmit": 0}
        recv = counters.bytes_recv
        sent = counters.bytes_sent
        if self._last_net is None:
            self._last_net = (recv, sent)
            self._last_time = now
            return {"receive_rate": 0.1, "send_rate": 0.1, "receive": recv, "transmit": sent}
        dt = max(1e-6, now - self._last_time)
        recv_rate = max(0.1, (recv - self._last_net[0]) / dt / 1024)  # KB/s
        send_rate = max(0.1, (sent - self._last_net[1]) / dt / 1024)
        self._last_net = (recv, sent)
        self._last_time = now
        return {"receive_rate": recv_rate, "send_rate": send_rate, "receive": recv, "transmit": sent}

    def get_stats(self, db_manager=None) -> Optional[GPUStats]:
        try:
            # GPU 信息
            query = [
                "nvidia-smi",
                "--query-gpu=index,utilization.gpu,memory.used,memory.total,temperature.gpu,power.draw,power.limit,name",
                "--format=csv,noheader,nounits",
            ]
            output = self._run_cmd(query)
            if not output:
                return None
            all_gpus = []
            for line in output.splitlines():
                parts = [p.strip() for p in line.split(",")]
                if len(parts) < 8:
                    continue
                idx, util, mem_used, mem_total, temp, p_draw, p_limit, name = parts[:8]
                all_gpus.append(
                    {
                        "index": int(idx),
                        "util": float(util),
                        "memory_used": float(mem_used),
                        "memory_total": float(mem_total),
                        "temperature": float(temp),
                        "power_usage": float(p_draw) if p_draw else 0.0,
                        "power_limit": float(p_limit) if p_limit else 0.0,
                        "info": name,
                    }
                )
            if not all_gpus:
                return None

            cpu_util = psutil.cpu_percent(interval=None)
            mem = psutil.virtual_memory()
            disk = psutil.disk_usage("/")
            net = self._network_speed()

            stats = GPUStats(
                gpus=all_gpus,
                cpu_util=cpu_util,
                memory_util=mem.percent,
                disk_util=disk.percent,
                disk_io_latency=0.0,
                network_io=net,
                cpu_info=platform.processor(),
                gpu_count=len(all_gpus),
                total_memory=int(mem.total / (1024**3)),
            )
            # 本地模式无需 db 存储，保持接口一致即可
            # 如果之前有错误但现在成功了，重置错误标记
            if self._get_stats_error_logged:
                self._get_stats_error_logged = False
            return stats
        except Exception as exc:  # noqa: BLE001
            # 只在第一次错误时记录日志，避免重复报错
            if not self._get_stats_error_logged:
                logger.warning("获取本地GPU状态错误: %s (后续错误将静默处理)", exc)
                self._get_stats_error_logged = True
            return None
