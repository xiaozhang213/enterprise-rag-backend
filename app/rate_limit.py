"""
速率限制器单独放一个模块，避免 app.main 和路由模块互相 import 造成循环依赖。
"""
from slowapi import Limiter
from slowapi.util import get_remote_address

limiter = Limiter(key_func=get_remote_address)  # 按客户端IP限流
