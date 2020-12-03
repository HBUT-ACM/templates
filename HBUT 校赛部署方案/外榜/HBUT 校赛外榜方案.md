# HBUT 外榜方案

##  方案1(低配版)

#### 前置准备

- 一台有公网ip的 VPS(linux, 且带宽越高越好) 
- 一个外网域名，并DNS解析至该VPS公网ip
- 在domserver本机上/VPS上安装 [frp](https://github.com/fatedier/frp/tree/master)

#### 在domserver上配置frpc.ini

![image-20201203171117399](https://flysky-tencent-1302120781.cos.ap-chengdu.myqcloud.com/markdownImg/image-20201203171117399.png)

Server_addr & custom_domains 都是外榜域名 

默认穿透至server_port  

#### 在VPS上配置frps.ini

![image-20201203171222774](https://flysky-tencent-1302120781.cos.ap-chengdu.myqcloud.com/markdownImg/image-20201203171222774.png)

bind_port:保持同上图server_port相同

vhost_http_port: VPS上的穿透的出来的端口

#### 可以配置 nginx 把 vhost_http_port 映射到 80或者443端口上（选做）

#### 先在VPS上启动 frps

```bash
./frps -c frps.ini
```

#### 在Domserver上启动frpc

```bash
./frpc -c frpc.ini
```

#### Domserver 起来之后 访问  [域名]:[vhost_http_port]/public 能看到外榜

## 方案2(高配版)

#### 前置准备

- 方案1 能work
- 高带宽VPS

TODO: @isaacchen

