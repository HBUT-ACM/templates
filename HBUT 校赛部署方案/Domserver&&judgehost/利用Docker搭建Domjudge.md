# 利用Docker搭建Domjudge

### Docker 安装指南 

- [ubuntu](https://www.runoob.com/docker/ubuntu-docker-install.html)

- [debian](https://www.runoob.com/docker/debian-docker-install.html)

- [centos](https://www.runoob.com/docker/centos-docker-install.html)

- [windows](https://www.runoob.com/docker/windows-docker-install.html)

- [mac](https://www.runoob.com/docker/macos-docker-install.html)

### Docker 换源配置

- 方案1: 修改daemon配置文件 /etc/docker/daemon.json (如果没有该文件，新建一个然后将如下内容复制进去)：

  ```json
  {
    "registry-mirrors" : [
      "http://registry.docker-cn.com",
      "http://docker.mirrors.ustc.edu.cn",
      "http://hub-mirror.c.163.com"
    ],
    "insecure-registries" : [
      "registry.docker-cn.com",
      "docker.mirrors.ustc.edu.cn"
    ],
    "debug" : true,
    "experimental" : true
  }
  
  ```

  ```ssh
  # 使之生效
  sudo systemctl daemon-reload 
  sudo systemctl restart docker
  ```

- 方案2:

  [拿到阿里云上专属的加速链接(含win)]( https://cr.console.aliyun.com/cn-hangzhou/instances/mirrors)

- 方案3: 

  [别人的博客](https://www.runoob.com/docker/docker-mirror-acceleration.html)

  

### [MariaDB](https://hub.docker.com/r/_/mariadb/) container 

  - 拉取MariaDB 镜像(dj-mariadb)    &&  创建一个数据库用户(用户名： domjudge 密码rootpw) 配置一些环境变量

    ```ssh
    docker run -it --name dj-mariadb -e MYSQL_ROOT_PASSWORD=rootpw -e MYSQL_USER=domjudge -e MYSQL_PASSWORD=djpw -e MYSQL_DATABASE=domjudge -p 13306:3306 mariadb --max-connections=10000
    ```

      

### DOMserver container   

- 拉取DOMserver镜像 

  ```
  docker run --link dj-mariadb:mariadb -itd -e MYSQL_HOST=mariadb -e MYSQL_USER=domjudge -e MYSQL_DATABASE=domjudge -e MYSQL_PASSWORD=djpw -e MYSQL_ROOT_PASSWORD=rootpw -e CONTAINER_TIMEZONE=Asia/Shanghai -p 12345:80 --name domserver domjudge/domserver:latest
  ```

  其中 -p 12345:80 代表把容器内的80端口映射到本地12345端口

- Domserver 的admin和judgehost用户的初始密码应该在启动时显示出来，如果没有显示，可以试试这个： (最好记一下这个，后面要用到， 如果忘记密码，[看这里](https://www.domjudge.org/docs/manual/master/config-basic.html#resetting-the-password-for-a-user))

  ```ssh
  docker exec -it domserver cat /opt/domjudge/domserver/etc/initial_admin_password.secret
  docker exec -it domserver cat /opt/domjudge/domserver/etc/restapi.secret
  ```

- 容器跑起来之后就可以通过 http://localhost:12345/ 登录进入管理员账号 有界面说明domserver配置没有问题了

  ```
  # 看nginx 的日志
  docker exec -it domserver [command]
  [command]
  -- nginx-access-log: tail the access log of nginx.
  -- nginx-error-log: tail the error log of nginx.
  -- symfony-log: for DOMjudge using Symfony (i.e. 6.x and higher), tail the symfony log.
  
  # 重启nginx php 命令
  docker exec -it domserver supervisorctl restart [service]
  [service]
  -- nginx
  -- php
  ```

### Judgehost container

- 拉取judgehost 镜像

  ```
  # 连到本地的 Domserver
  docker run -itd --privileged -v /sys/fs/cgroup:/sys/fs/cgroup:ro --name judgehost-0 --link domserver:domserver --hostname judgedaemon-0 -e DAEMON_ID=0 -e CONTAINER_TIMEZONE=Asia/Shanghai -e JUDGEDAEMON_PASSWORD=6rHUsWKSMnuPVzSl domjudge/judgehost:latest
  
  # 连到远端的Domserver
  
  docker run -itd --privileged -v /sys/fs/cgroup:/sys/fs/cgroup:ro --name judgehost-0  --hostname judgedaemon-1 -e DAEMON_ID=0 -e CONTAINER_TIMEZONE=Asia/Shanghai -e JUDGEDAEMON_PASSWORD=6rHUsWKSMnuPVzSl -e DOMSERVER_BASEURL=http://117.152.79.148/ domjudge/judgehost:latest
  # -v 挂载目录 配置cgroup 
  
  

  # 坑 ⚠️ 
  # 同一台设备配置若干个judgehost 要保证每个DEAMON_ID不相同
  # 若干个judgehost 连到同一个domserver 要保证每个judgehost的 hostname不同 否则网页端看不到
  #upd：2020.11.5

  # 如果配置judgehost出现如下错误：
  # Error: cgroup support missing memory features in running kernel. Unable to continue.
  # To fix this, please make the following changes:
  # 1. In /etc/default/grub, add 'cgroup_enable=memory swapaccount=1' to GRUB_CMDLINE_LINUX_DEFAULT
  # 2. Run update-grub
  # 3. Reboot  
  ```


---
upd: 安装 [adminer](https://hub.docker.com/_/adminer) (数据库管理工具, 用于手动修改数据库里面的若干数据 (用于更新logo & 运维) )
- 拉取adminer 镜像

  ```sh
  docker run --link dj-mariadb:domdb -p 8080:8080 adminer
  ```

- 


  完结撒花🎉

