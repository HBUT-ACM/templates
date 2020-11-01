# 利用Docker搭建Domjudge

### Docker 安装指南 

- [ubuntu]: https://www.runoob.com/docker/ubuntu-docker-install.html	"ubuntu Docker 安装"

- [centos]: https://www.runoob.com/docker/centos-docker-install.html	"centos Docker 安装"

- [windows]: https://www.runoob.com/docker/windows-docker-install.html	"windows Docker 安装"

- [mac]: https://www.runoob.com/docker/macos-docker-install.html	"MacOS Docker 安装"

### Docker 换源配置

- 方案1: 修改daemon配置文件 /etc/docker/daemon.json ：

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

  [拿到阿里云上专属于你的加速链接(含win)]: https://cr.console.aliyun.com/cn-hangzhou/instances/mirrors

- 方案3: 

  [别人的博客]: https://www.runoob.com/docker/docker-mirror-acceleration.html

  

### [MariaDB](https://hub.docker.com/r/_/mariadb/) container 

  - 拉取MariaDB 镜像(dj-mariadb)    &&  创建一个数据库用户(用户名： domjudge 密码rootpw) 配置一些环境变量

    ```ssh
    docker run -it --name dj-mariadb -e MYSQL_ROOT_PASSWORD=rootpw -e MYSQL_USER=domjudge -e MYSQL_PASSWORD=djpw -e MYSQL_DATABASE=domjudge -p 13306:3306 mariadb --max-connections=10000
    ```

      

### DOMserver container   

- 拉取DOMserver镜像 

  ```
  docker run --link dj-mariadb:mariadb -it -e MYSQL_HOST=mariadb -e MYSQL_USER=domjudge -e MYSQL_DATABASE=domjudge -e MYSQL_PASSWORD=djpw -e MYSQL_ROOT_PASSWORD=rootpw -p 12345:80 --name domserver domjudge/domserver:latest
  ```

  其中 -p 12345:80 代表把容器内的80端口映射到本地12345端口

- Domserver 的admin和judgehost用户的初始密码应该在启动时显示出来，如果没有显示，可以试试这个：

  ```ssh
  docker exec -it domserver cat /opt/domjudge/domserver/etc/initial_admin_password.secret
  docker exec -it domserver cat /opt/domjudge/domserver/etc/restapi.secret
  ```

- 

  



