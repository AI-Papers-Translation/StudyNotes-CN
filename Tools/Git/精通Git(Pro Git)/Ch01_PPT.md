---
marp: true
theme: gitppt
size: 4:3
header: '**CEDU3D**'
footer: 'zYx.Tom'
paginate: true
---
<!--
_class: lead gaia
_paginate: false
-->

# Pro Git

## Git 入门

---

## 版本控制

* 本地版本控制系统
* 集中式版本控制系统
* 分布式版本控制系统

---

## 本地版本控制系统
<!--
_class: lead
-->
![本地版本控制图解 bg right:85% fit](https://git-scm.com/book/en/v2/images/local.png)

---

## 集中式版本控制系统
<!--
_class: lead
-->
![集中化的版本控制图解 bg right:85% fit](https://git-scm.com/book/en/v2/images/centralized.png)

---

### 集中式版本控制系统（CVCS）

* 集中式版本控制系统（Centralized Version Control Systems，CVCS）
  * CVS、SubVersion、Perforce
  * 单一的集中管理的服务器，保存所有文件的修订版本
  * 协同工作时都通过客户端连接到服务器，提取最新的提交
  * 优点：权限管理、资源同步、协同工作、降低维护成本
  * 缺点：资源同步、协同工作

---

## 分布式版本控制系统
<!--
_class: lead
-->
![分布式版本控制图解 bg right:85% fit](https://git-scm.com/book/en/v2/images/distributed.png)

---

### 分布式版本管理系统（DVCS）

* 分布式版本管理系统（Distributed Version Control System，DVCS）
  * Git、Mercurial、Bazaar、Darcs
  * 客户端将服务器的代码仓库完整镜像到本地
    * 包括完整的历史记录
  * 客户端与服务器实现数据同步
    * 任何服务器发生故障后，都可以使用本地仓库完成恢复
    * 不同客户端可以与不同服务器交互

---

## Git简史

* 速度够快
* 设计简单
* 完全分布式
* 对非线性开发模式的支持
* 管理超大规模项目

---
<!--
_class: lead
-->
### 传统｜存储与初始版本的差异

![存储每个文件与初始版本的差异 bg right:85% fit](https://git-scm.com/book/en/v2/images/deltas.png)

---
<!--
_class: lead
-->

### 创新｜存储项目改变的快照

![Git 存储项目随时间改变的快照 bg right:85% fit](https://git-scm.com/book/en/v2/images/snapshots.png)

---

### 本地完成操作

* Git只对本地操作执行命令，保证了处理速度
* Git存储前计算校验和，引用校验和，保证了数据完整性
  * 本地存储无法绕过Git进行提交
  * 网络传输，保证传输文件不发生错误
* Git只向仓库添加数据
  * 不执行删除数据

---

### Git项目状态与工作流程

项目的三种状态

* 工作区：对项目的某个版本独立提取出来的内容。
* 暂存区：保存了下次需要提交的文件列表信息
* Git仓库目录：保存项目的元数据和对象数据库的地方

Git工作流程

* 状态为“已经修改”：在工作区中修改文件，
* 状态为“已经暂存”：将工作区中修改文件提交到暂存区，
* 状态为“已经更新”：将暂存区的内容提交仓库目录

---
<!--
_class: lead
-->
### 项目工作流程

![工作区、暂存区以及 Git 目录 bg right:85% fit](https://git-scm.com/book/en/v2/images/areas.png)

---

## 命令行模式

Git既有命令行模式，也有GUI模式。

注：建议学习时多使用命令行模式，因为命令行模式可以保证Git命令的全部覆盖

---

## Git配置

* 配置Git变量：`git config`
  * `/etc/gitconfig`：系统级配置
    * 使用 `git config --system` 访问
  * `~/.gitconfig`或 `~/.config/git/config`
    * 用户级配置
    * 使用 `git config --global` 访问
  * `.git/config`
    * 仓库级配置
    * 使用 `git config --local` 访问


注：
1. 系统级配置： Windows 7 在 `C:\ProgramData\Git\config`
2. 小级别覆盖大级别

---

## 配置参数样例

* 配置用户信息
  * `git config --global user.name 'zhuyuanxiang'`
  * `git config --global user.email '526614962@qq.com'`
* 配置文本编辑器
  * `git config --global core.editor vim`
* 查看全部配置信息
  * `git config --list`
    * 可以存在重复配置，使用最后一个配置结果
* 查看单个配置信息
  * `git config user.name`
* 查看配置来源
  * `git config --show-origin user.name`
---

## 获得帮助

命令：

```shell
git help <verb>
git <verb> help
man git-<verb>
```

样例：

```shell
git help config
```
