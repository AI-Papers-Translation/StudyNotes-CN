# [MYSQL数据库引擎区别详解](https://www.cnblogs.com/zhangjinghe/p/7599988.html)

## **数据库引擎介绍**

MySQL数据库引擎取决于MySQL在安装的时候是如何被编译的。要添加一个新的引擎，就必须重新编译MYSQL。在缺省情况下，MYSQL支持三个引擎：ISAM、MYISAM和HEAP。另外两种类型INNODB和BERKLEY（BDB），也常常可以使用。如果技术高超，还可以使用MySQL+API自己做一个引擎。下面介绍几种数据库引擎：

### **ISAM**

ISAM是一个定义明确且历经时间考验的数据表格管理方法，它在设计之时就考虑到 数据库被查询的次数要远大于更新的次数。因此，ISAM执行读取操作的速度很快，而且不占用大量的内存和存储资源。ISAM的两个主要不足之处在于，它不 支持事务处理，也不能够容错：如果你的硬盘崩溃了，那么数据文件就无法恢复了。如果你正在把ISAM用在关键任务应用程序里，那就必须经常备份你所有的实 时数据，通过其复制特性，MYSQL能够支持这样的备份应用程序。

### **MyISAM**

MyISAM是MySQL的ISAM扩展格式和缺省的数据库引擎。除了提供ISAM里所没有的索引和字段管理的大量功能，MyISAM还使用一种表格锁定的机制，来优化多个并发的读写操作，其代价是你需要经常运行OPTIMIZE TABLE命令，来恢复被更新机制所浪费的空间。MyISAM还有一些有用的扩展，例如用来修复数据库文件的MyISAMCHK工具和用来恢复浪费空间的 MyISAMPACK工具。MYISAM强调了快速读取操作，这可能就是为什么MySQL受到了WEB开发如此青睐的主要原因：在WEB开发中你所进行的大量数据操作都是读取操作。所以，大多数虚拟主机提供商和INTERNET平台提供商只允许使用MYISAM格式。MyISAM格式的一个重要缺陷就是不能在表损坏后恢复数据。

### **HEAP**

：HEAP允许只驻留在内存里的临时表格。驻留在内存里让HEAP要比ISAM和MYISAM都快，但是它所管理的数据是不稳定的，而且如果在关机之前没有进行保存，那么所有的数据都会丢失。在数据行被删除的时候，HEAP也不会浪费大量的空间。HEAP表格在你需要使用SELECT表达式来选择和操控数据的时候非常有用。要记住，在用完表格之后就删除表格。
  **InnoDB**：InnoDB数据库引擎都是造就MySQL灵活性的技术的直接产品，这项技术就是MYSQL+API。在使用MYSQL的时候，你所面对的每一个挑战几乎都源于ISAM和MyISAM数据库引擎不支持事务处理（transaction process）也不支持外来键。尽管要比ISAM和 MyISAM引擎慢很多，但是InnoDB包括了对事务处理和外来键的支持，这两点都是前两个引擎所没有的。如前所述，如果你的设计需要这些特性中的一者 或者两者，那你就要被迫使用后两个引擎中的一个了。
  如果感觉自己的确技术高超，你还能够使用MySQL+API来创建自己的数据库引擎。这个API为你提供了操作字段、记录、表格、数据库、连接、安全帐号的功能，以及建立诸如MySQL这样DBMS所需要的所有其他无数功能。深入讲解API已经超出了本文的范围，但是你需要了解MySQL+API的存在及其可交换引擎背后的技术，这一点是很重要的。估计这个插件式数据库引擎的模型甚至能够被用来为MySQL创建本地的XML提供器（XML provider）。（任何读到本文的MySQL+API开发人员可以把这一点当作是个要求。）

### **MyISAM与InnoDB的区别**

　　InnoDB和MyISAM是许多人在使用MySQL时最常用的两个表类型，这两个表类型各有优劣，视具体应用而定。基本的差别为：MyISAM类型不支持事务处理等高级处理，而InnoDB类型支持。MyISAM类型的表强调的是性能，其执行数度比InnoDB类型更快，但是不提供事务支持，而InnoDB提供事务支持已经外部键等高级数据库功能。

### **以下是一些细节和具体实现的差别：**

1.InnoDB不支持FULLTEXT类型的索引。

2.InnoDB 中不保存表的具体行数，也就是说，执行select count(*) fromtable时，InnoDB要扫描一遍整个表来计算有多少行，但是MyISAM只要简单的读出保存好的行数即可。注意的是，当count(*)语句包含where条件时，两种表的操作是一样的。

3.对于AUTO_INCREMENT类型的字段，InnoDB中必须包含只有该字段的索引，但是在MyISAM表中，可以和其他字段一起建立联合索引。

4.DELETE FROM table时，InnoDB不会重新建立表，而是一行一行的删除。

5.LOAD TABLE FROMMASTER操作对InnoDB是不起作用的，解决方法是首先把InnoDB表改成MyISAM表，导入数据后再改成InnoDB表，但是对于使用的额外的InnoDB特性(例如外键)的表不适用。

另外，InnoDB表的行锁也不是绝对的，假如在执行一个SQL语句时MySQL不能确定要扫描的范围，InnoDB表同样会锁全表，例如updatetable set num=1 where name like “a%”
两种类型最主要的差别就是Innodb支持事务处理与外键和行级锁.而MyISAM不支持.所以MyISAM往往就容易被人认为只适合在小项目中使用。
我作为使用MySQL的用户角度出发，Innodb和MyISAM都是比较喜欢的，但是从我目前运维的数据库平台要达到需求：99.9%的稳定性，方便的扩展性和高可用性来说的话，MyISAM绝对是我的首选。

**原因如下：
** 

1、首先我目前平台上承载的大部分项目是读多写少的项目，而MyISAM的读性能是比Innodb强不少的。

2、MyISAM的索引和数据是分开的，并且索引是有压缩的，内存使用率就对应提高了不少。能加载更多索引，而Innodb是索引和数据是紧密捆绑的，没有使用压缩从而会造成Innodb比MyISAM体积庞大不小。

3、从平台角度来说，经常隔1，2个月就会发生应用开发人员不小心update一个表where写的范围不对，导致这个表没法正常用了，这个时候MyISAM的优越性就体现出来了，随便从当天拷贝的压缩包取出对应表的文件，随便放到一个数据库目录下，然后dump成sql再导回到主库，并把对应的binlog补上。如果是Innodb，恐怕不可能有这么快速度，别和我说让Innodb定期用导出xxx.sql机制备份，因为我平台上最小的一个数据库实例的数据量基本都是几十G大小。

4、从我接触的应用逻辑来说，select count(*) 和order by是最频繁的，大概能占了整个sql总语句的60%以上的操作，而这种操作Innodb其实也是会锁表的，很多人以为Innodb是行级锁，那个只是where对它主键是有效，非主键的都会锁全表的。

5、还有就是经常有很多应用部门需要我给他们定期某些表的数据，MyISAM的话很方便，只要发给他们对应那表的frm.MYD,MYI的文件，让他们自己在对应版本的数据库启动就行，而Innodb就需要导出xxx.sql了，因为光给别人文件，受字典数据文件的影响，对方是无法使用的。

6、如果和MyISAM比insert写操作的话，Innodb还达不到MyISAM的写性能，如果是针对基于索引的update操作，虽然MyISAM可能会逊色Innodb,但是那么高并发的写，从库能否追的上也是一个问题，还不如通过多实例分库分表架构来解决。
*

*7、如果是用MyISAM的话，merge引擎可以大大加快应用部门的开发速度，他们只要对这个merge表做一些selectcount(*)操作，非常适合大项目总量约几亿的rows某一类型(如日志，调查统计)的业务表。
当然Innodb也不是绝对不用，用事务的项目如模拟炒股项目，我就是用Innodb的，活跃用户20多万时候，也是很轻松应付了，因此我个人也是很喜欢Innodb的，只是如果从数据库平台应用出发，我还是会首MyISAM。
另外，可能有人会说你MyISAM无法抗太多写操作，但是我可以通过架构来弥补，说个我现有用的数据库平台**容量：**主从数据总量在几百T以上，每天十多亿pv的动态页面，还有几个大项目是通过数据接口方式调用未算进pv总数，(其中包括一个大项目因为初期memcached没部署,导致单台数据库每天处理9千万的查询)。而我的整体数据库服务器平均负载都在0.5-1左右。

**一般来说，MyISAM适合：
**

(1)做很多count 的计算；
(2)插入不频繁，查询非常频繁；
(3)没有事务。

**InnoDB适合：
**

(1)可靠性要求比较高，或者要求事务；
(2)表更新和查询都相当的频繁，并且表锁定的机会比较大的情况指定数据引擎的创建
让所有的灵活性成为可能的开关是提供给ANSI SQL的MySQL扩展——TYPE参数。MySQL能够让你在表格这一层指定数据库引擎，所以它们有时候也指的是table formats。下面的示例代码表明了如何创建分别使用MyISAM、ISAM和HEAP引擎的表格。要注意，创建每个表格的代码是相同的，除了最后的 TYPE参数，这一参数用来指定数据引擎。

**以下为引用的内容：
**

复制代码代码如下:


CREATE TABLE tblMyISAM (
id INT NOT NULL AUTO_INCREMENT,
PRIMARY KEY (id),
value_a TINYINT
) TYPE=MyISAM
CREATE TABLE tblISAM (
id INT NOT NULL AUTO_INCREMENT,
PRIMARY KEY (id),
value_a TINYINT
) TYPE=ISAM
CREATE TABLE tblHeap (
id INT NOT NULL AUTO_INCREMENT,
PRIMARY KEY (id),
value_a TINYINT
) TYPE=Heap


你也可以使用ALTER TABLE命令，把原有的表格从一个引擎移动到另一个引擎。下面的代码显示了如何使用ALTER TABLE把MyISAM表格移动到InnoDB的引擎：

 

**以下为引用的内容：
**

复制代码代码如下:


ALTER TABLE tblMyISAM CHANGE TYPE=InnoDB


MySQL用三步来实现这一目的。首先，这个表格的一个副本被创建。然后，任何输入数据的改变都被排入队列，同时这个副本被移动到另一个引擎。最后，任何排入队列的数据改变都被送交到新的表格里，而原来的表格被删除。

复制代码代码如下:


ALTER TABLE捷径


如果只是想把表格从ISAM更新为MyISAM，你可以使用MySQL_convert_table_format命令，而不需要编写ALTER TABLE表达式。

 

你可以使用SHOW TABLE命令（这是MySQL对ANSI标准的另一个扩展）来确定哪个引擎在管理着特定的表格。SHOW TABLE会返回一个带有多数据列的结果集，你可以用这个结果集来查询获得所有类型的信息：数据库引擎的名称在Type字段里。下面的示例代码说明了 SHOW TABLE的用法：

复制代码代码如下:


SHOW TABLE STATUS FROM tblInnoDB


 你可以用SHOW CREATE TABLE [TableName]来取回SHOW TABLE能够取回的信息。 
一般情况下，MySQL会默认提供多种存储引擎，可以通过下面的查看:
（1）看你的MySQL现在已提供什么存储引擎: mysql> show engines；
（2）看你的MySQL当前默认的存储引擎: mysql> show variables like '%storage_engine%'；
（3）你要看某个表用了什么引擎(在显示结果里参数engine后面的就表示该表当前用的存储引擎): mysql> show create table 表名；
最后，如果你想使用没有被编译成MySQL也没有被激活的引擎，那是没有用的，MySQL不会提示这一点。而它只会给你提供一个缺省格式（MyISAM）的表格。除了使用缺省的表格格式外，还有办法让MySQL给出错误提示，但是就现在而言，如果不能肯定特定的数据库引擎是否可用的话，你要使用SHOW TABLE来检查表格格式。
**更多的选择意味着更好的性能
**用于特定表格的引擎都需要重新编译和追踪，考虑到这种的额外复杂性，为什么你还是想要使用非缺省的数据库引擎呢？答案很简单：要调整数据库来满足你的要求。
可以肯定的是，MyISAM的确快，但是如果你的逻辑设计需要事务处理，你就可以自由使用支持事务处理的引擎。进一步讲，由于MySQL能够允许你在表格这一层应用数据库引擎，所以你可以只对需要事务处理的表格来进行性能优化，而把不需要事务处理的表格交给更加轻便的MyISAM引擎。对于 MySQL而言，灵活性才是关键。

 

**性能测试
**所有的性能测试在：Micrisoft window xp sp2 ， Intel(R) Pentinum(R) M processor 1.6oGHz 1G 内存的电脑上测试。
测试方法：连续提交10个query， 表记录总数：38万 ， 时间单位 s
引擎类型MyISAMInnoDB 性能相差
count 0.00083573.01633609
查询主键  0.005708 0.157427.57
查询非主键  24.01 80.37 3.348
更新主键  0.008124 0.8183100.7
更新非主键  0.004141 0.02625 6.338
插入  0.004188 0.369488.21
（1）加了索引以后，对于MyISAM查询可以加快：4 206.09733倍，对InnoDB查询加快510.72921倍，同时对MyISAM更新速度减慢为原来的1/2，InnoDB的更新速度减慢为原来的1/30。要看情况决定是否要加索引，比如不查询的log表，不要做任何的索引。
（2）如果你的数据量是百万级别的，并且没有任何的事务处理，那么用MyISAM是性能最好的选择。
（3）InnoDB表的大小更加的大，用MyISAM可省很多的硬盘空间。

在我们测试的这个38w的表中，表占用空间的情况如下：

引擎类型MyISAM  InnoDB
数据 53,924 KB  58,976 KB
索引 13,640 KB  21,072 KB

占用总空间 67,564 KB  80,048 KB

另外一个176W万记录的表， 表占用空间的情况如下：
引擎类型MyIsam  InnorDB
数据 56,166 KB  90,736 KB
索引 67,103 KB  88,848 KB

占用总空间 123,269 KB179,584 KB

**其他**
  MySQL 官方对InnoDB是这样解释的：InnoDB给MySQL提供了具有提交、回滚和崩溃恢复能力的事务安全（ACID兼容）存储引擎。InnoDB锁定在行级并且也在SELECT语句提供一个Oracle风格一致的非锁定读，这些特色增加了多用户部署和性能。没有在InnoDB中扩大锁定的需要，因为在InnoDB中行级锁定适合非常小的空间。InnoDB也支持FOREIGN KEY强制。在SQL查询中，你可以自由地将InnoDB类型的表与其它MySQL的表的类型混合起来，甚至在同一个查询中也可以混合。
  InnoDB是为处理巨大数据量时的最大性能设计，它的CPU效率可能是任何其它基于磁盘的关系数据库引擎所不能匹敌的。
  InnoDB存储引擎被完全与MySQL服务器整合，InnoDB存储引擎为在主内存中缓存数据和索引而维持它自己的缓冲池。InnoDB存储它的表＆索引在一个表空间中，表空间可以包含数个文件（或原始磁盘分区）。这与MyISAM表不同，比如在MyISAM表中每个表被存在分离的文件中。InnoDB 表可以是任何尺寸，即使在文件尺寸被限制为2GB的操作系统上。
  InnoDB默认地被包含在MySQL二进制分发中。Windows Essentials installer使InnoDB成为Windows上MySQL的默认表。
  InnoDB被用来在众多需要高性能的大型数据库站点上产生。著名的Internet新闻站点Slashdot.org运行在InnoDB上。 Mytrix, Inc.在InnoDB上存储超过1TB的数据，还有一些其它站点在InnoDB上处理平均每秒800次插入/更新的.