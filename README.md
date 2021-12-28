# Logistics On Map

The implementation of the Logistics environment on China map.



## 目录结构

```
|-- platform_lib
      |-- README.md
      |-- main.py  // 环境测试主函数
      |-- algo     // 智能体算法实例，每类算法需封装成类，并提供 choose_action 函数
            |-- random_agent.py  // 当前环境下的随机策略智能体
      |-- env      // 仿真环境
            |-- obs_interfaces   // observation 观测类接口
                  |-- observation.py
            |-- simulators       // 模拟器
                  |-- game.py
            |-- rendering        // 环境可视化
                  |-- resources      // 资源文件
                  |-- viewer.py      // 可视化代码（只针对当前的中国地图的物流环境）
            |-- config.json      // 相关配置文件
            |-- chooseenv.py     // 继承自 Jidi
            |-- logisticsenv.py  // 物流仿真器逻辑代码
      |-- map      // 地图数据（Json 格式），记录了每个节点和每条道路的参数
            |-- map_0.json      // 当前在中国地图上的物流相关参数
      |-- utils    // 工具函数（继承自 Jidi）
```



## 地图数据格式

地图数据的 Json 文件共包含以下字段：

1. `n_goods`（`int`）：货物数量；

2. `goods`（`list(dict)`）：每种货物相关的参数，具体如下：

   2.1 `key`（`int`）：货物编号（从0开始）；

   2.2 `volume`（`float`）：货物体积（容量）；

   2.3 `price`（`float`）：货物价格。

3. `n_vertex`（`int`）：节点数量；

4. `vertices`（`list(dict)`）：每个节点相关的参数，具体如下：

   4.1 `key`（`int`）：节点编号（从0开始）；

   4.2 `production`（`list`）：长度=`n_goods`，该节点每种货物的每日生产量；

   4.3 `init_storage`（`list`）：长度=`n_goods`，该节点每种货物的初始库存量；

   4.4 `upper_capacity`（`float`）：该节点库存容量的上限；

   4.5 `store_cost`（`float`）：该节点存储单位体积货物，一日所需要的成本；

   4.6 `loss_cost`（`list`）：长度=`n_goods`，该节点舍弃多余的每种货物时，造成的损失成本；

   4.7 `lambda`（`list`）：长度=`n_goods`，该节点每种货物每日需求量的均值（即泊松分布中的参数lambda）。

5. `is_graph_directed`（`bool`）：该地图的道路为有向道路（`true`）还是无向道路（`false`）；

6. `roads`（`list(dict)`）：每条道路相关的参数，具体如下：

   6.1 `start`（`int`）：起始节点的`key`；

   6.2 `end`（`int`）：终止节点的`key`；

   6.3 `upper_capacity`（`float`）：该道路每日最高的运输容量；

   6.4 `trans_time`（`int/float`）：该道路上运输所需时间（单位为小时，间隔为0.5）；

   6.5 `trans_cost`（`float`）：该道路上运输单位体积货物，一小时所需要的成本；

7. `is_abs_coords`（可选，`bool`）：属性`coords`中的坐标是否为绝对坐标；

8. `coords`（可选，`list`）：长度=`n_vertices`，每个节点在图中的（绝对/相对）坐标，若是绝对坐标则不作任何处理，若是相对坐标则根据画布大小计算对应的绝对坐标；若不指定则由程序自动排列每个节点的位置。



## 主程序说明

### 算法添加

在`algo`中添加了新的智能体算法后，可以在`main.py`中导入该算法，并在`Agent_dict`中加入该算法的名称和对应的类。具体使用时只需将命令行参数`--algo`指定为该算法的名称（或将`main`函数中`algo`参数的默认值修改为该算法的名称）。



### 其余参数

在`main`函数中，除了`algo`参数用于指定测试所用的算法外，还有以下参数：

1. `map`：指定地图编号（默认为0），若在`map`文件夹中添加新地图，需要将名称修改为`map_{No}.json`；
2. `step_per_update`：可视化界面两次更新之间，step的间隔（默认为1）；
3. `silence`：是否在命令行打印环境中的各类参数，便于调试（默认为false，即打印）。



## 注意

当前`map`文件夹中`map_0.json`的地图数据基本均**未调试完成**。只有**道路连接方式**（`roads`）以及每条道路上的**运输时间**（`trans_time`）是确定的（且道路上的运输时间是根据真实导航查询得到的）。

根据先前经验，调整参数时，每个节点的`production`、`init_storage`、`lambda`等参数应设置得**较大**，这样有利于环境的稳定性。

同时，为使环境保持动态平衡，应保证所有节点每种货物的总生产量（`production[i]`之和）**等于**总需求量均值（`lambda[i]`之和）。

