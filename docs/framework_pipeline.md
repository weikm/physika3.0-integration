# 背景
 framework定义了4个各关键概念：World、Scene、Object、Solver，他们的相互关系可简单描述为World包含Scene，Scene包含Object，Solver作用到Object。各关键概念都定义有构造函数、initialize、reset、step、run等改变自身状态的函数（如下表所示），各关键概念通过一致定义的状态节点达成时序对齐的目的。有必要明确，在仿真计算全生命周期中的几个关键状态节点，避免上层算法在实现时的不一致。
|  framework     |  constructor     |  initialize     |  reset     |  step     |  run     |  clear     |  destructor     | 
| ----------- | ----------- | ----------- | ----------- | ----------- | ----------- | ----------- | ----------- | 
| World      | Y      | N      | Y      | Y      | Y      | Y      | Y      | 
| Scene   | Y   | N   | Y   | Y   | Y   | N   | Y   | 
| Object      | Y      | N      | Y      | N      | N      | N      | Y      | 
| Solver   | Y   | Y   | Y   | Y   | Y   | N   | Y   | 
可以看到：
1. 仅Object没有step和run函数，因为在仿真过程中Object的状态变化是由与其关联的Solver的step和run驱动的。
2. 仅World有clear函数，因为World是个单例，运行时仅有一个World实例，World通过clear函数来达到清空内容的目的。
3. 仅Solver有initialize函数，因为Solver通常有一堆配置项，Solver在initialize函数调用前设置需要的配置，通过initialize函数将Solver达到计算ready状态。
4. 其他几个函数都是所有类都共有的，稍后详细说明。
# 生命周期
我们定义仿真计算生命周期的几个关键状态：
1. **新创建（constructed）:** 对象刚刚创建，处于默认或空的状态，不能直接用于计算
2. **已初始化（initialized）：** 已经按照前置配置参数初始化到计算ready状态
3. **已清空（cleared）：** 包含的内容已全被移除
4. **仿真中（running）：** 循环调用step函数中，或run函数已调用未返回
5. **已结束（finished）：** 仿真循环已结束
6. **已销毁（destructed）：** 生命周期结束，对象被销毁
生命周期的状态流转图如下所示：
![alt text](../data/readme_resources/framework_pipeline.png "framework pipeline")