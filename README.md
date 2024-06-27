<a name="Y6ch9"></a>
### 1. 常用的两类HMR方法 
- 2D图像或2D关键点数据中恢复3D人体姿势和形状的方法。一种常见的方法是使用人体模型的参数来表示姿势和形状，例如SMPL模型。然后，利用回归或优化技术来解决这个问题。
   - **回归方法**：通过训练一个模型，将2D图像或关键点作为输入，直接预测出人体模型的参数，包括姿势和形状参数。这个过程可以看作是从2D数据到3D参数的一个直接映射。
   - **优化方法**：另一种方法是利用优化技术，在给定的2D数据下，通过调整人体模型的参数来最小化与观测数据之间的重投影误差。通常，这个过程涉及到一个迭代优化算法，不断地调整模型参数直到达到最小化误差的目标。
- **Motivation**
   - 解决现有方法的mesh alignment的问题
- **Challenge**
   - DDIM inverse 
<a name="IjcEi"></a>
### 2. SMPL
<a name="bEo4R"></a>
#### 2.1 总概

- 建模人体为姿势参数![](https://cdn.nlark.com/yuque/__latex/a4f585090e47f94b9474d28e941f390d.svg#card=math&code=%28pose%29%20%CE%B8&id=Ap2uz)和![](https://cdn.nlark.com/yuque/__latex/7f09e53ea228a8bcc6d9525b08eec922.svg#card=math&code=%28shape%29%20%CE%B2&id=NUHC3)两种参数，姿势参数是一个![](https://cdn.nlark.com/yuque/__latex/1edbd86dead2235483094fb979b3f8aa.svg#card=math&code=24%5Ctimes3&id=jmOdx)的矩阵。形状参数是一个长度为![](https://cdn.nlark.com/yuque/__latex/134c802fc5f0924cf1ea838feeca6c5e.svg#card=math&code=10&id=KTyES)的向量
- SMPL模型定义了从人体参数到身体网格的映射![](https://cdn.nlark.com/yuque/__latex/ed3ee195d1e328a5945627374d2f080d.svg#card=math&code=%5Cmu%20%28%CE%B8%2C%20%CE%B2%29&id=C2CfZ)，body mesh 定义为![](https://cdn.nlark.com/yuque/__latex/6f5dde593f0bc27956e14b5eaec2ed17.svg#card=math&code=M&id=tqbfU)，维度为![](https://cdn.nlark.com/yuque/__latex/bea3af74c42709c7acc62a17f0f3c283.svg#card=math&code=6982%5Ctimes3&id=cqftz)，对一个给定的mesh ![](https://cdn.nlark.com/yuque/__latex/6f5dde593f0bc27956e14b5eaec2ed17.svg#card=math&code=M&id=pnS65)，我们可以通过一个预训练好的回归器![](https://cdn.nlark.com/yuque/__latex/a36915ecf0b5605493f5aeaf1480a9ac.svg#card=math&code=W&id=iLQPH)回归一个3D Body Joints：![](https://cdn.nlark.com/yuque/__latex/c72bfa9f927dc02a3f6010360026ceb3.svg#card=math&code=J%3DWM&id=bEnqS)
<a name="iSL2h"></a>
#### 2.2 完成一个什么样的事情

- 我们想从观测到的值y反推影响它的隐变量
- 即我们想从图片的2D关键点等信息恢复出SMPL的参数{θ, β}
- 一般通过优化的手段解决这个问题，但是最小化损失函数实现这个目标，现在输入是2D图片I和对应SMPL估算得到的两个系数值{θ, β}，我们要在观测值y的guidance下改进这两个SMPL系数。
<a name="EEebx"></a>
### 3.实验细节

- 使用`6D`表示`3D rotation`
- `inverse`的隐变量![](https://cdn.nlark.com/yuque/__latex/48d05334b5b0710d63edb6b4b3ac631c.svg#card=math&code=x_0&id=vi0jE)设置维度为144维的向量
- `denosing model`由3层MLP组成，在时间维度![](https://cdn.nlark.com/yuque/__latex/cead1760d9d5723460c4b8d4028f113a.svg#card=math&code=t&id=pmaZt)和特征维度![](https://cdn.nlark.com/yuque/__latex/79ce3c7a71877c2ff01695e38ade43ca.svg#card=math&code=s&id=fdvlT)上设置有一个condition
   - 输入包含姿势参数![](https://cdn.nlark.com/yuque/__latex/125a97181496f444fc00b132f2b869f1.svg#card=math&code=%CE%B8&id=Sk9tb)时间步长![](https://cdn.nlark.com/yuque/__latex/cead1760d9d5723460c4b8d4028f113a.svg#card=math&code=t&id=muxji)和图像特征![](https://cdn.nlark.com/yuque/__latex/b891664b42113aee13f0bac25eb998e5.svg#card=math&code=c&id=j8Na1)的噪声样本![](https://cdn.nlark.com/yuque/__latex/21c4616d966dca0cdc4d982b04f94933.svg#card=math&code=x_t&id=gpeXc)
   - 首先使用线性层投影![](https://cdn.nlark.com/yuque/__latex/21c4616d966dca0cdc4d982b04f94933.svg#card=math&code=x_t&id=dyPoe)到第一个特征维度上![](https://cdn.nlark.com/yuque/__latex/f21c3e5ed33c1ec90a981a2360d0fcbf.svg#card=math&code=h%5E%7B%281%29%7D&id=L7wup)作为第一个MLP输入
   - 然后对每个MLP块的输入特征通过`scaling`和`shifting`的方式得到![](https://cdn.nlark.com/yuque/__latex/29d6c48f17b081aa85a783db2ba4449d.svg#card=math&code=h%5E%7B%28i%29%7D_%7Bt%7D%3Dt_sh%5E%7B%28i%29%7D%2Bt_b&id=Dd1tE)(![](https://cdn.nlark.com/yuque/__latex/0bf06563362bd1d9af020ad0b1b89857.svg#card=math&code=t_s&id=JX2PQ)和![](https://cdn.nlark.com/yuque/__latex/b03bb17222c6a3ce023dc9c64ff465dd.svg#card=math&code=t_b&id=jsiO7)是一个输出维度是![](https://cdn.nlark.com/yuque/__latex/b26e838a55964360ec43dfaf01a2fb13.svg#card=math&code=2%5Ctimes144&id=RIfxz)，计算公式![](https://cdn.nlark.com/yuque/__latex/ea91a30007f1c3f9e90d58ee68e3e8a0.svg#card=math&code=%28t_s%2Ct_b%29%3DMLP%28%5Cphi%28t%29%29&id=kC1kG)) 
   - 每个MLP都会根据图像特征进行条件设置，方法是![](https://cdn.nlark.com/yuque/__latex/0cb3f07964243731add922f7d0ac3a6e.svg#card=math&code=concate%28h%5E%7B%28i%29%7D%2C%20c%29%29&id=OJXWl)


