import numpy as np
import matplotlib.pyplot as plt
from PySVM.SVM.SVC import BiLinearSVC, LinearSVC, KernelSVC, NuSVC
from sklearn.datasets import make_classification, make_circles, make_moons
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

np.random.seed(2022)

fig, axes = plt.subplots(2, 2, figsize=(12, 10)) #fig (Figure)：代表整个大画板（窗口）,axes (Axes)：代表里面的小格子
                                                 #参数 (2, 2) 表示创建一个 2 行 2 列 的网格（如果你想操作左上角的图，你需要写 axes[0, 0]；操作右下角的图，要写 axes[1, 1]）
axes = axes.flatten() #axes = axes.flatten()，原本 axes 是个 2 * 2 的矩阵，通过 .flatten()，它被转换成了一个一维列表

datasets = [ #datasets 是一个列表，里面的每个元素是一个元组 (Tuple)： ("数据集名称", (特征矩阵 X, 标签向量 y))
    ("Linearly Separable", make_classification(n_samples=200, n_features=2, n_redundant=0, 
                                               n_informative=2, n_clusters_per_class=1, 
                                               random_state=2022)), #数据被明显地分为两堆，中间可以用一条直线切开
    ("Circles", make_circles(n_samples=200, noise=0.1, factor=0.5, random_state=2022)), #一类点在中心，另一类点像圆环一样包围在外面
    ("Moons", make_moons(n_samples=200, noise=0.1, random_state=2022)), #两类数据像两个交错的弯月
    ("XOR-like", make_classification(n_samples=200, n_features=2, n_redundant=0, 
                                     n_informative=2, n_clusters_per_class=1, 
                                     class_sep=1.0, random_state=42)) #数据分布在对角区域，模拟经典的异或问题（XOR）
]

for idx, (name, (X, y)) in enumerate(datasets):
    ax = axes[idx]
    
    scaler = StandardScaler() #创建一个标准化器实例
    X_scaled = scaler.fit_transform(X) #fit (拟合)：计算数据 X 中每一列特征的平均值 mu 和标准差 sigma
                                       #transform (转换)：应用上面的公式，把原始数据 X 转化为标准化后的数据 X_scaled
    
    model = BiLinearSVC(
        C=1.0,
        max_iter=1000,
        tol=1e-5,
        cache_size=256
    )
    model.fit(X_scaled, y)
    
    
    h = 0.02 #h 是步长
    x_min, x_max = X_scaled[:, 0].min() - 0.5, X_scaled[:, 0].max() + 0.5 #X 轴的范围
    y_min, y_max = X_scaled[:, 1].min() - 0.5, X_scaled[:, 1].max() + 0.5 #Y 轴的范围
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h)) #xx 是所有格子的横坐标，yy 是所有格子的纵坐标
    
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()]) #xx 和 yy 原本是二维矩阵（像一张方格纸）。.ravel() 的作用是将它们"拉平"成一维的长向量
                                                     #np.c_[...]：这是按列连接（Column-wise stacking）。它把拉平后的 X 坐标和 Y 坐标像拉链一样对齐，组合成一个个坐标点 (x, y)
                                                     #model.predict(...) 对这成千上万个网格点逐一进行判断，如果模型认为某个点属于类别 1，就返回 1；如果认为属于类别 0，就返回 0
    Z = Z.reshape(xx.shape) #把这串长向量重新折叠回原来的二维矩阵形状
    
    ax.contourf(xx, yy, Z, alpha=0.3, cmap=plt.cm.RdYlBu) #ax.contourf(...)：填充背景颜色
    ax.contour(xx, yy, Z, colors='black', linewidths=1, alpha=0.5) #ax.contour(...)：画出边界线
    
    scatter = ax.scatter(X_scaled[:, 0], X_scaled[:, 1], c=y, cmap=plt.cm.RdYlBu, 
                         edgecolors='black', linewidths=1, s=50) #把原始的 200 个数据点画上去
    
    train_score = model.score(X_scaled, y)
    ax.set_title(f'{name}\nAccuracy: {train_score:.3f}', fontsize=10)
    ax.set_xlabel('Feature 1')
    ax.set_ylabel('Feature 2')
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('bi_linear_svc_classification.png', dpi=150, bbox_inches='tight')
plt.show()

print("Visualization saved as 'bi_linear_svc_classification.png'")

# Test LinearSVC with multiclass datasets
print("\nTesting LinearSVC with multiclass datasets...")

# Create multiclass datasets for visualization
# Note: With 2 informative features, max classes = 2^2 = 4
# For 5+ classes, we use 3 informative features but only visualize first 2 dimensions
multiclass_datasets = [
    ("3 Classes", make_classification(n_samples=300, n_features=2, n_redundant=0,
                                      n_informative=2, n_clusters_per_class=1,
                                      n_classes=3, random_state=2022)),
    ("4 Classes", make_classification(n_samples=400, n_features=2, n_redundant=0,
                                      n_informative=2, n_clusters_per_class=1,
                                      n_classes=4, random_state=2022)),
    ("5 Classes", make_classification(n_samples=500, n_features=3, n_redundant=0,
                                       n_informative=3, n_clusters_per_class=1,
                                       n_classes=5, random_state=2022)),
    ("6 Classes", make_classification(n_samples=600, n_features=3, n_redundant=0,
                                       n_informative=3, n_clusters_per_class=1,
                                       n_classes=6, random_state=2022))
]

fig2, axes2 = plt.subplots(2, 2, figsize=(12, 10))
axes2 = axes2.flatten()

for idx, (name, (X, y)) in enumerate(multiclass_datasets):
    ax = axes2[idx]
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # For datasets with more than 2 features, use only first 2 for visualization
    X_vis = X_scaled[:, :2] if X_scaled.shape[1] > 2 else X_scaled
    
    # Test with OVR strategy
    model = LinearSVC(
        C=1.0,
        max_iter=1000,
        tol=1e-5,
        cache_size=256,
        multiclass="ovr"
    )
    model.fit(X_scaled, y)  # Train on all features
    
    h = 0.02
    x_min, x_max = X_vis[:, 0].min() - 0.5, X_vis[:, 0].max() + 0.5
    y_min, y_max = X_vis[:, 1].min() - 0.5, X_vis[:, 1].max() + 0.5
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    
    # For prediction on mesh, use first 2 features only for visualization
    # Create full feature vectors with zeros for additional dimensions
    if X_scaled.shape[1] > 2:
        mesh_points = np.zeros((xx.ravel().shape[0], X_scaled.shape[1]))
        mesh_points[:, :2] = np.c_[xx.ravel(), yy.ravel()]
        Z = model.predict(mesh_points)
    else:
        Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    
    ax.contourf(xx, yy, Z, alpha=0.3, cmap=plt.cm.RdYlBu)
    ax.contour(xx, yy, Z, colors='black', linewidths=1, alpha=0.5)
    
    scatter = ax.scatter(X_vis[:, 0], X_vis[:, 1], c=y, cmap=plt.cm.RdYlBu,
                         edgecolors='black', linewidths=1, s=50)
    
    train_score = model.score(X_scaled, y)  # Score on all features
    ax.set_title(f'{name} (OVR)\nAccuracy: {train_score:.3f}', fontsize=10)
    ax.set_xlabel('Feature 1')
    ax.set_ylabel('Feature 2')
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('linear_svc_ovr_classification.png', dpi=150, bbox_inches='tight')
plt.show()

print("LinearSVC (OVR) visualization saved as 'linear_svc_ovr_classification.png'")

# Test with OVO strategy
fig3, axes3 = plt.subplots(2, 2, figsize=(12, 10))
axes3 = axes3.flatten()

for idx, (name, (X, y)) in enumerate(multiclass_datasets):
    ax = axes3[idx]
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # For datasets with more than 2 features, use only first 2 for visualization
    X_vis = X_scaled[:, :2] if X_scaled.shape[1] > 2 else X_scaled
    
    # Test with OVO strategy
    model = LinearSVC(
        C=1.0,
        max_iter=1000,
        tol=1e-5,
        cache_size=256,
        multiclass="ovo"
    )
    model.fit(X_scaled, y)  # Train on all features
    
    h = 0.02
    x_min, x_max = X_vis[:, 0].min() - 0.5, X_vis[:, 0].max() + 0.5
    y_min, y_max = X_vis[:, 1].min() - 0.5, X_vis[:, 1].max() + 0.5
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    
    # For prediction on mesh, use first 2 features only for visualization
    # Create full feature vectors with zeros for additional dimensions
    if X_scaled.shape[1] > 2:
        mesh_points = np.zeros((xx.ravel().shape[0], X_scaled.shape[1]))
        mesh_points[:, :2] = np.c_[xx.ravel(), yy.ravel()]
        Z = model.predict(mesh_points)
    else:
        Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    
    ax.contourf(xx, yy, Z, alpha=0.3, cmap=plt.cm.RdYlBu)
    ax.contour(xx, yy, Z, colors='black', linewidths=1, alpha=0.5)
    
    scatter = ax.scatter(X_vis[:, 0], X_vis[:, 1], c=y, cmap=plt.cm.RdYlBu,
                         edgecolors='black', linewidths=1, s=50)
    
    train_score = model.score(X_scaled, y)  # Score on all features
    ax.set_title(f'{name} (OVO)\nAccuracy: {train_score:.3f}', fontsize=10)
    ax.set_xlabel('Feature 1')
    ax.set_ylabel('Feature 2')
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('linear_svc_ovo_classification.png', dpi=150, bbox_inches='tight')
plt.show()

print("LinearSVC (OVO) visualization saved as 'linear_svc_ovo_classification.png'")

# Test KernelSVC with RBF kernel on binary classification datasets
print("\nTesting KernelSVC (RBF) with binary classification datasets...")

fig4, axes4 = plt.subplots(2, 2, figsize=(12, 10))
axes4 = axes4.flatten()

for idx, (name, (X, y)) in enumerate(datasets):
    ax = axes4[idx]
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    model = KernelSVC(
        C=1.0,
        kernel='rbf',
        gamma='scale',
        max_iter=1000,
        tol=1e-5,
        cache_size=256,
        multiclass="ovr"
    )
    model.fit(X_scaled, y)
    
    h = 0.02
    x_min, x_max = X_scaled[:, 0].min() - 0.5, X_scaled[:, 0].max() + 0.5
    y_min, y_max = X_scaled[:, 1].min() - 0.5, X_scaled[:, 1].max() + 0.5
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    
    ax.contourf(xx, yy, Z, alpha=0.3, cmap=plt.cm.RdYlBu)
    ax.contour(xx, yy, Z, colors='black', linewidths=1, alpha=0.5)
    
    scatter = ax.scatter(X_scaled[:, 0], X_scaled[:, 1], c=y, cmap=plt.cm.RdYlBu,
                         edgecolors='black', linewidths=1, s=50)
    
    train_score = model.score(X_scaled, y)
    ax.set_title(f'{name} (RBF Kernel)\nAccuracy: {train_score:.3f}', fontsize=10)
    ax.set_xlabel('Feature 1')
    ax.set_ylabel('Feature 2')
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('kernel_svc_rbf_classification.png', dpi=150, bbox_inches='tight')
plt.show()

print("KernelSVC (RBF) visualization saved as 'kernel_svc_rbf_classification.png'")

# Test KernelSVC with RBF kernel and OVR strategy on multiclass datasets
print("\nTesting KernelSVC (RBF, OVR) with multiclass datasets...")

fig5, axes5 = plt.subplots(2, 2, figsize=(12, 10))
axes5 = axes5.flatten()

for idx, (name, (X, y)) in enumerate(multiclass_datasets):
    ax = axes5[idx]
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # For datasets with more than 2 features, use only first 2 for visualization
    X_vis = X_scaled[:, :2] if X_scaled.shape[1] > 2 else X_scaled
    
    # Test with RBF kernel and OVR strategy
    model = KernelSVC(
        C=1.0,
        kernel='rbf',
        gamma='scale',
        max_iter=1000,
        tol=1e-5,
        cache_size=256,
        multiclass="ovr"
    )
    model.fit(X_scaled, y)  # Train on all features
    
    h = 0.02
    x_min, x_max = X_vis[:, 0].min() - 0.5, X_vis[:, 0].max() + 0.5
    y_min, y_max = X_vis[:, 1].min() - 0.5, X_vis[:, 1].max() + 0.5
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    
    # For prediction on mesh, use first 2 features only for visualization
    # Create full feature vectors with zeros for additional dimensions
    if X_scaled.shape[1] > 2:
        mesh_points = np.zeros((xx.ravel().shape[0], X_scaled.shape[1]))
        mesh_points[:, :2] = np.c_[xx.ravel(), yy.ravel()]
        Z = model.predict(mesh_points)
    else:
        Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    
    ax.contourf(xx, yy, Z, alpha=0.3, cmap=plt.cm.RdYlBu)
    ax.contour(xx, yy, Z, colors='black', linewidths=1, alpha=0.5)
    
    scatter = ax.scatter(X_vis[:, 0], X_vis[:, 1], c=y, cmap=plt.cm.RdYlBu,
                         edgecolors='black', linewidths=1, s=50)
    
    train_score = model.score(X_scaled, y)  # Score on all features
    ax.set_title(f'{name} (RBF, OVR)\nAccuracy: {train_score:.3f}', fontsize=10)
    ax.set_xlabel('Feature 1')
    ax.set_ylabel('Feature 2')
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('kernel_svc_rbf_ovr_classification.png', dpi=150, bbox_inches='tight')
plt.show()

print("KernelSVC (RBF, OVR) visualization saved as 'kernel_svc_rbf_ovr_classification.png'")

# Test KernelSVC with RBF kernel and OVO strategy on multiclass datasets
print("\nTesting KernelSVC (RBF, OVO) with multiclass datasets...")

fig6, axes6 = plt.subplots(2, 2, figsize=(12, 10))
axes6 = axes6.flatten()

for idx, (name, (X, y)) in enumerate(multiclass_datasets):
    ax = axes6[idx]
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # For datasets with more than 2 features, use only first 2 for visualization
    X_vis = X_scaled[:, :2] if X_scaled.shape[1] > 2 else X_scaled
    
    # Test with RBF kernel and OVO strategy
    model = KernelSVC(
        C=1.0,
        kernel='rbf',
        gamma='scale',
        max_iter=1000,
        tol=1e-5,
        cache_size=256,
        multiclass="ovo"
    )
    model.fit(X_scaled, y)  # Train on all features
    
    h = 0.02
    x_min, x_max = X_vis[:, 0].min() - 0.5, X_vis[:, 0].max() + 0.5
    y_min, y_max = X_vis[:, 1].min() - 0.5, X_vis[:, 1].max() + 0.5
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    
    # For prediction on mesh, use first 2 features only for visualization
    # Create full feature vectors with zeros for additional dimensions
    if X_scaled.shape[1] > 2:
        mesh_points = np.zeros((xx.ravel().shape[0], X_scaled.shape[1]))
        mesh_points[:, :2] = np.c_[xx.ravel(), yy.ravel()]
        Z = model.predict(mesh_points)
    else:
        Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    
    ax.contourf(xx, yy, Z, alpha=0.3, cmap=plt.cm.RdYlBu)
    ax.contour(xx, yy, Z, colors='black', linewidths=1, alpha=0.5)
    
    scatter = ax.scatter(X_vis[:, 0], X_vis[:, 1], c=y, cmap=plt.cm.RdYlBu,
                         edgecolors='black', linewidths=1, s=50)
    
    train_score = model.score(X_scaled, y)  # Score on all features
    ax.set_title(f'{name} (RBF, OVO)\nAccuracy: {train_score:.3f}', fontsize=10)
    ax.set_xlabel('Feature 1')
    ax.set_ylabel('Feature 2')
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('kernel_svc_rbf_ovo_classification.png', dpi=150, bbox_inches='tight')
plt.show()

print("KernelSVC (RBF, OVO) visualization saved as 'kernel_svc_rbf_ovo_classification.png'")

# Test NuSVC with RBF kernel on binary classification datasets
print("\nTesting NuSVC (RBF) with binary classification datasets...")

fig7, axes7 = plt.subplots(2, 2, figsize=(12, 10))
axes7 = axes7.flatten()

for idx, (name, (X, y)) in enumerate(datasets):
    ax = axes7[idx]
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    model = NuSVC(
        nu=0.5,
        kernel='rbf',
        gamma='scale',
        max_iter=1000,
        tol=1e-5,
        cache_size=256,
        multiclass="ovr"
    )
    model.fit(X_scaled, y)
    
    h = 0.02
    x_min, x_max = X_scaled[:, 0].min() - 0.5, X_scaled[:, 0].max() + 0.5
    y_min, y_max = X_scaled[:, 1].min() - 0.5, X_scaled[:, 1].max() + 0.5
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    
    ax.contourf(xx, yy, Z, alpha=0.3, cmap=plt.cm.RdYlBu)
    ax.contour(xx, yy, Z, colors='black', linewidths=1, alpha=0.5)
    
    scatter = ax.scatter(X_scaled[:, 0], X_scaled[:, 1], c=y, cmap=plt.cm.RdYlBu,
                         edgecolors='black', linewidths=1, s=50)
    
    train_score = model.score(X_scaled, y)
    ax.set_title(f'{name} (NuSVC RBF)\nAccuracy: {train_score:.3f}', fontsize=10)
    ax.set_xlabel('Feature 1')
    ax.set_ylabel('Feature 2')
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('nu_svc_rbf_classification.png', dpi=150, bbox_inches='tight')
plt.show()

print("NuSVC (RBF) visualization saved as 'nu_svc_rbf_classification.png'")

# Test NuSVC with RBF kernel and OVR strategy on multiclass datasets
print("\nTesting NuSVC (RBF, OVR) with multiclass datasets...")

fig8, axes8 = plt.subplots(2, 2, figsize=(12, 10))
axes8 = axes8.flatten()

for idx, (name, (X, y)) in enumerate(multiclass_datasets):
    ax = axes8[idx]
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    X_vis = X_scaled[:, :2] if X_scaled.shape[1] > 2 else X_scaled
    
    model = NuSVC(
        nu=0.5,
        kernel='rbf',
        gamma='scale',
        max_iter=1000,
        tol=1e-5,
        cache_size=256,
        multiclass="ovr"
    )
    model.fit(X_scaled, y)
    
    h = 0.02
    x_min, x_max = X_vis[:, 0].min() - 0.5, X_vis[:, 0].max() + 0.5
    y_min, y_max = X_vis[:, 1].min() - 0.5, X_vis[:, 1].max() + 0.5
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    
    if X_scaled.shape[1] > 2:
        mesh_points = np.zeros((xx.ravel().shape[0], X_scaled.shape[1]))
        mesh_points[:, :2] = np.c_[xx.ravel(), yy.ravel()]
        Z = model.predict(mesh_points)
    else:
        Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    
    ax.contourf(xx, yy, Z, alpha=0.3, cmap=plt.cm.RdYlBu)
    ax.contour(xx, yy, Z, colors='black', linewidths=1, alpha=0.5)
    
    scatter = ax.scatter(X_vis[:, 0], X_vis[:, 1], c=y, cmap=plt.cm.RdYlBu,
                         edgecolors='black', linewidths=1, s=50)
    
    train_score = model.score(X_scaled, y)
    ax.set_title(f'{name} (NuSVC RBF, OVR)\nAccuracy: {train_score:.3f}', fontsize=10)
    ax.set_xlabel('Feature 1')
    ax.set_ylabel('Feature 2')
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('nu_svc_rbf_ovr_classification.png', dpi=150, bbox_inches='tight')
plt.show()

print("NuSVC (RBF, OVR) visualization saved as 'nu_svc_rbf_ovr_classification.png'")

# Test NuSVC with RBF kernel and OVO strategy on multiclass datasets
print("\nTesting NuSVC (RBF, OVO) with multiclass datasets...")

fig9, axes9 = plt.subplots(2, 2, figsize=(12, 10))
axes9 = axes9.flatten()

for idx, (name, (X, y)) in enumerate(multiclass_datasets):
    ax = axes9[idx]
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    X_vis = X_scaled[:, :2] if X_scaled.shape[1] > 2 else X_scaled
    
    model = NuSVC(
        nu=0.5,
        kernel='rbf',
        gamma='scale',
        max_iter=1000,
        tol=1e-5,
        cache_size=256,
        multiclass="ovo"
    )
    model.fit(X_scaled, y)
    
    h = 0.02
    x_min, x_max = X_vis[:, 0].min() - 0.5, X_vis[:, 0].max() + 0.5
    y_min, y_max = X_vis[:, 1].min() - 0.5, X_vis[:, 1].max() + 0.5
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    
    if X_scaled.shape[1] > 2:
        mesh_points = np.zeros((xx.ravel().shape[0], X_scaled.shape[1]))
        mesh_points[:, :2] = np.c_[xx.ravel(), yy.ravel()]
        Z = model.predict(mesh_points)
    else:
        Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    
    ax.contourf(xx, yy, Z, alpha=0.3, cmap=plt.cm.RdYlBu)
    ax.contour(xx, yy, Z, colors='black', linewidths=1, alpha=0.5)
    
    scatter = ax.scatter(X_vis[:, 0], X_vis[:, 1], c=y, cmap=plt.cm.RdYlBu,
                         edgecolors='black', linewidths=1, s=50)
    
    train_score = model.score(X_scaled, y)
    ax.set_title(f'{name} (NuSVC RBF, OVO)\nAccuracy: {train_score:.3f}', fontsize=10)
    ax.set_xlabel('Feature 1')
    ax.set_ylabel('Feature 2')
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('nu_svc_rbf_ovo_classification.png', dpi=150, bbox_inches='tight')
plt.show()

print("NuSVC (RBF, OVO) visualization saved as 'nu_svc_rbf_ovo_classification.png'")
