import numpy as np
from tensorflow import keras
from tensorflow.keras import layers, models
from sklearn.model_selection import train_test_split
from PIL import Image
import os
import sys

def load_data():

    X = np.load('X.npy')
    Y = np.load('Y.npy')
    return X, Y


def preprocess_data(X, Y, test_size=0.2):

    # 将图像数据归一化到 [0, 1] 范围
    X = X.astype('float32') / 255.0 if X.max() > 1.0 else X.astype('float32')
    
    # 如果图像没有通道维度，添加通道维度（灰度图）
    if len(X.shape) == 3:
        X = np.expand_dims(X, axis=-1)
    
    # 处理标签：支持数值标签和 one-hot 编码两种格式
    if len(Y.shape) == 2 and Y.shape[1] > 1:
        # Y 已经是 one-hot 编码格式
        Y_cat = Y.astype('float32')
        num_classes = Y.shape[1]
        Y = np.argmax(Y, axis=1)  # 转换为数值标签用于分层抽样
    else:
        # Y 是数值标签格式
        Y = Y.astype('int32')
        num_classes = len(np.unique(Y))
        Y_cat = keras.utils.to_categorical(Y, num_classes)  # 转换为 one-hot 编码
    
    # 划分训练集和测试集，使用分层抽样保持类别平衡
    X_train, X_test, Y_train, Y_test = train_test_split(
        X, Y_cat, 
        test_size=test_size, 
        random_state=42, 
        stratify=Y
    )
    
    return X_train, X_test, Y_train, Y_test, num_classes


def build_model(input_shape, num_classes):

    model = models.Sequential([
        # 第一卷积块
        layers.Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=input_shape),
        layers.BatchNormalization(),  # 批归一化加速收敛
        layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),  # 降采样，减少参数
        layers.Dropout(0.25),  # 随机失活防止过拟合
        
        # 第二卷积块
        layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),
        
        # 第三卷积块
        layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),
        
        # 全连接层
        layers.Flatten(),  # 展平为一维向量
        layers.Dense(256, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.5),  # 更高的失活率防止过拟合
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.3),
        
        # 输出层
        layers.Dense(num_classes, activation='softmax')  # softmax 输出概率分布
    ])
    
    return model


def train_model(model, X_train, Y_train, epochs=50, batch_size=32):

    # 编译模型：优化器使用 Adam，损失函数使用交叉熵
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    # 定义回调函数
    callbacks = [
        # 早停
        keras.callbacks.EarlyStopping(
            monitor='val_accuracy', 
            patience=10, 
            restore_best_weights=True
        ),
        # 学习率衰减
        keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss', 
            factor=0.5, 
            patience=5, 
            min_lr=1e-7
        ),
        # 模型保存：只保存验证集准确率最高的模型
        keras.callbacks.ModelCheckpoint(
            'best_model.keras', 
            monitor='val_accuracy', 
            save_best_only=True
        )
    ]
    
    # 开始训练
    return model.fit(
        X_train, Y_train,
        batch_size=batch_size,
        epochs=epochs,
        validation_split=0.1,  # 使用 10% 训练数据作为验证集
        callbacks=callbacks
    )


def evaluate_model(model, X_test, Y_test):

    loss, acc = model.evaluate(X_test, Y_test, verbose=0)
    print(f"测试集准确率: {acc * 100:.2f}%")


def predict_image(image_path, model_path='best_model.keras'):

    # 检查模型文件是否存在
    if not os.path.exists(model_path):
        print(f"错误：模型文件 '{model_path}' 不存在！")
        print("请先运行 'python main.py' 训练模型。")
        return
    
    # 加载模型
    model = keras.models.load_model(model_path)
    input_shape = model.input_shape[1:]  # 获取输入形状（忽略批量维度）
    
    # 加载并预处理图片
    img = Image.open(image_path).convert('L')  # 转换为灰度图
    img = img.resize((input_shape[1], input_shape[0]))  # 调整为模型输入大小
    img_array = np.array(img).astype('float32') / 255.0  # 归一化
    
    # 添加通道维度和批量维度
    if len(input_shape) == 3:
        img_array = np.expand_dims(img_array, axis=-1)
    img_array = np.expand_dims(img_array, axis=0)
    
    # 进行预测
    pred = model.predict(img_array, verbose=0)
    digit = np.argmax(pred[0])  # 获取概率最大的类别
    confidence = pred[0][digit] * 100  # 计算置信度
    
    # 输出结果
    print(f"预测数字: {digit}, 置信度: {confidence:.2f}%")
    return digit, confidence


def main():
    """
    主函数：执行完整的模型训练流程
    """
    print("加载数据...")
    X, Y = load_data()
    
    print("预处理数据...")
    X_train, X_test, Y_train, Y_test, num_classes = preprocess_data(X, Y)
    input_shape = X_train.shape[1:]
    
    print(f"数据形状: {X_train.shape}, 类别数: {num_classes}")
    
    print("构建模型...")
    model = build_model(input_shape, num_classes)
    model.summary()  # 打印模型结构
    
    print("\n开始训练...")
    train_model(model, X_train, Y_train)
    
    print("\n评估模型...")
    evaluate_model(model, X_test, Y_test)
    print("\n模型已保存为 'best_model.keras'")


if __name__ == '__main__':
    # 判断命令行参数
    if len(sys.argv) == 2:
        # 如果提供了图片路径，执行预测
        if sys.argv[1].endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
            predict_image(sys.argv[1])
        else:
            # 未知参数，显示帮助信息
            print("未知参数！")
            print("使用方法:")
            print("  python main.py              # 训练模型")
            print("  python main.py image.jpg    # 预测单张图片")
    else:
        # 无参数，执行训练流程
        main()
