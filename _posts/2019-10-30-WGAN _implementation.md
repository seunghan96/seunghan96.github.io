---
title: WGAN (Implementation with Python)
categories: [DL,GAN]
tags: [Deep Learning, WGAN]
excerpt: WGAN
---

(위키북스의 '케라스로 구현하는 고급 딥러닝 알고리즘'을 참고하여 작성하였습니다)

# 3. WGAN 구현하기

- DCGAN과의 차이점 : Loss Function을 **베서슈타인 손실 함수**로 사용한다는 점
- 그 외의 원리는 전부 동일하다

## (1) Generator
- input : noise vector와, 만들어 낼 image의 크기
- output : (fake) image를 만드는 generator

layer의 구성 :
- 1) Batch Normalization
- 2) Activation Function ( ReLU & Sigmoid )
- 3) Conv2DTranspose ( Deconvolution을 해주는 layer )


```python
def Generator(inputs, img_size):
    # (1) image크기 조정
    img_resize = img_size//4
    
    # (2) parameter
    kernel_s = 5
    filters = [64,64,32,1]
    
    # (3) input
    x = Dense(img_resize * img_resize * filters[0])(inputs)
    x = Reshape((img_resize,img_resize,filters[0]))(x)
    
    for f in filters:
        if f > filters[-2]: # first 2 layers : stride=2, last 2 layers : stride=1
            stride = 2
        else :
            stride =1
        x = BatchNormalization()(x) # (1) BN
        x = Activation('relu')(x) # (2) ReLU
        x = Conv2DTranspose(filters=f,kernel_size=kernel_s,strides=stride,padding='same')(x) # (3) Deconvolution
    
    x = Acivation('Sigmoid')(x) # (4) Sigmoid
    G = Model(inputs,x,name='generator')
    
    return G
```

## (2) Discriminator
- input : image
- output : 0~1 사이의 값 (0:fake ~ 1:real)


```python
def Discriminator(inputs):
    
    # (1) parameter
    kernel_s = 5
    filters = [32,64,128,256]
    
    x = inputs
    
    for f in filters:
        if f == filters[-1]: # first 3 filters : stride=2, last layer : stride=1
            stride = 1
        else :
            stride = 2
        x = ReLU(x) # (1) ReLU
        x = Conv2D(filters=f, kernel_size=kernel_s, strides=stride,padding='same')(x) # (2) Convolutional Layer
    
    x = Flatten()(x) # (3) Flatten
    x = Dense(1)(x) # (4) Dense
    x = Activation('sigmoid')(x) # (5) Sogmoid
    D = Model(inputs,x,name='discriminator')
    
    return D    
```

## (3) Implement GAN with Generator & Discriminator


```python
def build_GAN():
    (x_train,_),(_,_) = mnist.load_data() # only need image ( no label )
    
    # 1. Reshape Image
    img_size = x_train.shape[1] # (= 28)
    x_train = np.reshape(x_train, [-1,img_size,img_size,1]) # into 28,28,1
    x_train = x_train.astype('float32')/255
    
    # 2. parameter
    model_name = 'WGAN'
    dim = 100 
    n_critic = 5 # Discriminator가 5회 훈련되는 동안, Generator는 1회 훈련
    clip_val = 0.01 # Discriminator의 weight를 제한
    batch_size = 64 
    train_steps = 10000
    lr = 2e-4    
    input_shape = (img_size,img_size,1)
    
    # 3-1. Discriminator
    inputs = Input(shape=input_shape, name='D_input')
    D = Discriminator(inputs, activation='linear')
    D.compile(loss=wasserstein_loss, optimizer=RMSprop(lr=lr),metrics=['accuracy']) # 베서슈타인 손실 함수 사용!
    D.summary()
    
    # 3-2. Generator
    inputs2 = Input(shape=(dim,), name='G_input')
    G = Generator(inputs2, img_size)
    G.summary()
    
    # 4. Adversarial Update
    D.trainable = False    
    GAN = Model(inputs2, D(G(inputs2)),name=model_name)
    GAN.compile(loss=wasserstein_loss,optimizer=RMSprop(lr=lr))
    GAN.summary()
    
    # 5. Training
    models = (G,D,GAN)
    params = (dim,batch_size,n_critic,clip_val,train_steps,model_name)
    train(models,x_train,params)
```

## (4) Train


```python
def WGAN_train(models,x_train,params):
    
    G,D,GAN = models
    (dim,batch_size,n_critic,clip_val,train_steps,model_name) = params
    save_point = 500
    
    # sample data ( 확인용 )
    noise_vec = np.random.uniform(-1,1,size=[16,dim])
    train_size = x_train.shape[0]
    label_real = np.nes((batch_size,1))
    
    ####################[  Discriminator  ]###################
    
    # 1번의 train_step
    for i in range(train_steps):
        
        # n_critic번의 Discriminator 훈련        
        loss = 0
        acc = 0
        
        for _ in range(n_critic):
            
            # 1) real image
            randoms = np.random.randint(0,train_size,size=batch_size) # select random images
            image_real = x_train[randoms]
            
            # 2) fake image
            noise = np.random.uniform(-1,1,size=[batch_size,latent_size]) 
            image_fake = G.predict(noise)
        
            # 기존의 방법 : real & fake image를 결합했었음
            # 새로운 방법 : real로 이루어진 batch & fake로 이루어진 batch -> 교대로 훈련
            # x = np.concatenate((image_real,image_fake))
            # y = np.ones([2*batch_size,1])
            # y[batch_size:,:] = 0
            
            # 3) train
            real_loss, real_acc = D.train_on_batch(image_real,label_real) # 진짜 image 학습
            fake_loss, fake_acc = D.train_on_batch(image_fake,label_real) # 가짜 image 학습
            
            loss += 0.5*(real_loss + fake_loss)
            acc += 0.5*(real_acc + fake_acc)
            
            for layer in D.layers:
                W = layer.get_weights()
                W = [np.clip(w, -clip_val, clip_val) for w in W]
                layer.set_weights(W)
            
            loss /= n_critic
            acc /= n_critic
            temp = "%d : [Discriminator loss : %f, accuracy : $f]" % (i,loss,acc)
       
    
    ####################[  Generator  ]###################
    
    # 1) train
    loss, acc = GAN.train_on_batch(noise_vec,label_real)
    temp = "%d : [Generator loss : %4, accuracy : $f]" % (i,loss,acc)
    print(temp)
    
    # 2) show results
    if (i+1) % save_point ==0:
        if (i+1) == train_steps:
            show = True
        else :
            show = False            
            plot_images(G,noise_input = noise_vec,show=show,
                       step=(i+1),model_name=model_name)
            
    G.save(model_name + ".h5")
```


```python

```
