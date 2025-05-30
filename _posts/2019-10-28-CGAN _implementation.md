---
title: Implementing CGAN with TF2
categories: [DL,GAN]
tags: [Deep Learning, DCGAN]
excerpt: CGAN
---

(위키북스의 '케라스로 구현하는 고급 딥러닝 알고리즘'을 참고하여 작성하였습니다)

# 2. CGAN 구현하기

- DCGAN과의 차이점 : y label값을 condition으로 주어진다는 점!
- 그 외의 원리는 전부 동일하다

## (1) Generator
- input : noise vector와, 생성하고싶은 image의 label값과, 만들어 낼 image의 크기
- output : (fake) image를 만드는 generator

layer의 구성 :
- 1) Batch Normalization
- 2) Activation Function ( ReLU & Sigmoid )
- 3) Conv2DTranspose ( Deconvolution을 해주는 layer )


```python
def Generator(inputs,y_labels img_size): # CGAN이므로, y label값도 같이 주어진다
    # image크기를 조정한다
    img_resize = img_size//4
    
    # parameter
    kernel_s = 5
    filters = [64,64,32,1]
    
    # input
    x = concatenate([inputs,y_labels],axis=1)
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
    
    x = Acivation('Sigmoid')(x)
    G = Model([inputs,y_labels],x,name='generator')
    
    return G
```



## (2) Discriminator

- input : image
- output : 0~1 사이의 값 (0:fake ~ 1:real)


```python
def Discriminator(inputs): # Discriminator에도 마찬가지로 y label을 input으로 주어야 한다.
    
    # parameter
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
    x = Activation('sigmoid')(x) # (5) 
    D = Model([inputs,y_labels],x,name='discriminator')
    
    return D    
```



## (3) Implement GAN with Generator & Discriminator

- DCGAN과 동일


```python
def build_GAN():
    (x_train,_),(_,_) = mnist.load_data() # only need image ( no label )
    
    # 1. Reshape Image
    img_size = x_train.shape[1] # (= 28)
    x_train = np.reshape(x_train, [-1,img_size,img_size,1]) # into 28,28,1
    x_train = x_train.astype('float32')/255
    
    # 2. parameter
    model_name = 'CGAN'
    dim = 100 # dimension of latent vector
    batch_size = 64 
    train_steps = 10000
    lr = 2e-4
    decay = 6e-8
    input_shape = (img_size,img_size,1)
    
    # 3-1. Discriminator
    inputs = Input(shape=input_shape, name='D_input')
    D = Discriminator(inputs)
    D.complie(loss='binary_crossentropy', optimizer=RMSprop(lr=lr,decay=decay),metrics=['accuracy'])
    D.summary()
    
    # 3-2. Generator
    inputs2 = Input(shape=(dim,), name='G_input')
    G = Generator(inputs2, img_size)
    G.summary()
    
    # 4. Adversarial Update
    D.trainable = False    
    GAN = Model(inputs2, D(G(inputs2)),name=model_name)
    GAN.compile(loss='binary_crossentropy',optimizer=RMSprop(lr=lr*0.5, decay=decay*0.5))
    GAN.summary()
    
    # 5. Training
    models = (G,D,GAN)
    params = (dim,batch_size,train_steps,model_name)
    train(models,x_train,params)
```



## (4) Train


```python
def CGAN_train(models,data,params):
    
    G,D,GAN = models
    x_train, y_train = data
    dim,batch_size,train_steps,num_labels,model_name = params
    save_point = 500
    
    noise_vec = np.random.uniform(-1,1,size=[16,dim])
    noise_class = np.eye(num_labels)[np.arange(0,16) % num_labels] # DCGAN과 다른 점
    train_size = x_train.shape[0]
    
    for i in range(train_steps):
        
        ####################[  Discriminator  ]###################
        # 1) real image
        randoms = np.random.randint(0,train_size,size=batch_size) # select random images
        image_real = x_train[randoms] 
        image_real_label = y_train[randoms] # DCGAN과 다른 점
        
        # 2) fake image
        noise = np.random.uniform(-1,1,size=[batch_size,latent_size]) 
        image_fake_label = np.eye(num_labels)[np.random.choice(num_labels,batch_size)]  # DCGAN과 다른 점
        image_fake = G.predict([noise,image_fake_label])
        
        
        # 3) concatenate images
        x = np.concatenate((image_real,image_fake))
        y = np.ones([2*batch_size,1])
        y[batch_size:,:] = 0
        
        # 4) train
        loss, acc = D.train_on_batch(x,y)
        temp = "%d : [Discriminator Loss : %f, acc : %f]" % (i,loss,acc)
        
        ####################[  Generator  ]###################
        noise = np.random.uniform(-1,1,size=[batch_size,latent_size]) 
        image_fake_label = np.eye(num_labels)[np.random.choice(num_labels,batch_size)]  # DCGAN과 다른 점
        y = np.ones([batch_size,1])
        
        loss, acc = GAN.train_on_batch([noise,image_fake_label],y)
        record = "%d : [Generator Loss : %f, acc : %f]" % (temp,loss,acc)
        print(record)
        
        if (i+1) % save_point ==0:
            if (i+1) == train_steps:
                show = True
            else :
                show = False
            
            plot_images(G,noise_input = noise_vec,noise_class=noise_class,
                        show=show,step=(i+1),model_name=model_name)
            
    G.save(model_name + ".h5")
```


```python

```
