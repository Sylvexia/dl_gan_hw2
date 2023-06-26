# 深度學習 作業二

## Abstract:
This project implement basic dcgan and wgan to generate human face.

## Author

洪祐鈞 a1085125

## Model link

[Model link](https://drive.google.com/drive/folders/1IUb9ye0qXdEQ16CNAZ38MDmwFbpMIrau?usp=sharing)

## How to run?

The data is not included in the repo.

run
```
python dcgan.py
python wgan.py --path data/img_align_celeba
```

For dcgan, the data path relative path should be something like ```data/img_align_celeba/```, since its staight copy of the teacher's example. (Also almost the same as pytorch documentation), the thing I add is just saving the model.

For wgan that I've implement, you can and should add the path that point to the directory of the data.
I've implemented tracking loss to tensorboard to track during the training realtime.

run
```
tensorboard --logdir logs
```
to see loss and generated per steps.

## GAN model experiment and result

Basically all model are run with about 15000 iteration, the image size is all ```64*64``` for fair comparison. Also my poor computer cannot run ```128*128``` size image since it would overflow the gpu memory.

### DCGAN

Here, the model is basically straight copy from what teacher gave us. The change I've made doesn't seemed to make any good difference. The biggest differnce happened when I modify the learning rate, or drastically increase the batch size to say, 720. The model performance significantly dropped.

#### Training

![picture 2](images/b8dd72eedea8ec2ad18c3823b16df8f1cec176ca0ba9c32f8d5c430ceb38f352.png)  

The training loss basically goes everywhere, and hard to tell if the model is performed well or not.

#### Animation:

![picture 1](images/b1a1f4e06d7f3cba83c7d265356c60a9d285845c95d96248939f18af1ddc4b70.gif)  

#### Comparison:

![picture 3](images/e0162800d8de1c79759d7da1f93581f26fc0301b4e2bee3e6cc250550a36ce13.png)  

### WGAN

Well, from what I've know, we want the data and generator probability distribution to be close as much as possible. The original GAN use Jensne-Shannon divergence as loss function but it was treated like binary classifcation, if the distribution is not overlap, no matter how close the distrubution is, the JS divergence would still be the same. So WGAN introduce Wasserstein distance to resolve such issue. The idea of the Wasserstein distance is to calculate what kind of the moving plan as a metric such that the generator and data distribution be the smallest. So it can actually extract more information than JS divergence even if the distribution is not overlapped. It just like a earth mover, contantly push the pile of the earth closer together.

In the paper, they changed the original Discriminator to critic, when training critic, we need to train the critic more times than the generator. At the end of the critic iteration loop, we need to do weight clipping to limit the weight 

Also WGAN take away the sigmoid layer from critic, so the critic output would not between 0 and 1.

The code implement gradient penalty as the original paper proposed weight clipping, the idea is basically sample true and fake data, the interpolation of the 2 sampled data between them should be close to 1. Also the batch normalization is changed to instance normalization layer, which would not do the normalization accross the batch.

Acually the word above is just what I've heard from [NTU-Hung-yi Lee WGAN theory intruduction](https://www.youtube.com/watch?v=jNY1WBb8l4U) and [WGAN implementation from scratch (with gradient penalty](https://www.youtube.com/watch?v=pG0QZ7OddX4), I still have no idea what was I saying. I suck at mathematics in general.

The implementation in [WGAN implementation from scratch (with gradient penalty](https://www.youtube.com/watch?v=pG0QZ7OddX4) had a error that might be fixed by me is that, we should sample new data every batch during the critics training. But the result I got doesn't matter that much.

The model architecture is the same otherwise as the dcgan.

#### Loss:

![picture 4](images/8c788014fab69bf8bbb69f69e982e11949e294a6496f4e2516bfcd44a6ce258f.png)

The red line is Generate loss, and blue line is Critics loss.

#### Comparison

real:

![picture 5](images/4267a39749aab0d4ceebb2c0ed6ac32a87bd3392c130864b99c4bcd5ffe406be.png)  

fake:

![picture 6](images/b4708904de5e9eaa06adab982f0777395654af400bb44eb04ed5ac3e9ec29a74.png)  

#### Animation

![picture 10](images/5a14c3ff7a74c772cd8fbc100f498026bee8d2d02427a9c1b085b563b0cfc0f2.gif)

### LightweightGAN

Basically, I use the [repo here](https://github.com/lucidrains/lightweight-gan). As chosen paper comparison.

#### Loss

![picture 8](images/83137585ffe6ed665e809168229f553b1021b44b48579f35802fd783b6b97baf.jpg)

The model did implement gradient penalty and the loss is gliched like everywhere, unlike WGAN, why???

#### Result

![picture 9](images/d13f589112f3880d25c51e6344912950434d6c3a3f1dbfbbcc6e1d1813fa04d1.jpg)

## Conclusion

I don't really have that much of conclusion, this experiment is a total mess.
Here's some notes:
- Large batch size seems to make gan model perform worse, different from what I've read.
- For the generating result, I think DCGAN>LightweightGAN>WGAN, most of the DCGAN faces looks like faces, the face generated from Lightweight gan have some really weird blue region spot, but still can pick some good face. For the WGAN, although the training loss looks complete normal, but most of the face is really skewed.

## Final Words

The covid almost killed me that made me even hard to came up with anything or even code anything to implement what i thought, also gan model takes like forever to train. It's hard to tell any change matters or not, my hardware resource is abysmal that a lot of GAN model I run just does not work. Somehow on kaggle, even with 2 GPU, It somehow even run slower than 3050 4gb laptop gpu (I had bottleneck using kaggle cpu), and it disconnect every 12 hrs. 

I have mass frustration doing this project. As Hung-yi Lee professor in NTU said: "No pain, no GAN".