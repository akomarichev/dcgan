import imageio
images = []
for i in range(1, 25):
    images.append(imageio.imread('gen_images/after_epoch_'+str(i)+'.jpg'))
imageio.mimsave('mnist_generated.gif', images)
