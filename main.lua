-- (c) Artem Komarichev, 2017
local mnist = require 'mnist'
local optim = require 'optim'
local image = require 'image'
local gnuplot = require 'gnuplot'

paths.rmall('gen_images', 'yes')  -- remove existing directory with all files
paths.mkdir('gen_images')         -- create a new one

local cmd = torch.CmdLine()
cmd:text()
cmd:text('The implementation of DCGAN in Torch-7. Run the model on MNIST dataset.')
cmd:text('List of parameters: ')
-----------------------------------
cmd:option('-cuda',       'true',       'true -> run model on GPU; false - run model on CPU')
cmd:option('-batchSize',  100,          '100 by default')
cmd:option('-cudnn',      'fastest',    'options: fastest | deterministic')
cmd:option('-manualSeed', 1024,         'set manual seed')
cmd:option('-optimizer',  'adam',       'options: adam | sgd')
cmd:option('-LR',         0.0002,       'learning rate')
cmd:option('-momentum',   0.5,          'beta1 for adam algorithm')
cmd:option('-epochs',     25,           'number of epochs')
cmd:option('-imgSize',    28,           'image size')
cmd:option('-channels',   1,            'number of channels')
cmd:option('-ndf',        64,           'number of filters in the discriminator')
cmd:option('-ngf',        64,           'number of filters in the generator')
cmd:option('-nz',         100,          'dimension of the noise')
cmd:option('-noise',      'normal',     'uniform or normal')
------------------------------------
cmd:text()

local opt = cmd:parse(arg or {})

torch.manualSeed(opt.manualSeed)
torch.setdefaulttensortype('torch.FloatTensor')

-- Load MNIST data
-- Normalize data between -1 and 1
local trainData = mnist.traindataset().data:float():add(-127.5):div(127.5):view(-1, 1, opt.imgSize, opt.imgSize)
local testData = mnist.testdataset().data:float():add(-127.5):div(127.5):view(-1, 1, opt.imgSize, opt.imgSize)
local Ntrain = trainData:size(1)
local Ntest = testData:size(1)

-- Create models
local netG = dofile('models/netG.lua')(opt)
local netD = dofile('models/netD.lua')(opt)

-- Create criterion
local criterionBCE = nn.BCECriterion()

-- Create tensors
local input = torch.Tensor(opt.batchSize, opt.channels, opt.imgSize, opt.imgSize)
local noise = torch.Tensor(opt.batchSize, opt.nz, 1, 1)
local target = torch.Tensor(opt.batchSize)

local noise_test = torch.Tensor(opt.batchSize, opt.nz, 1, 1)
if opt.noise == 'uniform' then
    noise_test:uniform(-1, 1)
elseif opt.noise == 'normal' then
    noise_test:normal(0, 1)
end

-- Run the model on the GPU
if opt.cuda then
  require 'cudnn'
  require 'cunn'
  trainData:cuda()
  testData:cuda()
  ------------------
  criterionBCE:cuda()
  ------------------
  input = input:cuda()
  noise = noise:cuda()
  target = target:cuda()
  noise_test = noise_test:cuda()
  ------------------
  cudnn.convert(netD, cudnn):cuda()
  cudnn.convert(netG, cudnn):cuda()
  cudnn.benchmark = true
end

optimStateG = {
  learningRate = opt.LR,
  beta1 = opt.momentum,
}

optimStateD = {
  learningRate = opt.LR,
  beta1 = opt.momentum,
}

local parametersG, gradParametersG = netG:getParameters()
local parametersD, gradParametersD = netD:getParameters()

loss_D_real, loss_D_fake, loss_G = 0, 0, 0
local sum_loss_D_real, sum_loss_D_fake, sum_loss_G = 0, 0, 0
local losses_D_real, losses_D_fake, losses_G = {}, {}, {}

local function fDx()
  gradParametersD:zero()

  input:copy(batch)
  -- make real targets
  target:fill(1)

  netD:forward(input)
  loss_D_real = criterionBCE:forward(netD.output, target)
  criterionBCE:backward(netD.output, target)
  netD:backward(input, criterionBCE.gradInput)

  -- create noise
  if opt.noise == 'uniform' then -- regenerate random noise
    noise:uniform(-1, 1)
  elseif opt.noise == 'normal' then
    noise:normal(0, 1)
  end

  -- maek fake targets
  target:zero()
  -- feed noise to the generator
  local output_netG = netG:forward(noise)

  netD:forward(output_netG)
  loss_D_fake = criterionBCE:forward(netD.output, target)
  criterionBCE:backward(netD.output, target)
  netD:backward(output_netG, criterionBCE.gradInput)

  loss = loss_D_real + loss_D_fake

  return loss, gradParametersD
end

local function fGx()
  gradParametersG:zero()

  -- make real targets
  target:fill(1)

  loss_G = criterionBCE:forward(netD.output, target)
  criterionBCE:backward(netD.output, target)
  netD:updateGradInput(input, criterionBCE.gradInput)
  netG:backward(noise, netD.gradInput)

  return loss_G, gradParametersG
end

local function plotLoss(loss, description)
  local plots = {{description..' loss', torch.linspace(1, #loss, #loss), torch.Tensor(loss), '-'}}
  gnuplot.pngfigure('gen_images/loss_'.. description ..'.png')
  gnuplot.plot(table.unpack(plots))
  gnuplot.ylabel('Loss')
  gnuplot.xlabel('Epoch #')
  gnuplot.axis({0,opt.epochs,0,3})
  gnuplot.plotflush()
  gnuplot.close()
end

batch = nil

for epoch = 1, opt.epochs do
  for n = 1, Ntrain, opt.batchSize do
    batch = trainData:narrow(1, n, opt.batchSize)

    optim[opt.optimizer](fDx, parametersD, optimStateD)
    optim[opt.optimizer](fGx, parametersG, optimStateG)

    sum_loss_D_real = sum_loss_D_real + loss_D_real
    sum_loss_D_fake = sum_loss_D_fake + loss_D_fake
    sum_loss_G = sum_loss_G + loss_G
  end

  local trainSize = Ntrain / opt.batchSize
  losses_D_real[#losses_D_real + 1] = sum_loss_D_real / trainSize
  losses_D_fake[#losses_D_fake + 1] = sum_loss_D_fake / trainSize
  losses_G[#losses_G + 1] = sum_loss_G / trainSize

  sum_loss_D_real, sum_loss_D_fake, sum_loss_G = 0, 0, 0

  print('# Epoch number ' .. epoch .. ' with the total ' .. opt.epochs .. ' epochs.')

  local output_netG = netG:forward(noise_test)

  dd = image.toDisplayTensor{input=output_netG:float(),
    padding=2,
    nrow=math.floor(math.sqrt(opt.batchSize)),
    symmetric=false, scaleeach=true, saturate=false}

  image.save('gen_images/after_epoch_'.. epoch ..'.jpg', dd)

  plotLoss(losses_D_real, 'real')
  plotLoss(losses_D_fake, 'fake')
  plotLoss(losses_G, 'gen')

  parametersD, gradParametersD = nil, nil
  parametersG, gradParametersG = nil, nil
  parametersD, gradParametersD = netD:getParameters()
  parametersG, gradParametersG = netG:getParameters()

end
