local nn = require 'nn'
local utils = paths.dofile('utils.lua')

local SpatialConvolution = nn.SpatialConvolution
local SBatchNorm = nn.SpatialBatchNormalization

local function createModel(opt)

  local model = nn.Sequential()

  -- input size: 1 x 28 x 28
  model:add(SpatialConvolution(opt.channels, opt.ndf, 4, 4, 2, 2, 1, 1))
  model:add(nn.LeakyReLU(0.2, true))
  -- input size: 64 x 14 x 14
  model:add(SpatialConvolution(opt.ndf, opt.ndf * 2, 4, 4, 2, 2, 1, 1))
  model:add(SBatchNorm(opt.ndf * 2)):add(nn.LeakyReLU(0.2, true))
  -- input size: 128 x 7 x 7
  model:add(SpatialConvolution(opt.ndf * 2, 1, 7, 7))
  model:add(nn.Sigmoid())
  -- input size: 1 x 1 x 1
  model:add(nn.View(1):setNumInputDims(3))

  utils.weightINIT(model)

  return model
end

return createModel
