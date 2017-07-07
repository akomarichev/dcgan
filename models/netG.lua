local nn = require 'nn'
local utils = paths.dofile('utils.lua')

local SpatialFullConvolution = nn.SpatialFullConvolution
local ReLU = nn.ReLU
local SBatchNorm = nn.SpatialBatchNormalization
local Tanh = nn.Tanh

local function createModel(opt)
  local model = nn.Sequential()

  model:add(SpatialFullConvolution(opt.nz, opt.ngf * 2, 7, 7))
  model:add(SBatchNorm(opt.ngf * 2)):add(ReLU(true))
  -- output size: 128 x 7 x 7
  model:add(SpatialFullConvolution(opt.ngf * 2, opt.ngf, 4, 4, 2, 2, 1, 1))
  model:add(SBatchNorm(opt.ngf)):add(ReLU(true))
  -- output size: 64 x 14 x 14
  model:add(SpatialFullConvolution(opt.ngf, opt.channels, 4, 4, 2, 2, 1, 1))
  model:add(Tanh())
  -- output size: 1 x 28 x 28

  utils.weightINIT(model)

  return model
end

return createModel
