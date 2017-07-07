local utils = {}

function utils.weightINIT(model)
  for k,v in pairs(model:findModules('Convolution')) do
     v.weight:normal(0.0, 0.02)
     v.noBias()
  end

  for k,v in pairs(model:findModules('nn.SpatialBatchNormalization')) do
     v.weight:normal(1.0, 0.02)
     v.bias:zero()
  end
end

return utils
