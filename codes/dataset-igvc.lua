require 'torch'
-- require 'paths'

-- mnist = {}

-- mnist.path_remote = 'https://s3.amazonaws.com/torch7/data/mnist.t7.tgz'
-- mnist.path_dataset = 'mnist.t7'
-- mnist.path_trainset = paths.concat(mnist.path_dataset, 'train_32x32.t7')
-- mnist.path_testset = paths.concat(mnist.path_dataset, 'test_32x32.t7')

-- function mnist.download()
--    if not paths.filep(mnist.path_trainset) or not paths.filep(mnist.path_testset) then
--       local remote = mnist.path_remote
--       local tar = paths.basename(remote)
--       os.execute('wget ' .. remote .. '; ' .. 'tar xvf ' .. tar .. '; rm ' .. tar)
--    end
-- end

function loadTrainSet(maxLoad, geometry)
   return loadDataset('train', maxLoad, geometry)
end

function loadTestSet(maxLoad, geometry)
   return loadDataset('test', maxLoad, geometry)
end

function loadDataset(dataType, maxLoad)
   -- mnist.download()

   local data = torch.load('../data/patches/'..dataType..'_data.t7')
   -- local data = data.reshape(data:size(1),geometry[1],geometry[2])
   local labels = torch.load('../data/patches/'..dataType..'_label.t7')
   local labels = labels + 1

   local nExample = data:size(1)
   print('Loading ' .. nExample .. ' examples')
   if maxLoad and maxLoad > 0 and maxLoad < nExample then
      nExample = maxLoad
      print('<igvc> loading only ' .. nExample .. ' examples')
   end
   -- data = data[{{1,nExample},{},{},{}}]
   data = data[{{1,nExample},{},{}}]
   labels = labels[{{1,nExample}}]
   print('<igvc> done')

   local dataset = {}
   dataset.data = data
   dataset.labels = labels

   function dataset:normalize(mean_, std_)
      local mean = mean_ or data:view(data:size(1), -1):mean(1)
      local std = std_ or data:view(data:size(1), -1):std(1, true)
      for i=1,data:size(1) do
         data[i]:add(-mean[1][i])
         if std[1][i] > 0 then
            tensor:select(2, i):mul(1/std[1][i])
         end
      end
      return mean, std
   end

   function dataset:normalizeGlobal(mean_, std_)
      local std = std_ or data:std()
      local mean = mean_ or data:mean()
      data:add(-mean)
      data:mul(1/std)
      return mean, std
   end

   function dataset:size()
      return nExample
   end

   local labelvector = torch.zeros(2)

   setmetatable(dataset, {__index = function(self, index)
              -- print(index)
			     local input = self.data[index]
			     local class = self.labels[index]
			     local label = labelvector:zero()
              -- print('reached here')
			     label[class] = 1
              -- print('did not reach here')
			     local example = {input, label}
                                       return example
   end})

   return dataset
end