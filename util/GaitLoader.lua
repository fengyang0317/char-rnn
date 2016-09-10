require 'mattorch'

local GaitLoader = {}
GaitLoader.__index = GaitLoader

function GaitLoader.create(data_dir, src_angle, dst_angle, split_fractions)
    -- split_fractions is e.g. {0.9, 0.1, 0.1}

    local self = {}
    setmetatable(self, GaitLoader)
    
    self.total = 124
    self.src_angle = src_angle
    self.dst_angle = dst_angle
    self.data_dir = data_dir

    -- perform safety checks on split_fractions
    assert(split_fractions[1] >= 0 and split_fractions[1] <= 1, 'bad split fraction ' .. split_fractions[1] .. ' for train, not between 0 and 1')
    assert(split_fractions[2] >= 0 and split_fractions[2] <= 1, 'bad split fraction ' .. split_fractions[2] .. ' for val, not between 0 and 1')
    assert(split_fractions[3] >= 0 and split_fractions[3] <= 1, 'bad split fraction ' .. split_fractions[3] .. ' for test, not between 0 and 1')
    if split_fractions[3] == 0 then 
        -- catch a common special case where the user might not want a test set
        self.ntrain = math.floor(self.total * split_fractions[1])
        self.nval = self.total - self.ntrain
        self.ntest = 0
    else
        -- divide data to train/val and allocate rest to test
        self.ntrain = math.floor(self.total * split_fractions[1])
        self.nval = math.floor(self.total * split_fractions[2])
        self.ntest = self.total - self.nval - self.ntrain -- the rest goes to test (to ensure this adds up exactly)
    end

    self.split_sizes = {self.ntrain, self.nval, self.ntest}
    self.batch_ix = {0,0,0}

    collectgarbage()
    return self
end

function GaitLoader:reset_batch_pointer(split_index, batch_index)
    batch_index = batch_index or 0
    self.batch_ix[split_index] = batch_index
end

function GaitLoader:get_feat_size()
  loaded = mattorch.load(self.data_dir .. '076-bg-02-018-048.mat')
  return loaded['features']:storage():size()
end

function GaitLoader:next_batch(split_index)
    if self.split_sizes[split_index] == 0 then
        -- perform a check here to make sure the user isn't screwing something up
        local split_names = {'train', 'val', 'test'}
        print('ERROR. Code requested a batch for split ' .. split_names[split_index] .. ', but this split has no data.')
        os.exit() -- crash violently
    end
    -- split_index is integer: 1 = train, 2 = val, 3 = test
    self.batch_ix[split_index] = self.batch_ix[split_index] + 1
    if self.batch_ix[split_index] > self.split_sizes[split_index] then
        self.batch_ix[split_index] = 1 -- cycle around to beginning
    end
    -- pull out the correct next batch
    local ix = self.batch_ix[split_index]
    if split_index == 2 then ix = ix + self.ntrain end -- offset by train set size
    if split_index == 3 then ix = ix + self.ntrain + self.nval end -- offset by train + val
    
    local nm_id = torch.random(6)
    local cmd = io.popen(string.format('ls %s%03d-nm-%02d-%s-*', self.data_dir, ix, nm_id, self.src_angle))
    local ans = cmd:read('*all')
    local fsrc = string.split(ans, '\n')
    
    cmd = io.popen(string.format('ls %s%03d-nm-%02d-%s-*', self.data_dir, ix, nm_id, self.dst_angle))
    ans = cmd:read('*all')
    local fdst = string.split(ans, '\n')
    
    local ldir = #self.data_dir
    local st = math.max(tonumber(fsrc[1]:sub(ldir + 15, ldir + 17)), tonumber(fdst[1]:sub(ldir + 15, ldir + 17)))
    local en = math.min(tonumber(fsrc[#fsrc]:sub(ldir + 15, ldir + 17)), tonumber(fdst[#fdst]:sub(ldir + 15, ldir + 17)))
    
    local x = {}
    local y = {}
    
    for i = st,en do
      local loaded = mattorch.load(string.format('%s%03d-nm-%02d-%s-%03d.mat', self.data_dir, ix, nm_id, self.src_angle, i))
      x[#x] = loaded['features']
      loaded = mattorch.load(string.format('%s%03d-nm-%02d-%s-%03d.mat', self.data_dir, ix, nm_id, self.dst_angle, i))
      y[#y] = loaded['features']
    end
    
    return x, y
end

return GaitLoader

