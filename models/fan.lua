-- Face Alignment Network
--
-- How far are we from solving the 2D \& 3D Face Alignment problem? (and a dataset of 230,000 3D facial landmarks)
-- Adrian Bulat and Georgios Tzimiropoulos
-- ICCV 2017
--

local cudnn = require 'cudnn'

-- Define some short names
local conv = cudnn.SpatialConvolution
local batchnorm = nn.SpatialBatchNormalization
local relu = cudnn.ReLU
local upsample = nn.SpatialUpSamplingNearest

-- Opts
local nModules = 1
local nFeats = 256
local nStack = 8


local function convBlock(numIn, numOut, order)
    local cnet = nn.Sequential()
        :add(batchnorm(numIn,1e-5,false))
        :add(relu(true))
        :add(conv(numIn,numOut/2,3,3,1,1,1,1):noBias())
        :add(nn.ConcatTable()
            :add(nn.Identity())
            :add(nn.Sequential()
                :add(nn.Sequential()
                    :add(batchnorm(numOut/2,1e-5,false))
                    :add(relu(true))
                    :add(conv(numOut/2,numOut/4,3,3,1,1,1,1):noBias())
                )
                :add(nn.ConcatTable()
                    :add(nn.Identity())
                    :add(nn.Sequential()
                        :add(batchnorm(numOut/4,1e-5,false))
                        :add(relu(true))
                        :add(conv(numOut/4,numOut/4,3,3,1,1,1,1):noBias())
                    )
                )
                :add(nn.JoinTable(2))
            )
        )
        :add(nn.JoinTable(2))
    return cnet
end

-- Skip layer
local function skipLayer(numIn,numOut)
    if numIn == numOut  then
        return nn.Identity()
    else
        return nn.Sequential()
            :add(batchnorm(numIn,1e-5,false))
            :add(relu(true))
            :add(conv(numIn,numOut,1,1):noBias())
    end
end

-- Residual block
local function Residual(numIn,numOut)
    return nn.Sequential()
        :add(nn.ConcatTable()
            :add(convBlock(numIn,numOut))
            :add(skipLayer(numIn,numOut)))
        :add(nn.CAddTable(true))
end



function  (opt)
    nModules = opt.nModules
    nFeats = opt.nFeats
    nStack = opt.nStacks

    local model = nn.Sequential()

    -- Initial processing of the image
    model:add(conv(3,64,7,7,2,2,3,3))         -- 128
    model:add(relu())
    model:add(batchnorm(64))
    model:add(Residual(64,128))
    model:add(nn.SpatialMaxPooling(2,2,2,2))                      -- 64
    model:add(Residual(128,128))
    model:add(Residual(128,nFeats))
    
    model:add(Residual(nFeats,nFeats))
    model:add(Residual(nFeats,nFeats))
    model:add(Residual(nFeats,nFeats))
    model:add(batchnorm(numIn,1e-5,false))
    model:add(relu(true))
    model:add(conv(nFeats,nFeats, 1,1))

    model:add(Residual(nFeats,nFeats))
    model:add(Residual(nFeats,nFeats))
    model:add(Residual(nFeats,nFeats))
    model:add(batchnorm(numIn,1e-5,false))
    model:add(relu(true))
    model:add(conv(nFeats,nFeats, 1,1))

    model:add(Residual(nFeats,nFeats))
    model:add(Residual(nFeats,nFeats))
    model:add(Residual(nFeats,nFeats))
    model:add(batchnorm(numIn,1e-5,false))
    model:add(relu(true))
    model:add(conv(nFeats,68, 1,1))


    

    return model

end

return createModel



