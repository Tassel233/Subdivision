#include "vkutil.hpp"

// SOLUTION_TAGS: vulkan-(ex-[^1]|cw-.)

#include <vector>

#include <cstdio>
#include <cassert>

#include "error.hpp"
#include "to_string.hpp"

namespace labutils
{
    ShaderModule load_shader_module(VulkanContext const& aContext, char const* aSpirvPath)
    {
        assert(aSpirvPath);

        if (std::FILE* fin = std::fopen(aSpirvPath, "rb"))
        {
            std::fseek(fin, 0, SEEK_END);
            auto const bytes = std::size_t(std::ftell(fin));
            std::fseek(fin, 0, SEEK_SET);

            // SPIR-V consists of a number of 32-bit = 4 byte words
            assert(0 == bytes % 4);
            auto const words = bytes / 4;

            std::vector<std::uint32_t> code(words);

            std::size_t offset = 0;
            while (offset != words)
            {
                auto const read = std::fread(code.data() + offset, sizeof(std::uint32_t), words - offset, fin);

                if (0 == read)
                {
                    auto const err = std::ferror(fin), eof = std::feof(fin);
                    std::fclose(fin);

                    throw Error("Error reading ¡®%s¡¯: ferror = %d, feof = %d", aSpirvPath, err, eof);
                }

                offset += read;
            }

            std::fclose(fin);

            VkShaderModuleCreateInfo moduleInfo{};
            moduleInfo.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
            moduleInfo.codeSize = bytes;
            moduleInfo.pCode = code.data();

            VkShaderModule smod = VK_NULL_HANDLE;
            if (auto const res = vkCreateShaderModule(aContext.device, &moduleInfo, nullptr, &smod);
                VK_SUCCESS != res)
            {
                throw Error("Unable to create shader module from %s\n"
                    "vkCreateShaderModule() returned %s", aSpirvPath, to_string(res).c_str());
            }

            return ShaderModule(aContext.device, smod);
        }

        throw Error("Cannot open ¡®%s¡¯ for reading", aSpirvPath);
    }


    CommandPool create_command_pool(VulkanContext const& aContext, VkCommandPoolCreateFlags aFlags)
    {
        VkCommandPoolCreateInfo poolInfo{};
        poolInfo.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
        poolInfo.queueFamilyIndex = aContext.graphicsFamilyIndex;
        poolInfo.flags = aFlags;

        VkCommandPool cpool = VK_NULL_HANDLE;
        if (auto const res = vkCreateCommandPool(aContext.device, &poolInfo, nullptr, &cpool); VK_SUCCESS != res)
        {
            throw Error("Unable to create command pool\n"
                "vkCreateCommandPool() returned %s", to_string(res).c_str());
        }

        return CommandPool(aContext.device, cpool);

    }

    VkCommandBuffer alloc_command_buffer(VulkanContext const& aContext, VkCommandPool aCmdPool)
    {
        VkCommandBufferAllocateInfo cbufInfo{};
        cbufInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
        cbufInfo.commandPool = aCmdPool;
        cbufInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
        cbufInfo.commandBufferCount = 1;

        VkCommandBuffer cbuff = VK_NULL_HANDLE;
        if (auto const res = vkAllocateCommandBuffers(aContext.device, &cbufInfo, &cbuff); VK_SUCCESS != res)
        {
            throw Error("Unable to allocate command buffer\n"
                "vkAllocateCommandBuffers() returned %s", to_string(res).c_str());
        }

        return cbuff;

    }


    Fence create_fence(VulkanContext const& aContext, VkFenceCreateFlags aFlags)
    {
        VkFenceCreateInfo fenceInfo{};
        fenceInfo.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
        fenceInfo.flags = aFlags;

        VkFence fence = VK_NULL_HANDLE;
        if (auto const res = vkCreateFence(aContext.device, &fenceInfo, nullptr, &fence); VK_SUCCESS != res)
        {
            throw Error("Unable to create fence\n"
                "vkCreateFence() returned %s", to_string(res).c_str());
        }

        return Fence(aContext.device, fence);

    }

    Semaphore create_semaphore(VulkanContext const& aContext)
    {
        VkSemaphoreCreateInfo semaphoreInfo{};
        semaphoreInfo.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO;

        VkSemaphore semaphore = VK_NULL_HANDLE;
        if (auto const res = vkCreateSemaphore(aContext.device, &semaphoreInfo, nullptr, &semaphore); VK_SUCCESS != res)
        {
            throw Error("Unable to create semaphore\n"
                "vkCreateSemaphore() returned %s", to_string(res).c_str());
        }

        return Semaphore(aContext.device, semaphore);
    }


    void buffer_barrier(
        VkCommandBuffer aCmdBuff, 
        VkBuffer aBuffer, 
        VkAccessFlags aSrcAccessMask,
        VkAccessFlags aDstAccessMask, 
        VkPipelineStageFlags aSrcStageMask,
        VkPipelineStageFlags aDstStageMask,
        VkDeviceSize aSize, 
        VkDeviceSize aOffset,
        uint32_t aSrcQueueFamilyIndex, 
        uint32_t aDstQueueFamilyIndex)
    {
        VkBufferMemoryBarrier bbarrier{};
        bbarrier.sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER;
        bbarrier.srcAccessMask = aSrcAccessMask;
        bbarrier.dstAccessMask = aDstAccessMask;
        bbarrier.buffer = aBuffer;
        bbarrier.size = aSize;
        bbarrier.offset = aOffset;
        bbarrier.srcQueueFamilyIndex = aSrcQueueFamilyIndex;
        bbarrier.dstQueueFamilyIndex = aDstQueueFamilyIndex;

        vkCmdPipelineBarrier(
            aCmdBuff,
            aSrcStageMask, aDstStageMask,
            0,
            0, nullptr,
            1, &bbarrier,
            0, nullptr
        );
    }


    DescriptorPool create_descriptor_pool(VulkanContext const& aContext, std::uint32_t aMaxDescriptors, std::uint32_t aMaxSets)
    {
        VkDescriptorPoolSize const pools[] = {
            { VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, aMaxDescriptors },
            { VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, aMaxDescriptors }
        };

        VkDescriptorPoolCreateInfo poolInfo{};
        poolInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
        poolInfo.maxSets = aMaxSets;
        poolInfo.poolSizeCount = sizeof(pools) / sizeof(pools[0]);
        poolInfo.pPoolSizes = pools;

        VkDescriptorPool pool = VK_NULL_HANDLE;
        if (auto const res = vkCreateDescriptorPool(aContext.device, &poolInfo, nullptr, &pool); VK_SUCCESS != res)
        {
            throw Error("Unable to create descriptor pool\n"
                "vkCreateDescriptorPool() returned %s", to_string(res).c_str());
        }

        return DescriptorPool(aContext.device, pool);
    }


    VkDescriptorSet alloc_desc_set(VulkanContext const& aContext, VkDescriptorPool aPool, VkDescriptorSetLayout aSetLayout)
    {
        VkDescriptorSetAllocateInfo allocInfo{};
        allocInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
        allocInfo.descriptorPool = aPool;
        allocInfo.descriptorSetCount = 1;
        allocInfo.pSetLayouts = &aSetLayout;

        VkDescriptorSet dset = VK_NULL_HANDLE;
        if (auto const res = vkAllocateDescriptorSets(aContext.device, &allocInfo, &dset); VK_SUCCESS != res)
        {
            throw Error("Unable to allocate descriptor set\n"
                "vkAllocateDescriptorSets() returned %s", to_string(res).c_str());
        }

        return dset;
    }

}
