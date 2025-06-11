#ifndef VKUTIL_HPP_9DE3C6CC_921D_46FD_8452_A7F18E276E2A
#define VKUTIL_HPP_9DE3C6CC_921D_46FD_8452_A7F18E276E2A
// SOLUTION_TAGS: vulkan-(ex-[^1]|cw-.)

#include <volk/volk.h>

#include "vkobject.hpp"
#include "vulkan_context.hpp"


namespace labutils
{
	ShaderModule load_shader_module( VulkanContext const&, char const* aSpirvPath );

	CommandPool create_command_pool( VulkanContext const&, VkCommandPoolCreateFlags = 0 );
	VkCommandBuffer alloc_command_buffer( VulkanContext const&, VkCommandPool );

	Fence create_fence( VulkanContext const&, VkFenceCreateFlags = 0 );
	Semaphore create_semaphore( VulkanContext const& );

    void buffer_barrier(
        VkCommandBuffer,
        VkBuffer,
        VkAccessFlags aSrcAccessMask,
        VkAccessFlags aDstAccessMask,
        VkPipelineStageFlags aSrcStageMask,
        VkPipelineStageFlags aDstStageMask,
        VkDeviceSize aSize = VK_WHOLE_SIZE,
        VkDeviceSize aOffset = 0,
        uint32_t aSrcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
        uint32_t aDstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED
    );


    DescriptorPool create_descriptor_pool(
        VulkanContext const&,
        std::uint32_t aMaxDescriptors = 2048,
        std::uint32_t aMaxSets = 1024
    );

    VkDescriptorSet alloc_desc_set(
        VulkanContext const&,
        VkDescriptorPool,
        VkDescriptorSetLayout
    );

}
#endif // VKUTIL_HPP_9DE3C6CC_921D_46FD_8452_A7F18E276E2A
