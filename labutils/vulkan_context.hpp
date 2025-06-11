#ifndef VULKAN_CONTEXT_HPP_F7F7F8DC_7182_47C0_9456_B748A6DD8070
#define VULKAN_CONTEXT_HPP_F7F7F8DC_7182_47C0_9456_B748A6DD8070
// SOLUTION_TAGS: vulkan-(ex-[^1]|cw-.)

#include <volk/volk.h>

#include <cstdint>

namespace labutils
{
	class VulkanContext
	{
		public:
			VulkanContext(), ~VulkanContext();

			// Move-only
			VulkanContext( VulkanContext const& ) = delete;
			VulkanContext& operator= (VulkanContext const&) = delete;

			VulkanContext( VulkanContext&& ) noexcept;
			VulkanContext& operator= (VulkanContext&&) noexcept;

		public:
			VkInstance instance = VK_NULL_HANDLE;
			VkPhysicalDevice physicalDevice = VK_NULL_HANDLE;


			VkDevice device = VK_NULL_HANDLE;

			std::uint32_t graphicsFamilyIndex = 0;
			VkQueue graphicsQueue = VK_NULL_HANDLE;

			
			//bool haveDebugUtils = false;
			VkDebugUtilsMessengerEXT debugMessenger = VK_NULL_HANDLE;
	};

	VulkanContext make_vulkan_context();
}

#endif // VULKAN_CONTEXT_HPP_F7F7F8DC_7182_47C0_9456_B748A6DD8070
