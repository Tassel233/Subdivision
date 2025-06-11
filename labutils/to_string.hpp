#ifndef TO_STRING_HPP_929A6F66_319A_4446_BEF4_491805421D28
#define TO_STRING_HPP_929A6F66_319A_4446_BEF4_491805421D28

#include <volk/volk.h>

#include <string>

#include <cstdint>

namespace labutils
{
	std::string to_string( VkResult );
	std::string to_string( VkPhysicalDeviceType );
	std::string to_string( VkDebugUtilsMessageSeverityFlagBitsEXT );

	std::string queue_flags( VkQueueFlags );
	std::string message_type_flags( VkDebugUtilsMessageTypeFlagsEXT );
	std::string memory_heap_flags( VkMemoryHeapFlags );
	std::string memory_property_flags( VkMemoryPropertyFlags );

	std::string driver_version( std::uint32_t aVendorId, std::uint32_t aDriverVersion );
}

#endif // TO_STRING_HPP_929A6F66_319A_4446_BEF4_491805421D28
