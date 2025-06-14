#include "vertex_data.hpp"

#include <limits>

#include <cstring> // for std::memcpy()

#include "../labutils/error.hpp"
#include "../labutils/vkutil.hpp"
#include "../labutils/to_string.hpp"
namespace lut = labutils;




//1. Create on - GPU buffer
//2. Create CPU / host - visible staging buffer
//3. Place data into the staging buffer(std::memcpy)
//4. Record commands to copy / transfer data from the staging buffer to the final on - GPU buffer
//5. Record appropriate buffer barrier for the final on - GPU buffer
//6. Submit commands for execution



ColorizedMesh create_triangle_mesh( labutils::VulkanContext const& aContext, labutils::Allocator const& aAllocator )
{
	// Vertex data
	static float const positions[] = {
		0.0f, -0.8f,
		-0.7f, 0.8f,
		+0.7f, 0.8f
	};
	static float const colors[] = {
		0.f, 0.f, 1.f,
		1.f, 0.f, 0.f,
		0.f, 1.f, 0.f
	};

	// Create final position and color buffers
	lut::Buffer vertexPosGPU = lut::create_buffer(
		aAllocator,
		sizeof(positions),
		VK_BUFFER_USAGE_VERTEX_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
		0, // no additional VmaAllocationCreateFlags
		VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE // or just VMA_MEMORY_USAGE_AUTO
	);

	lut::Buffer vertexColGPU = lut::create_buffer(
		aAllocator,
		sizeof(colors),
		VK_BUFFER_USAGE_VERTEX_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
		0, // no additional VmaAllocationCreateFlags
		VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE // or just VMA_MEMORY_USAGE_AUTO
	);

	// create CPU visible staging buffer
	lut::Buffer posStaging = lut::create_buffer(
		aAllocator,
		sizeof(positions),
		VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
		VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT
	);

	lut::Buffer colStaging = lut::create_buffer(
		aAllocator,
		sizeof(colors),
		VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
		VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT
	);

	// copy data into staging buffer
	void* posPtr = nullptr;
	if (auto const res = vmaMapMemory(aAllocator.allocator, posStaging.allocation, &posPtr); VK_SUCCESS != res)
	{
		throw lut::Error("Mapping memory for writing\n"
			"vmaMapMemory() returned %s", lut::to_string(res).c_str());
	}
	std::memcpy(posPtr, positions, sizeof(positions));
	vmaUnmapMemory(aAllocator.allocator, posStaging.allocation);

	void* colPtr = nullptr;
	if (auto const res = vmaMapMemory(aAllocator.allocator, colStaging.allocation, &colPtr); VK_SUCCESS != res)
	{
		throw lut::Error("Mapping memory for writing\n"
			"vmaMapMemory() returned %s", lut::to_string(res).c_str());
	}
	std::memcpy(colPtr, colors, sizeof(colors));
	vmaUnmapMemory(aAllocator.allocator, colStaging.allocation);

	// transfer data from staging buffer to GPU buffer 
	 
	// Create fence to wait for transfer completion
	lut::Fence uploadComplete = create_fence(aContext);

	// Create dedicated command pool & buffer for upload
	lut::CommandPool uploadPool = create_command_pool(aContext);
	VkCommandBuffer uploadCmd = alloc_command_buffer(aContext, uploadPool.handle);

	// Begin recording
	VkCommandBufferBeginInfo beginInfo{};
	beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
	beginInfo.flags = 0;
	beginInfo.pInheritanceInfo = nullptr;

	if (auto const res = vkBeginCommandBuffer(uploadCmd, &beginInfo); VK_SUCCESS != res)
	{
		throw lut::Error("Beginning command buffer recording\n"
			"vkBeginCommandBuffer() returned %s", lut::to_string(res).c_str());
	}

	// Copy position data
	VkBufferCopy pcopy{};
	pcopy.size = sizeof(positions);
	vkCmdCopyBuffer(uploadCmd, posStaging.buffer, vertexPosGPU.buffer, 1, &pcopy);

	// Barrier to ensure safe read in vertex shader
	lut::buffer_barrier(
		uploadCmd,
		vertexPosGPU.buffer,
		VK_ACCESS_TRANSFER_WRITE_BIT,
		VK_ACCESS_VERTEX_ATTRIBUTE_READ_BIT,
		VK_PIPELINE_STAGE_TRANSFER_BIT,
		VK_PIPELINE_STAGE_VERTEX_INPUT_BIT
	);

	// Copy color data
	VkBufferCopy ccopy{};
	ccopy.size = sizeof(colors);
	vkCmdCopyBuffer(uploadCmd, colStaging.buffer, vertexColGPU.buffer, 1, &ccopy);

	// Barrier for color buffer
	lut::buffer_barrier(
		uploadCmd,
		vertexColGPU.buffer,
		VK_ACCESS_TRANSFER_WRITE_BIT,
		VK_ACCESS_VERTEX_ATTRIBUTE_READ_BIT,
		VK_PIPELINE_STAGE_TRANSFER_BIT,
		VK_PIPELINE_STAGE_VERTEX_INPUT_BIT
	);

	// Finish recording
	if (auto const res = vkEndCommandBuffer(uploadCmd); VK_SUCCESS != res)
	{
		throw lut::Error("Ending command buffer recording\n"
			"vkEndCommandBuffer() returned %s", lut::to_string(res).c_str());
	}

	// Submit transfer commands
	VkSubmitInfo submitInfo{};
	submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
	submitInfo.commandBufferCount = 1;
	submitInfo.pCommandBuffers = &uploadCmd;

	if (auto const res = vkQueueSubmit(aContext.graphicsQueue, 1, &submitInfo, uploadComplete.handle); VK_SUCCESS != res)
	{
		throw lut::Error("Submitting commands\n"
			"vkQueueSubmit() returned %s", lut::to_string(res).c_str());
	}

	// Wait for commands to finish before we destroy the temporary resources required for the transfers
	// (staging buffers, command pool, ...)

	if (auto const res = vkWaitForFences(aContext.device, 1, &uploadComplete.handle, VK_TRUE, std::numeric_limits<std::uint64_t>::max()); VK_SUCCESS != res)
	{
		throw lut::Error("Waiting for upload to complete\n"
			"vkWaitForFences() returned %s", lut::to_string(res).c_str());
	}

	return ColorizedMesh{
	std::move(vertexPosGPU),
	std::move(vertexColGPU),
	sizeof(positions) / sizeof(float) / 2  // two floats per position
	};




}

ColorizedMesh create_plane_mesh(labutils::VulkanContext const& aContext, labutils::Allocator const& aAllocator)
{
	// Vertex data
	static float const positions[] = {
		-1.f, 0.f, -6.f, // v0
		-1.f, 0.f, +6.f, // v1
		+1.f, 0.f, +6.f, // v2

		-1.f, 0.f, -6.f, // v0
		+1.f, 0.f, +6.f, // v2
		+1.f, 0.f, -6.f  // v3
	};

	static float const colors[] = {
		0.4f, 0.4f, 1.0f, // c0
		0.4f, 1.0f, 0.4f, // c1
		1.0f, 0.4f, 0.4f, // c2

		0.4f, 0.4f, 1.0f, // c0
		1.0f, 0.4f, 0.4f, // c2
		1.0f, 0.4f, 0.0f  // c3
	};

	// Create final position and color buffers
	lut::Buffer vertexPosGPU = lut::create_buffer(
		aAllocator,
		sizeof(positions),
		VK_BUFFER_USAGE_VERTEX_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
		0, // no additional VmaAllocationCreateFlags
		VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE // or just VMA_MEMORY_USAGE_AUTO
	);

	lut::Buffer vertexColGPU = lut::create_buffer(
		aAllocator,
		sizeof(colors),
		VK_BUFFER_USAGE_VERTEX_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
		0, // no additional VmaAllocationCreateFlags
		VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE // or just VMA_MEMORY_USAGE_AUTO
	);

	// create CPU visible staging buffer
	lut::Buffer posStaging = lut::create_buffer(
		aAllocator,
		sizeof(positions),
		VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
		VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT
	);

	lut::Buffer colStaging = lut::create_buffer(
		aAllocator,
		sizeof(colors),
		VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
		VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT
	);

	// copy data into staging buffer
	void* posPtr = nullptr;
	if (auto const res = vmaMapMemory(aAllocator.allocator, posStaging.allocation, &posPtr); VK_SUCCESS != res)
	{
		throw lut::Error("Mapping memory for writing\n"
			"vmaMapMemory() returned %s", lut::to_string(res).c_str());
	}
	std::memcpy(posPtr, positions, sizeof(positions));
	vmaUnmapMemory(aAllocator.allocator, posStaging.allocation);

	void* colPtr = nullptr;
	if (auto const res = vmaMapMemory(aAllocator.allocator, colStaging.allocation, &colPtr); VK_SUCCESS != res)
	{
		throw lut::Error("Mapping memory for writing\n"
			"vmaMapMemory() returned %s", lut::to_string(res).c_str());
	}
	std::memcpy(colPtr, colors, sizeof(colors));
	vmaUnmapMemory(aAllocator.allocator, colStaging.allocation);

	// transfer data from staging buffer to GPU buffer 

	// Create fence to wait for transfer completion
	lut::Fence uploadComplete = create_fence(aContext);

	// Create dedicated command pool & buffer for upload
	lut::CommandPool uploadPool = create_command_pool(aContext);
	VkCommandBuffer uploadCmd = alloc_command_buffer(aContext, uploadPool.handle);

	// Begin recording
	VkCommandBufferBeginInfo beginInfo{};
	beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
	beginInfo.flags = 0;
	beginInfo.pInheritanceInfo = nullptr;

	if (auto const res = vkBeginCommandBuffer(uploadCmd, &beginInfo); VK_SUCCESS != res)
	{
		throw lut::Error("Beginning command buffer recording\n"
			"vkBeginCommandBuffer() returned %s", lut::to_string(res).c_str());
	}

	// Copy position data
	VkBufferCopy pcopy{};
	pcopy.size = sizeof(positions);
	vkCmdCopyBuffer(uploadCmd, posStaging.buffer, vertexPosGPU.buffer, 1, &pcopy);

	// Barrier to ensure safe read in vertex shader
	lut::buffer_barrier(
		uploadCmd,
		vertexPosGPU.buffer,
		VK_ACCESS_TRANSFER_WRITE_BIT,
		VK_ACCESS_VERTEX_ATTRIBUTE_READ_BIT,
		VK_PIPELINE_STAGE_TRANSFER_BIT,
		VK_PIPELINE_STAGE_VERTEX_INPUT_BIT
	);

	// Copy color data
	VkBufferCopy ccopy{};
	ccopy.size = sizeof(colors);
	vkCmdCopyBuffer(uploadCmd, colStaging.buffer, vertexColGPU.buffer, 1, &ccopy);

	// Barrier for color buffer
	lut::buffer_barrier(
		uploadCmd,
		vertexColGPU.buffer,
		VK_ACCESS_TRANSFER_WRITE_BIT,
		VK_ACCESS_VERTEX_ATTRIBUTE_READ_BIT,
		VK_PIPELINE_STAGE_TRANSFER_BIT,
		VK_PIPELINE_STAGE_VERTEX_INPUT_BIT
	);

	// Finish recording
	if (auto const res = vkEndCommandBuffer(uploadCmd); VK_SUCCESS != res)
	{
		throw lut::Error("Ending command buffer recording\n"
			"vkEndCommandBuffer() returned %s", lut::to_string(res).c_str());
	}

	// Submit transfer commands
	VkSubmitInfo submitInfo{};
	submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
	submitInfo.commandBufferCount = 1;
	submitInfo.pCommandBuffers = &uploadCmd;

	if (auto const res = vkQueueSubmit(aContext.graphicsQueue, 1, &submitInfo, uploadComplete.handle); VK_SUCCESS != res)
	{
		throw lut::Error("Submitting commands\n"
			"vkQueueSubmit() returned %s", lut::to_string(res).c_str());
	}

	// Wait for commands to finish before we destroy the temporary resources required for the transfers
	// (staging buffers, command pool, ...)

	if (auto const res = vkWaitForFences(aContext.device, 1, &uploadComplete.handle, VK_TRUE, std::numeric_limits<std::uint64_t>::max()); VK_SUCCESS != res)
	{
		throw lut::Error("Waiting for upload to complete\n"
			"vkWaitForFences() returned %s", lut::to_string(res).c_str());
	}

	return ColorizedMesh{
	std::move(vertexPosGPU),
	std::move(vertexColGPU),
	sizeof(positions) / sizeof(float) / 3  // three floats per position
	};

}

ModelMesh create_model_mesh(labutils::VulkanContext const& aContext, labutils::Allocator const& aAllocator, labutils::GltfModel const& aModel)
{

	const auto& vertices = aModel.get_vertices();   // std::vector<Vertex>
	const auto& indices = aModel.get_indices();    // std::vector<uint32_t>

	std::vector<glm::vec3> vPositions;
	vPositions.reserve(vertices.size());

	for (auto const& v : vertices)
		vPositions.push_back(v.pos);

	std::size_t posBufferSize = vPositions.size() * sizeof(glm::vec3);
	std::size_t indexBufferSize = indices.size() * sizeof(uint32_t);

	// Create final position and index buffers
	lut::Buffer vertexPosGPU = lut::create_buffer(
		aAllocator,
		posBufferSize,
		VK_BUFFER_USAGE_VERTEX_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
		0, // no additional VmaAllocationCreateFlags
		VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE // or just VMA_MEMORY_USAGE_AUTO
	);

	lut::Buffer indexGPU = lut::create_buffer(
		aAllocator,
		indexBufferSize,
		VK_BUFFER_USAGE_INDEX_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
		//VK_BUFFER_USAGE_VERTEX_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
		0, // no additional VmaAllocationCreateFlags
		VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE // or just VMA_MEMORY_USAGE_AUTO
	);

	// create CPU visible staging buffer
	lut::Buffer posStaging = lut::create_buffer(
		aAllocator,
		posBufferSize,
		VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
		VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT
	);

	lut::Buffer indexStaging = lut::create_buffer(
		aAllocator,
		indexBufferSize,
		VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
		VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT
	);

	// copy data into staging buffer
	void* posPtr = nullptr;
	if (auto const res = vmaMapMemory(aAllocator.allocator, posStaging.allocation, &posPtr); VK_SUCCESS != res)
	{
		throw lut::Error("Mapping memory for writing\n"
			"vmaMapMemory() returned %s", lut::to_string(res).c_str());
	}
	std::memcpy(posPtr, vPositions.data(), posBufferSize);
	vmaUnmapMemory(aAllocator.allocator, posStaging.allocation);

	void* indexPtr = nullptr;
	if (auto const res = vmaMapMemory(aAllocator.allocator, indexStaging.allocation, &indexPtr); VK_SUCCESS != res)
	{
		throw lut::Error("Mapping memory for writing\n"
			"vmaMapMemory() returned %s", lut::to_string(res).c_str());
	}
	std::memcpy(indexPtr, indices.data(), indexBufferSize);
	vmaUnmapMemory(aAllocator.allocator, indexStaging.allocation);

	// transfer data from staging buffer to GPU buffer 

	// Create fence to wait for transfer completion
	lut::Fence uploadComplete = create_fence(aContext);

	// Create dedicated command pool & buffer for upload
	lut::CommandPool uploadPool = create_command_pool(aContext);
	VkCommandBuffer uploadCmd = alloc_command_buffer(aContext, uploadPool.handle);

	// Begin recording
	VkCommandBufferBeginInfo beginInfo{};
	beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
	beginInfo.flags = 0;
	beginInfo.pInheritanceInfo = nullptr;

	if (auto const res = vkBeginCommandBuffer(uploadCmd, &beginInfo); VK_SUCCESS != res)
	{
		throw lut::Error("Beginning command buffer recording\n"
			"vkBeginCommandBuffer() returned %s", lut::to_string(res).c_str());
	}

	// Copy position data
	VkBufferCopy pcopy{};
	pcopy.size = posBufferSize;
	vkCmdCopyBuffer(uploadCmd, posStaging.buffer, vertexPosGPU.buffer, 1, &pcopy);

	// Barrier to ensure safe read in vertex shader
	lut::buffer_barrier(
		uploadCmd,
		vertexPosGPU.buffer,
		VK_ACCESS_TRANSFER_WRITE_BIT,
		VK_ACCESS_VERTEX_ATTRIBUTE_READ_BIT,
		VK_PIPELINE_STAGE_TRANSFER_BIT,
		VK_PIPELINE_STAGE_VERTEX_INPUT_BIT
	);

	// Copy index data
	VkBufferCopy icopy{};
	icopy.size = indexBufferSize;
	vkCmdCopyBuffer(uploadCmd, indexStaging.buffer, indexGPU.buffer, 1, &icopy);

	// Barrier for color buffer
	lut::buffer_barrier(
		uploadCmd,
		indexGPU.buffer,
		VK_ACCESS_TRANSFER_WRITE_BIT,
		//VK_ACCESS_VERTEX_ATTRIBUTE_READ_BIT,
		VK_ACCESS_INDEX_READ_BIT,
		VK_PIPELINE_STAGE_TRANSFER_BIT,
		VK_PIPELINE_STAGE_VERTEX_INPUT_BIT
	);

	// Finish recording
	if (auto const res = vkEndCommandBuffer(uploadCmd); VK_SUCCESS != res)
	{
		throw lut::Error("Ending command buffer recording\n"
			"vkEndCommandBuffer() returned %s", lut::to_string(res).c_str());
	}

	// Submit transfer commands
	VkSubmitInfo submitInfo{};
	submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
	submitInfo.commandBufferCount = 1;
	submitInfo.pCommandBuffers = &uploadCmd;

	if (auto const res = vkQueueSubmit(aContext.graphicsQueue, 1, &submitInfo, uploadComplete.handle); VK_SUCCESS != res)
	{
		throw lut::Error("Submitting commands\n"
			"vkQueueSubmit() returned %s", lut::to_string(res).c_str());
	}

	// Wait for commands to finish before we destroy the temporary resources required for the transfers
	// (staging buffers, command pool, ...)

	if (auto const res = vkWaitForFences(aContext.device, 1, &uploadComplete.handle, VK_TRUE, std::numeric_limits<std::uint64_t>::max()); VK_SUCCESS != res)
	{
		throw lut::Error("Waiting for upload to complete\n"
			"vkWaitForFences() returned %s", lut::to_string(res).c_str());
	}

	return ModelMesh{
	std::move(vertexPosGPU),
	std::move(indexGPU),
	static_cast<uint32_t>(indices.size())
	//sizeof(positions) / sizeof(float) / 3  // three floats per position
	};


}