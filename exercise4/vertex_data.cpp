#include "vertex_data.hpp"

#include <limits>
#include <iostream>
#include <cstring> // for std::memcpy()
#include <iomanip>

#include "../labutils/error.hpp"
#include "../labutils/vkutil.hpp"
#include "../labutils/to_string.hpp"
#include <glm/gtx/string_cast.hpp>
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
	const auto& vertices = aModel.m_vertices;   // std::vector<Vertex>
	const auto& indices = aModel.m_indices;    // std::vector<uint32_t>


	//const auto& vertices = aModel.m_quadVertices;   // std::vector<Vertex>
	//const auto& indices = aModel.m_quadIndices;    // std::vector<uint32_t>
	const auto& lineLists = aModel.m_quadLinelists;    // std::vector<uint32_t>


	std::vector<glm::vec3> vPositions;
	vPositions.reserve(vertices.size());

	for (auto const& v : vertices)
		vPositions.push_back(v.pos);

	std::size_t posBufferSize = vPositions.size() * sizeof(glm::vec3);
	std::size_t indexBufferSize = indices.size() * sizeof(uint32_t);
	std::size_t lineListsBufferSize = lineLists.size() * sizeof(uint32_t);


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
	lut::Buffer lineListsGPU = lut::create_buffer(
		aAllocator,
		lineListsBufferSize,
		VK_BUFFER_USAGE_INDEX_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
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
	lut::Buffer lineStaging = lut::create_buffer(
		aAllocator,
		lineListsBufferSize,
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

	void* lineListsPtr = nullptr;
	if (auto const res = vmaMapMemory(aAllocator.allocator, lineStaging.allocation, &lineListsPtr); VK_SUCCESS != res)
	{
		throw lut::Error("Mapping memory for writing\n"
			"vmaMapMemory() returned %s", lut::to_string(res).c_str());
	}
	std::memcpy(lineListsPtr, lineLists.data(), lineListsBufferSize);
	vmaUnmapMemory(aAllocator.allocator, lineStaging.allocation);

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


	lut::buffer_barrier(
		uploadCmd,
		indexGPU.buffer,
		VK_ACCESS_TRANSFER_WRITE_BIT,
		VK_ACCESS_INDEX_READ_BIT,
		VK_PIPELINE_STAGE_TRANSFER_BIT,
		VK_PIPELINE_STAGE_VERTEX_INPUT_BIT
	);

	// Copy lineLists data
	VkBufferCopy lcopy{};
	lcopy.size = lineListsBufferSize;
	vkCmdCopyBuffer(uploadCmd, lineStaging.buffer, lineListsGPU.buffer, 1, &lcopy);

	lut::buffer_barrier(
		uploadCmd,
		lineListsGPU.buffer,
		VK_ACCESS_TRANSFER_WRITE_BIT,
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
	std::move(lineListsGPU),
	static_cast<uint32_t>(indices.size()),
	static_cast<uint32_t>(lineLists.size()),

	//sizeof(positions) / sizeof(float) / 3  // three floats per position
	};


}


template<class T>
constexpr std::size_t std430_sizeof()
{
	return (sizeof(T) == 12) ? 16 : sizeof(T);
}

SubdivisionMesh create_model_mesh_extended(labutils::VulkanContext const& aContext,labutils::Allocator const& aAllocator, labutils::GltfModel const& aModel) {
	using namespace labutils;

	const uint32_t faceCount = static_cast<uint32_t>(aModel.m_quadFaces.size());

	SubdivisionMesh result{};

	// lambda fuction for read only buffer
	auto upload_vector = [&](auto const& vec, VkBufferUsageFlags usage, Buffer& outBuf)
		{
			using T = std::decay_t<decltype(vec[0])>;

			const std::size_t elemStd430 = std430_sizeof<T>();   // 16B for vec3, 8B for uvec2 ...
			const std::size_t allocSize = vec.size() * elemStd430;

			/* GPU buffer — allocSize */
			Buffer gpuBuf = create_buffer(
				aAllocator,
				allocSize,
				usage | VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
				0,
				VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE
			);

			/* staging buffer — 同样 allocSize！ */
			Buffer staging = create_buffer(
				aAllocator,
				allocSize,
				VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
				VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT
			);

			/* 填充数据（含 16B padding） */
			uint8_t* dst = nullptr;
			vmaMapMemory(aAllocator.allocator, staging.allocation, (void**)&dst);

			for (size_t i = 0; i < vec.size(); ++i)
			{
				std::memcpy(dst + i * elemStd430, &vec[i], sizeof(T));                 // 实际数据
				if constexpr (elemStd430 > sizeof(T))                                  // 填零 padding
					std::memset(dst + i * elemStd430 + sizeof(T), 0, elemStd430 - sizeof(T));
			}
			vmaUnmapMemory(aAllocator.allocator, staging.allocation);

			outBuf = std::move(gpuBuf);
			std::cout << "UPLOAD "
				<< typeid(T).name()
				<< " elem=" << vec.size()
				<< " allocSz=" << allocSize
				<< " stageSz=" << allocSize /* or dataSize */
				<< std::endl;
			return std::make_tuple(std::move(staging), outBuf.buffer, allocSize);      // ★ copy.size = allocSize
		};
	std::vector<std::tuple<Buffer, VkBuffer, std::size_t>> stagingPairs;

	auto& vertices = aModel.get_quad_vertices();
	// modified
	//std::vector<glm::vec3> controlPoints;
	//controlPoints.reserve(vertices.size());
	//for (auto const& v : vertices) controlPoints.push_back(v.pos);
	std::vector<glm::vec4> controlPoints;
	controlPoints.reserve(vertices.size());
	for (auto const& v : vertices)
		controlPoints.emplace_back(v.pos, 0.0f);

	// read only buffer
	auto stageCP = upload_vector(controlPoints, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, result.controlPoints);
	auto stageFQ = upload_vector(aModel.get_quad_faces(), VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, result.quadFaces);
	auto stageEL = upload_vector(aModel.m_edgeList, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, result.edgeList);
	auto stageEF = upload_vector(aModel.m_edgeToFace, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, result.edgeToFace);
	auto stageFEI = upload_vector(aModel.m_faceEdgeIndices, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, result.faceEdgeIndices);
	assert(aModel.m_faceEdgeIndices.size() == faceCount && "Mismatch in faceEdgeIndices");
	auto stageVFCount = upload_vector(aModel.m_vertexFaceCounts, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, result.vertexFaceCounts);
	auto stageVFIndex = upload_vector(aModel.m_vertexFaceIndices, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, result.vertexFaceIndices);
	auto stageVECount = upload_vector(aModel.m_vertexEdgeCounts, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, result.vertexEdgeCounts);
	auto stageVEIndex = upload_vector(aModel.m_vertexEdgeIndices, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, result.vertexEdgeIndices);

	stagingPairs.push_back(std::move(stageCP));
	stagingPairs.push_back(std::move(stageFQ));
	stagingPairs.push_back(std::move(stageEL));
	stagingPairs.push_back(std::move(stageEF));
	stagingPairs.push_back(std::move(stageFEI));
	stagingPairs.push_back(std::move(stageVFCount));
	stagingPairs.push_back(std::move(stageVFIndex));
	stagingPairs.push_back(std::move(stageVECount));
	stagingPairs.push_back(std::move(stageVEIndex));
	
	// lambda fuction for write buffer
	auto allocOutputBuffer = [&](std::size_t count, Buffer& outBuf) {
		// modified
		std::size_t size = count * sizeof(glm::vec4);
		outBuf = create_buffer(
			aAllocator,
			size,
			VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
			0,
			VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE
		);
		};
	auto allocUintBuffer = [&](std::size_t count, Buffer& outBuf) {
		std::size_t size = count * sizeof(uint32_t);
		outBuf = create_buffer(
			aAllocator,
			size,
			VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_VERTEX_BUFFER_BIT,
			0,
			VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE
		);
		};
	// write only buffer
	allocOutputBuffer(aModel.m_quadFaces.size(), result.facePoints);
	allocOutputBuffer(aModel.m_edgeList.size(), result.edgePoints);
	allocOutputBuffer(aModel.m_quadVertices.size(), result.updatedVertices);
	//allocOutputBuffer(aModel.m_quadFacesRaw.size() + aModel.m_edgeList.size() + aModel.m_quadVerticesRaw.size(), result.drawVertices);
	// 加上 VK_BUFFER_USAGE_VERTEX_BUFFER_BIT 用于 vkCmdBindVertexBuffers
	std::size_t drawVertCount = faceCount * 9;
	std::size_t drawVertSize = drawVertCount * sizeof(glm::vec4);

	result.drawVertices = create_buffer(
		aAllocator,
		drawVertSize,
		VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_VERTEX_BUFFER_BIT,
		0,
		VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE
	);



	result.vertexCount = static_cast<uint32_t>(aModel.m_quadVertices.size());
	result.edgeCount = static_cast<uint32_t>(aModel.m_edgeList.size());
	result.faceCount = static_cast<uint32_t>(aModel.m_quadFaces.size());

	std::size_t drawIdxCount = faceCount * 24;
	std::size_t drawIdxSize = drawIdxCount * sizeof(uint32_t);

	result.faceCount = faceCount;

	result.drawIndices = create_buffer(
		aAllocator,
		drawIdxSize,
		VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_INDEX_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
		0,
		VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE
	);


	Fence uploadFence = create_fence(aContext);
	CommandPool pool = create_command_pool(aContext);
	VkCommandBuffer cmdBuf = alloc_command_buffer(aContext, pool.handle);

	VkCommandBufferBeginInfo beginInfo{};
	beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
	vkBeginCommandBuffer(cmdBuf, &beginInfo);

	for (auto& [staging, gpu, size] : stagingPairs) {
		VkBufferCopy copy{};
		copy.size = size;
		vkCmdCopyBuffer(cmdBuf, staging.buffer, gpu, 1, &copy);

		buffer_barrier(
			cmdBuf, gpu,
			VK_ACCESS_TRANSFER_WRITE_BIT,
			VK_ACCESS_SHADER_READ_BIT,
			VK_PIPELINE_STAGE_TRANSFER_BIT,
			VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT
		);
	}

	vkEndCommandBuffer(cmdBuf);

	VkSubmitInfo submitInfo{};
	submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
	submitInfo.commandBufferCount = 1;
	submitInfo.pCommandBuffers = &cmdBuf;
	vkQueueSubmit(aContext.graphicsQueue, 1, &submitInfo, uploadFence.handle);
	vkWaitForFences(aContext.device, 1, &uploadFence.handle, VK_TRUE, UINT64_MAX);

	return result;
}

//void debug_readback_buffer(
//	labutils::VulkanContext const& aContext,
//	labutils::Allocator const& aAllocator,
//	VkQueue queue,
//	labutils::Buffer const& gpuBuffer,
//	std::size_t size,
//	std::string const& label)
//{
//
//
//	// 创建 staging buffer 用于拷贝回 CPU
//	labutils::Buffer readback = create_buffer(
//		aAllocator,
//		size,
//		VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
//		VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT
//	);
//
//	// 分配一次性 command buffer
//	auto pool = create_command_pool(aContext, VK_COMMAND_POOL_CREATE_TRANSIENT_BIT);
//	VkCommandBuffer cmdBuf = alloc_command_buffer(aContext, pool.handle);
//
//	VkCommandBufferBeginInfo beginInfo{};
//	beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
//	vkBeginCommandBuffer(cmdBuf, &beginInfo);
//
//	// 拷贝 GPU → staging
//	VkBufferCopy copy{};
//	copy.size = size;
//	vkCmdCopyBuffer(cmdBuf, gpuBuffer.buffer, readback.buffer, 1, &copy);
//	vkEndCommandBuffer(cmdBuf);
//
//	labutils::Fence fence = create_fence(aContext);
//	VkSubmitInfo submitInfo{};
//	submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
//	submitInfo.commandBufferCount = 1;
//	submitInfo.pCommandBuffers = &cmdBuf;
//	vkQueueSubmit(queue, 1, &submitInfo, fence.handle);
//	vkWaitForFences(aContext.device, 1, &fence.handle, VK_TRUE, UINT64_MAX);
//
//	// 映射数据并打印
//	void* mapped = nullptr;
//	vmaMapMemory(aAllocator.allocator, readback.allocation, &mapped);
//	std::cout << "[DEBUG BUFFER: " << label << "]\n";
//
//	const float* fdata = reinterpret_cast<const float*>(mapped);
//	//for (std::size_t i = 0; i < std::min(size / sizeof(float), size_t(16)); ++i)
//	//	std::cout << fdata[i] << " ";
//	//std::cout << std::endl;
//	std::size_t vec3Count = size / sizeof(glm::vec3);
//	glm::vec3* vecData = reinterpret_cast<glm::vec3*>(mapped);
//	std::cout << "[DEBUG BUFFER: " << label << "]\n";
//	for (std::size_t i = 0; i < vec3Count; ++i)
//		std::cout << "v" << i << ": " << glm::to_string(vecData[i]) << std::endl;
//
//	vmaUnmapMemory(aAllocator.allocator, readback.allocation);
//}


void debug_readback_buffer(
	labutils::VulkanContext const& aContext,
	labutils::Allocator     const& aAllocator,
	VkQueue                         queue,
	labutils::Buffer        const& gpuBuffer,
	std::size_t                     sizeBytes,   // ← 仍传总字节数
	std::string            const& label)
{
	using namespace labutils;

	// ---------- 复制到可读 staging ----------
	Buffer staging = create_buffer(
		aAllocator,
		sizeBytes,
		VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
		VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT);

	auto pool = create_command_pool(aContext, VK_COMMAND_POOL_CREATE_TRANSIENT_BIT);
	VkCommandBuffer cmdBuf = alloc_command_buffer(aContext, pool.handle);

	VkCommandBufferBeginInfo bi{ VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO };
	vkBeginCommandBuffer(cmdBuf, &bi);

	VkBufferCopy copy{ 0, 0, sizeBytes };
	vkCmdCopyBuffer(cmdBuf, gpuBuffer.buffer, staging.buffer, 1, &copy);
	vkEndCommandBuffer(cmdBuf);

	Fence fence = create_fence(aContext);
	VkSubmitInfo si{
		VK_STRUCTURE_TYPE_SUBMIT_INFO, // 1  sType
		nullptr,                       // 2  pNext
		0,                             // 3  waitSemaphoreCount
		nullptr,                       // 4  pWaitSemaphores
		nullptr,                       // 5  pWaitDstStageMask
		1,                             // 6  commandBufferCount
		&cmdBuf,                       // 7  pCommandBuffers
		0,                             // 8  signalSemaphoreCount
		nullptr                        // 9  pSignalSemaphores
	};

	vkQueueSubmit(queue, 1, &si, fence.handle);
	vkWaitForFences(aContext.device, 1, &fence.handle, VK_TRUE, UINT64_MAX);

	// ---------- 映射 ＆ 打印 ----------
	void* mapped = nullptr;
	vmaMapMemory(aAllocator.allocator, staging.allocation, &mapped);

	std::cout << "\n[DEBUG BUFFER: " << label << "]\n";

	struct alignas(16) Vec4 { float x, y, z, w; };   // 与 GPU 写入对齐
	auto* v = reinterpret_cast<const Vec4*>(mapped);
	std::size_t count = sizeBytes / sizeof(Vec4);

	for (std::size_t i = 0; i < count; ++i)
		std::cout << "v" << i << ": ("
		<< v[i].x << ", " << v[i].y << ", " << v[i].z << ")\n";

	vmaUnmapMemory(aAllocator.allocator, staging.allocation);
}

void debug_edge_list(
	labutils::VulkanContext const& ctx,
	labutils::Allocator     const& alloc,
	VkQueue                        queue,
	labutils::Buffer const& edgeBuf,
	std::size_t                    edgeCount)
{
	std::size_t size = edgeCount * sizeof(glm::uvec2);

	// 创建 staging buffer
	labutils::Buffer staging = create_buffer(
		alloc,
		size,
		VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
		VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT);

	// 拷贝 GPU → staging
	auto pool = create_command_pool(ctx, VK_COMMAND_POOL_CREATE_TRANSIENT_BIT);
	VkCommandBuffer cmd = alloc_command_buffer(ctx, pool.handle);

	VkCommandBufferBeginInfo bi{ VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO };
	vkBeginCommandBuffer(cmd, &bi);

	VkBufferCopy copy{ 0, 0, size };
	vkCmdCopyBuffer(cmd, edgeBuf.buffer, staging.buffer, 1, &copy);
	vkEndCommandBuffer(cmd);

	lut::Fence fence = create_fence(ctx);
	VkSubmitInfo si{ VK_STRUCTURE_TYPE_SUBMIT_INFO };
	si.commandBufferCount = 1;
	si.pCommandBuffers = &cmd;
	vkQueueSubmit(queue, 1, &si, fence.handle);
	vkWaitForFences(ctx.device, 1, &fence.handle, VK_TRUE, UINT64_MAX);

	// 映射并打印
	void* mapped = nullptr;
	vmaMapMemory(alloc.allocator, staging.allocation, &mapped);

	auto* edges = reinterpret_cast<const glm::uvec2*>(mapped);
	std::cout << "[DEBUG BUFFER: Edge List]\n";
	for (std::size_t i = 0; i < edgeCount; ++i)
		std::cout << "e" << i << ": (" << edges[i].x << ", " << edges[i].y << ")\n";

	vmaUnmapMemory(alloc.allocator, staging.allocation);
}

void debug_edge_to_face(
	labutils::VulkanContext const& ctx,
	labutils::Allocator     const& alloc,
	VkQueue                        queue,
	labutils::Buffer const& edgeFaceBuf,
	std::size_t                    edgeCount)
{
	std::size_t size = edgeCount * sizeof(glm::uvec2);

	// 创建 staging buffer
	labutils::Buffer staging = create_buffer(
		alloc,
		size,
		VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
		VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT);

	// 拷贝 GPU → staging
	auto pool = create_command_pool(ctx, VK_COMMAND_POOL_CREATE_TRANSIENT_BIT);
	VkCommandBuffer cmd = alloc_command_buffer(ctx, pool.handle);

	VkCommandBufferBeginInfo bi{ VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO };
	vkBeginCommandBuffer(cmd, &bi);

	VkBufferCopy copy{ 0, 0, size };
	vkCmdCopyBuffer(cmd, edgeFaceBuf.buffer, staging.buffer, 1, &copy);
	vkEndCommandBuffer(cmd);

	lut::Fence fence = create_fence(ctx);
	VkSubmitInfo si{ VK_STRUCTURE_TYPE_SUBMIT_INFO };
	si.commandBufferCount = 1;
	si.pCommandBuffers = &cmd;
	vkQueueSubmit(queue, 1, &si, fence.handle);
	vkWaitForFences(ctx.device, 1, &fence.handle, VK_TRUE, UINT64_MAX);

	// 映射并打印
	void* mapped = nullptr;
	vmaMapMemory(alloc.allocator, staging.allocation, &mapped);

	auto* facePairs = reinterpret_cast<const glm::uvec2*>(mapped);
	std::cout << "[DEBUG BUFFER: EdgeToFace]\n";
	for (std::size_t i = 0; i < edgeCount; ++i)
		std::cout << "e" << i << ": (face " << facePairs[i].x
		<< ", face " << facePairs[i].y << ")\n";

	vmaUnmapMemory(alloc.allocator, staging.allocation);
}


void debug_readback_indices(
	labutils::VulkanContext const& aContext,
	labutils::Allocator     const& aAllocator,
	VkQueue                         queue,
	labutils::Buffer        const& gpuBuffer,
	std::size_t                     sizeBytes,   // 依旧传 ▸ 总字节数
	std::string            const& label)       // 标签
{
	using namespace labutils;

	// ---------- 1. 复制到 CPU 可读 staging ----------
	Buffer staging = create_buffer(
		aAllocator,
		sizeBytes,
		VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
		VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT);

	auto pool = create_command_pool(aContext, VK_COMMAND_POOL_CREATE_TRANSIENT_BIT);
	VkCommandBuffer cmdBuf = alloc_command_buffer(aContext, pool.handle);

	VkCommandBufferBeginInfo bi{ VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO };
	vkBeginCommandBuffer(cmdBuf, &bi);

	VkBufferCopy copy{ 0, 0, sizeBytes };
	vkCmdCopyBuffer(cmdBuf, gpuBuffer.buffer, staging.buffer, 1, &copy);
	vkEndCommandBuffer(cmdBuf);

	Fence fence = create_fence(aContext);
	VkSubmitInfo si{ VK_STRUCTURE_TYPE_SUBMIT_INFO };
	si.commandBufferCount = 1;
	si.pCommandBuffers = &cmdBuf;

	vkQueueSubmit(queue, 1, &si, fence.handle);
	vkWaitForFences(aContext.device, 1, &fence.handle, VK_TRUE, UINT64_MAX);

	// ---------- 2. 映射 & 打印 ----------
	void* mapped = nullptr;
	vmaMapMemory(aAllocator.allocator, staging.allocation, &mapped);

	auto* idx = reinterpret_cast<const uint32_t*>(mapped);
	std::size_t indexCount = sizeBytes / sizeof(uint32_t);

	std::cout << "\n[DEBUG INDICES: " << label << "]  ("
		<< indexCount << " uint32)\n";

	for (std::size_t i = 0; i < indexCount; i += 3)
	{
		// 避免尾数不足 3
		if (i + 2 >= indexCount) break;

		std::cout << "t" << std::setw(3) << std::setfill('0') << (i / 3)
			<< ": " << idx[i] << ' '
			<< idx[i + 1] << ' '
			<< idx[i + 2] << '\n';
	}
	std::cout.flush();

	vmaUnmapMemory(aAllocator.allocator, staging.allocation);
}
