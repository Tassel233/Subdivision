#include "vertex_data.hpp"

#include <limits>
#include <iostream>
#include <cstring> // for std::memcpy()
#include <iomanip>
#include <cassert>
#include <numeric>
#include <cstring>

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



template<class T>
constexpr std::size_t std430_sizeof()
{
	return (sizeof(T) == 12) ? 16 : sizeof(T);
}




ModelMesh create_model_buffer_tri(lut::VulkanContext const& aContext, lut::Allocator const& aAllocator, lut::GltfModel const& aModel)
{
	const auto& vertices = aModel.m_vertices;   // std::vector<Vertex>
	const auto& indices = aModel.m_indices;    // std::vector<uint32_t>


	//const auto& vertices = aModel.m_quadVertices;   // std::vector<Vertex>
	//const auto& indices = aModel.m_quadIndices;    // std::vector<uint32_t>
	//const auto& lineLists = aModel.m_quadLinelists;    // std::vector<uint32_t>


	std::vector<glm::vec3> vPositions;
	vPositions.reserve(vertices.size());

	for (auto const& v : vertices)
		vPositions.push_back(v.pos);

	std::size_t posBufferSize = vPositions.size() * sizeof(glm::vec3);
	std::size_t indexBufferSize = indices.size() * sizeof(uint32_t);
	//std::size_t lineListsBufferSize = lineLists.size() * sizeof(uint32_t);


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


	lut::buffer_barrier(
		uploadCmd,
		indexGPU.buffer,
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
	static_cast<uint32_t>(indices.size()),
	};


}

SubdivisionMesh create_model_buffer(lut::VulkanContext const& aContext, lut::Allocator const& aAllocator, lut::GltfModel const& aModel)
{



	SubdivisionMesh result{};
	std::vector<std::tuple<lut::Buffer, VkBuffer, std::size_t>> stagingPairs;

	// lambda fuction for buffer
	auto upload_vector = [&](auto const& vec, VkBufferUsageFlags usage, labutils::Buffer& outBuf)
		{
			using T = std::decay_t<decltype(vec[0])>;

			const std::size_t elemStd430 = std430_sizeof<T>();
			const std::size_t allocSize = vec.size() * elemStd430;


			lut::Buffer gpuBuf = create_buffer(
				aAllocator,
				allocSize,
				usage | VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
				0,
				VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE
			);


			lut::Buffer staging = create_buffer(
				aAllocator,
				allocSize,
				VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
				VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT
			);


			uint8_t* dst = nullptr;
			vmaMapMemory(aAllocator.allocator, staging.allocation, (void**)&dst);

			for (size_t i = 0; i < vec.size(); ++i)
			{
				std::memcpy(dst + i * elemStd430, &vec[i], sizeof(T));                 
				if constexpr (elemStd430 > sizeof(T))                                  
					std::memset(dst + i * elemStd430 + sizeof(T), 0, elemStd430 - sizeof(T));
			}
			vmaUnmapMemory(aAllocator.allocator, staging.allocation);

			outBuf = std::move(gpuBuf);
			std::cout << "UPLOAD "
				<< typeid(T).name()
				<< " elem=" << vec.size()
				<< " allocSz=" << allocSize
				<< " stageSz=" << allocSize 
				<< std::endl;
			return std::make_tuple(std::move(staging), outBuf.buffer, allocSize);      
		};


	auto& vertices = aModel.m_quadVertices;
	std::vector<glm::vec4> controlPoints;
	controlPoints.reserve(vertices.size());
	for (auto const& v : vertices)
	{
		controlPoints.emplace_back(v.pos, 0.0f);
	}

	auto stageCP = upload_vector(controlPoints, 0, result.controlPoints);
	auto stageFQ = upload_vector(aModel.get_quad_faces(), 0, result.quadFaces);
	auto stageEL = upload_vector(aModel.m_edgeList, 0, result.edgeList);
	auto stageEF = upload_vector(aModel.m_edgeToFace, 0, result.edgeToFace);
	auto stageFEI = upload_vector(aModel.m_faceEdgeIndices, 0, result.faceEdgeIndices);
	auto stageVFCount = upload_vector(aModel.m_vertexFaceCounts, 0, result.vertexFaceCounts);
	auto stageVFIndex = upload_vector(aModel.m_vertexFaceIndices, 0, result.vertexFaceIndices);
	auto stageVECount = upload_vector(aModel.m_vertexEdgeCounts, 0, result.vertexEdgeCounts);
	auto stageVEIndex = upload_vector(aModel.m_vertexEdgeIndices, 0, result.vertexEdgeIndices);

	auto stagedrawdrawVertices = upload_vector(controlPoints, VK_BUFFER_USAGE_VERTEX_BUFFER_BIT, result.drawVertices);
	auto stagedrawdrawIndices = upload_vector(aModel.m_quadIndices, VK_BUFFER_USAGE_INDEX_BUFFER_BIT, result.drawIndices);
	auto stagedrawLinelists = upload_vector(aModel.m_quadLinelists, VK_BUFFER_USAGE_INDEX_BUFFER_BIT, result.drawLinelists);

	stagingPairs.push_back(std::move(stageCP));
	stagingPairs.push_back(std::move(stageFQ));
	stagingPairs.push_back(std::move(stageEL));
	stagingPairs.push_back(std::move(stageEF));
	stagingPairs.push_back(std::move(stageFEI));
	stagingPairs.push_back(std::move(stageVFCount));
	stagingPairs.push_back(std::move(stageVFIndex));
	stagingPairs.push_back(std::move(stageVECount));
	stagingPairs.push_back(std::move(stageVEIndex));

	stagingPairs.push_back(std::move(stagedrawdrawVertices));
	stagingPairs.push_back(std::move(stagedrawdrawIndices));
	stagingPairs.push_back(std::move(stagedrawLinelists));

	lut::Fence uploadFence = create_fence(aContext);
	lut::CommandPool pool = create_command_pool(aContext);
	VkCommandBuffer cmdBuf = alloc_command_buffer(aContext, pool.handle);

	VkCommandBufferBeginInfo beginInfo{};
	beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
	vkBeginCommandBuffer(cmdBuf, &beginInfo);

	for (auto& [staging, gpu, size] : stagingPairs) {
		VkBufferCopy copy{};
		copy.size = size;
		vkCmdCopyBuffer(cmdBuf, staging.buffer, gpu, 1, &copy);

		// Ensure copy stage finished before using
		lut::buffer_barrier(
			cmdBuf, gpu,
			VK_ACCESS_TRANSFER_WRITE_BIT,
			VK_ACCESS_SHADER_READ_BIT,
			VK_PIPELINE_STAGE_TRANSFER_BIT,
			VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT
		);
	}

	vkEndCommandBuffer(cmdBuf);
	result.facePoints = create_buffer(
		aAllocator,
		aModel.m_quadFaces.size() * sizeof(glm::vec4),
		VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
		0,
		VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE
	);
	result.edgePoints = create_buffer(
		aAllocator,
		aModel.m_edgeList.size() * sizeof(glm::vec4),
		VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
		0,
		VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE
	);
	result.updatedVertices = create_buffer(
		aAllocator,
		aModel.m_quadVertices.size() * sizeof(glm::vec4),
		VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
		0,
		VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE
	);

	VkSubmitInfo submitInfo{};
	submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
	submitInfo.commandBufferCount = 1;
	submitInfo.pCommandBuffers = &cmdBuf;
	vkQueueSubmit(aContext.graphicsQueue, 1, &submitInfo, uploadFence.handle);
	vkWaitForFences(aContext.device, 1, &uploadFence.handle, VK_TRUE, UINT64_MAX);

	result.vertexCount = aModel.m_quadVertices.size();
	result.edgeCount = aModel.m_edgeList.size();
	result.faceCount = aModel.m_quadFaces.size();

	return result;

}

SubdivisionMesh create_empty_buffer(
	lut::VulkanContext const& aContext,
	lut::Allocator const& aAllocator,
	std::size_t vertexCount,
	std::size_t edgeCount,
	std::size_t faceCount)
{
	SubdivisionMesh result{};

	

	auto alloc = [&](std::size_t size, VkBufferUsageFlags usage, labutils::Buffer& outBuf)
		{
			outBuf = create_buffer(
				aAllocator,
				size,
				usage | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
				0,
				VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE
			);
		};



	// === Geometry buffers ===
	alloc((vertexCount + edgeCount + faceCount) * sizeof(glm::vec4), NULL, result.controlPoints);
	alloc(4 * faceCount * sizeof(glm::uvec4), NULL, result.quadFaces);
	alloc((edgeCount * 2 + 4 * faceCount) * sizeof(glm::uvec2), NULL, result.edgeList);
	alloc((edgeCount * 2 + 4 * faceCount) * sizeof(glm::uvec2), NULL, result.edgeToFace);
	alloc(4 * faceCount * sizeof(glm::uvec4), NULL, result.faceEdgeIndices);
	//allocVertex(vertexCount, result.controlPoints);          // glm::vec4
	//allocUvec4(faceCount, result.quadFaces);                 // glm::uvec4
	//allocUvec2(edgeCount, result.edgeList);                  // glm::uvec2
	//allocUvec2(edgeCount * 2, result.edgeToFace);            // 2 faces per edge
	//allocUvec4(faceCount, result.faceEdgeIndices);           // 4 edges per face

	alloc((vertexCount + edgeCount + faceCount) * sizeof(uint32_t), NULL, result.vertexFaceCounts);
	alloc(16 * faceCount * sizeof(uint32_t), NULL, result.vertexFaceIndices);
	alloc((vertexCount + edgeCount + faceCount) * sizeof(uint32_t), NULL, result.vertexEdgeCounts);
	alloc(2 * (edgeCount * 2 + 4 * faceCount) * sizeof(uint32_t), NULL, result.vertexEdgeIndices);

	//allocUint(vertexCount, result.vertexFaceCounts);
	//allocUint(vertexCount * 4, result.vertexFaceIndices);    // max 4 faces per vertex
	//allocUint(vertexCount, result.vertexEdgeCounts);
	//allocUint(vertexCount * 4, result.vertexEdgeIndices);    // max 4 edges per vertex

	// === Compute outputs ===
	alloc(4 * faceCount * sizeof(glm::vec4), NULL, result.facePoints);
	alloc((edgeCount * 2 + 4 * faceCount) * sizeof(glm::vec4), NULL, result.edgePoints);
	alloc((vertexCount + edgeCount + faceCount) * sizeof(glm::vec4), NULL, result.updatedVertices);
	//allocVertex(faceCount, result.facePoints);               // glm::vec4
	//allocVertex(edgeCount, result.edgePoints);               // glm::vec4
	//allocVertex(vertexCount, result.updatedVertices);        // glm::vec4

	// === Drawing buffers ===
	alloc((vertexCount + edgeCount + faceCount) * sizeof(glm::vec4), VK_BUFFER_USAGE_VERTEX_BUFFER_BIT, result.drawVertices);
	alloc(faceCount * 24 * sizeof(uint32_t), VK_BUFFER_USAGE_INDEX_BUFFER_BIT, result.drawIndices);
	alloc((edgeCount * 2 + 4 * faceCount) * 2 * sizeof(uint32_t), VK_BUFFER_USAGE_INDEX_BUFFER_BIT, result.drawLinelists);
	//allocVertex(vertexCount + edgeCount + faceCount, result.drawVertices);  // conservative overalloc
	//allocUint(faceCount * 6, result.drawIndices);           // each quad becomes 6 indices
	//allocUint(edgeCount * 2, result.drawLinelists);         // 1 line = 2 indices


	return result;
}


SubdivisionMesh create_model_mesh_extended(labutils::VulkanContext const& aContext,labutils::Allocator const& aAllocator, labutils::GltfModel const& aModel) {
	using namespace labutils;

	const uint32_t faceCount = static_cast<uint32_t>(aModel.m_quadFaces.size());

	SubdivisionMesh result{};

	// lambda fuction for buffer
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
	auto stageVFCount = upload_vector(aModel.m_vertexFaceCounts, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, result.vertexFaceCounts);
	auto stageVFIndex = upload_vector(aModel.m_vertexFaceIndices, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, result.vertexFaceIndices);
	auto stageVECount = upload_vector(aModel.m_vertexEdgeCounts, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, result.vertexEdgeCounts);
	auto stageVEIndex = upload_vector(aModel.m_vertexEdgeIndices, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, result.vertexEdgeIndices);
	auto stagedrawVertices = upload_vector(aModel.m_quadLinelists, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_INDEX_BUFFER_BIT, result.drawLinelists);

	auto stageLinelists = upload_vector(aModel.m_quadLinelists, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_INDEX_BUFFER_BIT, result.drawLinelists);

	stagingPairs.push_back(std::move(stageCP));
	stagingPairs.push_back(std::move(stageFQ));
	stagingPairs.push_back(std::move(stageEL));
	stagingPairs.push_back(std::move(stageEF));
	stagingPairs.push_back(std::move(stageFEI));
	stagingPairs.push_back(std::move(stageVFCount));
	stagingPairs.push_back(std::move(stageVFIndex));
	stagingPairs.push_back(std::move(stageVECount));
	stagingPairs.push_back(std::move(stageVEIndex));
	stagingPairs.push_back(std::move(stageLinelists));

	// lambda fuction for write buffer
	auto allocOutputBuffer = [&](std::size_t count, Buffer& outBuf)
		{
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
	auto allocUintBuffer = [&](std::size_t count, Buffer& outBuf)
		{
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

void debug_readback_buffer(
	labutils::VulkanContext const& aContext,
	labutils::Allocator     const& aAllocator,
	VkQueue                         queue,
	labutils::Buffer        const& gpuBuffer,
	std::size_t                     sizeBytes,
	std::string            const& label)
{
	using namespace labutils;

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
		VK_STRUCTURE_TYPE_SUBMIT_INFO, 
		nullptr,                       
		0,                             
		nullptr,                       
		nullptr,                       
		1,                             
		&cmdBuf,                       
		0,                             
		nullptr                        
	};

	vkQueueSubmit(queue, 1, &si, fence.handle);
	vkWaitForFences(aContext.device, 1, &fence.handle, VK_TRUE, UINT64_MAX);

	// ---------- 映射 ＆ 打印 ----------
	void* mapped = nullptr;
	vmaMapMemory(aAllocator.allocator, staging.allocation, &mapped);

	std::cout << "\n[DEBUG BUFFER: " << label << "]\n";

	struct alignas(16) Vec4 { float x, y, z, w; };
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

	labutils::Buffer staging = create_buffer(
		alloc,
		size,
		VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
		VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT);


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


	labutils::Buffer staging = create_buffer(
		alloc,
		size,
		VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
		VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT);


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
	std::size_t                     sizeBytes,
	std::string            const& label)
{
	using namespace labutils;


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


	void* mapped = nullptr;
	vmaMapMemory(aAllocator.allocator, staging.allocation, &mapped);

	auto* idx = reinterpret_cast<const uint32_t*>(mapped);
	std::size_t indexCount = sizeBytes / sizeof(uint32_t);

	std::cout << "\n[DEBUG INDICES: " << label << "]  ("
		<< indexCount << " uint32)\n";

	for (std::size_t i = 0; i < indexCount; i += 3)
	{

		if (i + 2 >= indexCount) break;

		std::cout << "t" << std::setw(3) << std::setfill('0') << (i / 3)
			<< ": " << idx[i] << ' '
			<< idx[i + 1] << ' '
			<< idx[i + 2] << '\n';
	}
	std::cout.flush();

	vmaUnmapMemory(aAllocator.allocator, staging.allocation);
}

void debug_readback_edge_counts(
	labutils::VulkanContext const& aContext,
	labutils::Allocator     const& aAllocator,
	VkQueue                         queue,
	labutils::Buffer        const& gpuBuffer,
	std::size_t                     sizeBytes,
	std::string            const& label)
{
	using namespace labutils;


	Buffer staging = create_buffer(
		aAllocator,
		sizeBytes,
		VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
		VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT
	);

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


	void* mapped = nullptr;
	vmaMapMemory(aAllocator.allocator, staging.allocation, &mapped);

	const uint32_t* counts = reinterpret_cast<const uint32_t*>(mapped);
	const std::size_t vertexCount = sizeBytes / sizeof(uint32_t);

	std::cout << "\n[DEBUG VERTEX-EDGE-COUNTS: " << label << "]  ("
		<< vertexCount << " vertices)\n";

	uint32_t sum = 0, minC = UINT32_MAX, maxC = 0;
	for (std::size_t v = 0; v < vertexCount; ++v)
	{
		uint32_t c = counts[v];
		std::cout << "v" << std::setw(4) << std::setfill('0') << v
			<< ": " << c << '\n';

		sum += c;
		minC = std::min(minC, c);
		maxC = std::max(maxC, c);
	}

	std::cout << "-- summary: min = " << minC
		<< ", max = " << maxC
		<< ", sum = " << sum << '\n' << std::flush;

	vmaUnmapMemory(aAllocator.allocator, staging.allocation);
}

void debug_readback_linelist(
	labutils::VulkanContext const& aContext,
	labutils::Allocator     const& aAllocator,
	VkQueue                         queue,
	labutils::Buffer        const& gpuBuffer,
	std::size_t                     sizeBytes,
	std::string            const& label)
{
	using namespace labutils;


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


	void* mapped = nullptr;
	vmaMapMemory(aAllocator.allocator, staging.allocation, &mapped);

	auto* idx = reinterpret_cast<const uint32_t*>(mapped);
	std::size_t indexCount = sizeBytes / sizeof(uint32_t);

	std::cout << "\n[DEBUG LINES: " << label << "]  ("
		<< indexCount << " uint32)\n";

	for (std::size_t i = 0; i + 1 < indexCount; i += 2)
	{
		std::cout << "l" << std::setw(3) << std::setfill('0') << (i / 2)
			<< ": " << idx[i] << ' ' << idx[i + 1] << '\n';
	}

	std::cout.flush();

	vmaUnmapMemory(aAllocator.allocator, staging.allocation);
}
