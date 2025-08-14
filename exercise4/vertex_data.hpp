#pragma once

#include <cstdint>

#include "../labutils/vulkan_context.hpp"

#include "../labutils/vkbuffer.hpp"
#include "../labutils/allocator.hpp" 
#include "../labutils/gltf_model.hpp"


struct ColorizedMesh
{
	labutils::Buffer positions;
	labutils::Buffer colors;

	std::uint32_t vertexCount;
};

struct ModelMesh
{
	labutils::Buffer posBuffer;
	labutils::Buffer indexBuffer;


	std::uint32_t indicesCount;
};

struct SubdivisionMesh
{
	labutils::Buffer controlPoints;
	labutils::Buffer quadFaces;
	labutils::Buffer edgeList;
	labutils::Buffer edgeToFace;

	labutils::Buffer vertexFaceCounts;
	labutils::Buffer vertexFaceIndices;
	labutils::Buffer vertexEdgeCounts;
	labutils::Buffer vertexEdgeIndices;
	labutils::Buffer faceEdgeIndices;

	labutils::Buffer facePoints;
	labutils::Buffer edgePoints;
	labutils::Buffer updatedVertices;


	// For rendering
	labutils::Buffer drawVertices;
	labutils::Buffer drawIndices;
	labutils::Buffer drawLinelists;


	std::uint32_t vertexCount;
	std::uint32_t edgeCount;
	std::uint32_t faceCount;
	bool isValid() const { return drawVertices.buffer != VK_NULL_HANDLE; }
	void destroy(labutils::Allocator const& alloc)
	{
		auto d = alloc.allocator;        // VmaAllocator
		auto free = [&](labutils::Buffer& b) {
			if (b.buffer != VK_NULL_HANDLE)
				vmaDestroyBuffer(d, b.buffer, b.allocation);
			b.buffer = VK_NULL_HANDLE;
			b.allocation = VK_NULL_HANDLE;
			};
		free(controlPoints);   free(quadFaces);   free(edgeList);
		free(edgeToFace);      free(faceEdgeIndices);
		free(vertexFaceCounts); free(vertexFaceIndices);
		free(vertexEdgeCounts); free(vertexEdgeIndices);
		free(drawVertices);    free(drawIndices); free(drawLinelists);
		free(facePoints);      free(edgePoints);  free(updatedVertices);
		vertexCount = edgeCount = faceCount = 0;
	}
};


ModelMesh create_model_buffer_tri(labutils::VulkanContext const&, labutils::Allocator const&, labutils::GltfModel const&);

SubdivisionMesh create_model_buffer(labutils::VulkanContext const&, labutils::Allocator const&, labutils::GltfModel const&);
SubdivisionMesh create_model_mesh_extended(labutils::VulkanContext const&, labutils::Allocator const&, labutils::GltfModel const&);
SubdivisionMesh create_empty_buffer(labutils::VulkanContext const&, labutils::Allocator const&, std::size_t , std::size_t , std::size_t );


//void debug_readback_buffer(labutils::VulkanContext const& aContext, labutils::Allocator const& aAllocator, VkQueue queue, labutils::Buffer const& gpuBuffer, std::size_t size, std::string label);
void debug_readback_buffer(
	labutils::VulkanContext const& aContext,
	labutils::Allocator const& aAllocator,
	VkQueue queue,
	labutils::Buffer const& gpuBuffer,
	std::size_t size,
	std::string const& label);

void debug_edge_list(
	labutils::VulkanContext const& ctx,
	labutils::Allocator     const& alloc,
	VkQueue                        queue,
	labutils::Buffer const& edgeBuf,
	std::size_t                    edgeCount);

void debug_edge_to_face(
	labutils::VulkanContext const& ctx,
	labutils::Allocator     const& alloc,
	VkQueue                        queue,
	labutils::Buffer const& edgeFaceBuf,
	std::size_t                    edgeCount);

void debug_readback_indices(
	labutils::VulkanContext const& aContext,
	labutils::Allocator     const& aAllocator,
	VkQueue                         queue,
	labutils::Buffer        const& gpuBuffer,
	std::size_t                     sizeBytes,
	std::string            const& label);

void debug_readback_linelist(
	labutils::VulkanContext const& aContext,
	labutils::Allocator     const& aAllocator,
	VkQueue                         queue,
	labutils::Buffer        const& gpuBuffer,
	std::size_t                     sizeBytes,   // 总字节数
	std::string            const& label);

void debug_readback_edge_counts(
	labutils::VulkanContext const& aContext,
	labutils::Allocator     const& aAllocator,
	VkQueue                         queue,
	labutils::Buffer        const& gpuBuffer,
	std::size_t                     sizeBytes,   // 元素总字节数
	std::string            const& label);