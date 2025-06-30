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
	labutils::Buffer lineListsBuffer;


	std::uint32_t indicesCount;
	std::uint32_t lineListsCount;
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

	std::uint32_t vertexCount;
	std::uint32_t edgeCount;
	std::uint32_t faceCount;
};


ColorizedMesh create_triangle_mesh( labutils::VulkanContext const&, labutils::Allocator const& );

ColorizedMesh create_plane_mesh( labutils::VulkanContext const&, labutils::Allocator const& );

ModelMesh create_model_mesh(labutils::VulkanContext const&, labutils::Allocator const&, labutils::GltfModel const&);

SubdivisionMesh create_model_mesh_extended(labutils::VulkanContext const&, labutils::Allocator const&, labutils::GltfModel const&);

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